"""B2: per-frame set attention over players and ball, a bidirectional GRU over time and a
linear-chain CRF over H / A / D.

Design (literature review, approved before implementation): the social module is what matters
most in PathCRF's ablation (team-wise plus all-agent attention, as in Ball Radar); the temporal
module type matters little there (Bi-LSTM ~ attention), so a GRU is used for CPU cost; attending
across players before time follows SoccerNet 2026 player-centric spotting. Tokens carry no
identity across frames (tokens.py), so fragmenting tracks cannot break the model.

Outputs are P(alive) and P(home | alive), like B1, so the same decoder sweep applies. Losses:
DFL windows: CRF negative log-likelihood over the three states plus auxiliary cross-entropies on
both heads (PathCRF found auxiliary losses essential for CRF training); SkillCorner windows: the
team head on live frames only (its dead-ball label is approximate), with half the total weight.
"""

import numpy as np
import torch
from torch import nn

from .tokens import MAX_PER_TEAM, N_BALL_FEATS, N_PLAYER_FEATS

H, A, D, X = 0, 1, 2, 3

# PyTorch's inference fast path of TransformerEncoderLayer (nested tensors) returns NaN on frames
# where every player slot is padded (SkillCorner, broadcast replays); training mode is unaffected.
# Disabling it makes evaluation compute exactly what training computes.
torch.backends.mha.set_fastpath_enabled(False)


class PooledAttention(nn.Module):
    """Every token attends to the ball and to one summary (masked mean) per team only: linear in
    the number of players, and no player is dropped."""

    def __init__(self, d, heads):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, z, present):
        M = MAX_PER_TEAM
        h = self.norm(z)
        keys = [h[:, -1:]]
        for t in range(2):
            m = present[:, t * M:(t + 1) * M, None].float()
            keys.append((h[:, t * M:(t + 1) * M] * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1))
        kv = torch.cat(keys, 1)
        z = z + self.att(h, kv, kv, need_weights=False)[0]
        return z + self.ff(z)


class SetEncoder(nn.Module):
    """Players of both teams and the ball, one frame at a time.

    social = "full": team-wise then all-token attention over every player (PathCRF / Ball Radar);
             "knear": the same over the K_NEAR players nearest to the reference point only
                      (tokens.reference_point: the ball, its last known position, or the centroid);
             "pooled": each token attends to the ball and to one summary per team (linear cost).
    """

    def __init__(self, d, heads, n_ctx, social="full", k_near=8, n_player_feats=N_PLAYER_FEATS):
        super().__init__()
        self.social, self.k = social, k_near
        # nearest-player selection with ties (float16 distances, absent slots) broken by slot index:
        # topk breaks ties differently on different CPUs (AVX2 vs AVX-512), which changed the
        # submitted file between machines; a stable sort gives the same selection everywhere
        self.stable_topk = True
        self.player = nn.Sequential(nn.Linear(n_player_feats, d), nn.GELU(), nn.Linear(d, d))
        self.ball = nn.Sequential(nn.Linear(N_BALL_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.kind = nn.Embedding(3, d)  # home player, away player, ball
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True,
                                                   norm_first=True)
        if social == "pooled":
            self.layers = nn.ModuleList([PooledAttention(d, heads), PooledAttention(d, heads)])
        else:
            self.intra = layer()
            self.full = layer()
        team = torch.cat([torch.zeros(MAX_PER_TEAM), torch.ones(MAX_PER_TEAM), torch.full((1,), 2.0)])
        # intra-team block: a token sees its own team only (the ball sees itself)
        self.register_buffer("intra_mask", team[:, None] != team[None, :])
        self.register_buffer("kinds", team.long())
        self.out = nn.Linear(5 * d + n_ctx, d)

    def forward(self, players, ball, ctx, dref):
        # players (N, 2, M, F), ball (N, Fb), ctx (N, C), dref (N, 2, M) -> (N, d)
        N = players.shape[0]
        M = MAX_PER_TEAM
        p = self.player(players.reshape(N, 2 * M, -1))
        b = self.ball(ball)[:, None]
        z = torch.cat([p, b], 1) + self.kind(self.kinds)[None]
        present = torch.cat([players[..., 8].reshape(N, -1) > 0.5,
                             torch.ones(N, 1, dtype=torch.bool, device=z.device)], 1)
        if self.social == "pooled":
            for lay in self.layers:
                z = lay(z, present)
        elif self.social == "full":
            pad = ~present
            z = self.intra(z, src_mask=self.intra_mask, src_key_padding_mask=pad)
            z = self.full(z, src_key_padding_mask=pad)
        else:
            # keep the K nearest players of both teams (plus the ball), attend among them only
            key = dref.reshape(N, -1).float().masked_fill(~present[:, :-1], 1e4)
            if self.stable_topk:
                idx = torch.sort(key, dim=1, stable=True).indices[:, :self.k]
            else:
                idx = key.topk(self.k, dim=1, largest=False).indices
            sel = torch.cat([idx, torch.full((N, 1), 2 * M, dtype=torch.long)], 1)
            zs = z.gather(1, sel[..., None].expand(-1, -1, z.shape[2]))
            ps = present.gather(1, sel)
            team = self.kinds[sel]
            mask = (team[:, :, None] != team[:, None, :]).repeat_interleave(self.intra.self_attn.num_heads, 0)
            zs = self.intra(zs, src_mask=mask, src_key_padding_mask=~ps)
            zs = self.full(zs, src_key_padding_mask=~ps)
            z = z.scatter(1, sel[..., None].expand(-1, -1, z.shape[2]), zs)
            present = torch.zeros_like(present).scatter(1, sel, ps)
        pooled = [z[:, -1]]  # the ball token after attention
        for t in range(2):
            zt = z[:, t * M:(t + 1) * M]
            mt = present[:, t * M:(t + 1) * M, None].float()
            cnt = mt.sum(1).clamp(min=1)
            pooled.append((zt * mt).sum(1) / cnt)
            pooled.append((zt + (mt - 1) * 1e4).max(1).values * (cnt > 0).float())
        return self.out(torch.cat(pooled + [ctx], 1))


class TCN(nn.Module):
    """Dilated residual temporal convolutions, one MS-TCN stage (Farha & Gall 2019): non-causal,
    dilations 1..2^(L-1), so the receptive field covers 2^L frames; parallel in time."""

    def __init__(self, d_in, d, layers=9):
        super().__init__()
        self.inp = nn.Conv1d(d_in, d, 1)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, padding=2 ** k, dilation=2 ** k)
                                    for k in range(layers)])
        self.mix = nn.ModuleList([nn.Conv1d(d, d, 1) for _ in range(layers)])

    def forward(self, z):
        h = self.inp(z.transpose(1, 2))
        for conv, mix in zip(self.convs, self.mix):
            h = h + mix(nn.functional.gelu(conv(h)))
        return h.transpose(1, 2)


class NormTCN(nn.Module):
    """TCN with a pre-normalised residual block (LayerNorm over channels) and 1/sqrt(L)-scaled
    branches: the plain TCN above diverged at lr 2e-3 (training loss up to 23, B2 E3)."""

    def __init__(self, d_in, d, layers=9):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, padding=2 ** k, dilation=2 ** k)
                                    for k in range(layers)])
        self.mix = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.scale = layers ** -0.5
        self.out = nn.LayerNorm(d)

    def forward(self, z):
        h = self.inp(z)
        for norm, conv, mix in zip(self.norms, self.convs, self.mix):
            c = conv(norm(h).transpose(1, 2)).transpose(1, 2)
            h = h + self.scale * mix(nn.functional.gelu(c))
        return self.out(h)


class B2(nn.Module):
    def __init__(self, n_ctx, d=64, heads=4, hidden=96, init_costs=(2.5, 6.0, 6.0), social="full",
                 k_near=8, temporal="gru", change_head=False, n_player_feats=N_PLAYER_FEATS,
                 ball_head=False):
        super().__init__()
        self.enc = SetEncoder(d, heads, n_ctx, social=social, k_near=k_near,
                              n_player_feats=n_player_feats)
        self.temporal = temporal
        if temporal == "gru":
            self.gru = nn.GRU(d, hidden, num_layers=2, batch_first=True, bidirectional=True)
        elif temporal == "tcn_norm":
            self.tcn = NormTCN(d, 2 * hidden)
        else:
            self.tcn = TCN(d, 2 * hidden)
        # alive logit, home-given-alive logit (+ "possession changes here" logit, auxiliary only)
        # (+ ball x, y in the attacking frame, scaled as the ball token: F1, auxiliary only; always
        # the last two outputs)
        self.head = nn.Linear(2 * hidden, (3 if change_head else 2) + (2 if ball_head else 0))
        c_team, c_in, c_out = init_costs
        T = torch.zeros(3, 3)
        T[H, A] = T[A, H] = -c_team
        T[H, D] = T[A, D] = -c_in
        T[D, H] = T[D, A] = -c_out
        self.trans = nn.Parameter(T)

    def logits(self, players, ball, ctx, dref, chunk=4096):
        # players (B, T, 2, M, F) -> (B, T, 2)
        Bn, T = players.shape[:2]
        flat = lambda x: x.reshape(Bn * T, *x.shape[2:])
        P, Bl, C, R = flat(players), flat(ball), flat(ctx), flat(dref)
        z = torch.cat([self.enc(P[i:i + chunk], Bl[i:i + chunk], C[i:i + chunk], R[i:i + chunk])
                       for i in range(0, Bn * T, chunk)])
        z = z.reshape(Bn, T, -1)
        h = self.gru(z)[0] if self.temporal == "gru" else self.tcn(z)
        return self.head(h)

    @staticmethod
    def emissions(lg):
        la = nn.functional.logsigmoid(lg[..., 0])
        ld = nn.functional.logsigmoid(-lg[..., 0])
        lh = nn.functional.logsigmoid(lg[..., 1])
        lw = nn.functional.logsigmoid(-lg[..., 1])
        return torch.stack([la + lh, la + lw, ld], -1)

    def crf_nll(self, em, y):
        """Mean per-frame negative log-likelihood of label sequences y (B, T) in 0..2."""
        Bn, T, _ = em.shape
        alpha = em[:, 0]
        for t in range(1, T):
            alpha = torch.logsumexp(alpha[:, :, None] + self.trans[None], 1) + em[:, t]
        logz = torch.logsumexp(alpha, 1)
        score = em.gather(2, y[..., None])[..., 0].sum(1)
        score = score + self.trans[y[:, :-1], y[:, 1:]].sum(1)
        return ((logz - score) / T).mean()


def set_threads(n):
    torch.set_num_threads(n)
    torch.use_deterministic_algorithms(True)


def to_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
