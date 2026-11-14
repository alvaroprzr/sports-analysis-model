"""Convert the SkillCorner open data matches to the common format.

Source: github.com/SkillCorner/opendata (MIT), pinned commit below. The tracking files are Git LFS
objects; download them first (each is checked against the sha256 in its LFS pointer):

    git clone --depth 1 https://github.com/SkillCorner/opendata.git data/external/skillcorner_raw/repo
    python scripts/fetch_skillcorner.py --lfs   # fetch the LFS tracking files
    python scripts/fetch_skillcorner.py         # -> data/external/skillcorner_v3/<id>.npz
"""

import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from poss import skillcorner  # noqa: E402

# Params
COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
RAW_DIR = os.path.join(ROOT, "data", "external", "skillcorner_raw", "repo", "data", "matches")
OUT_DIR = os.path.join(ROOT, "data", "external", "skillcorner_v3")
LFS_URL = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/matches/{}/{}"


def fetch_lfs(mid):
    name = f"{mid}_tracking_extrapolated.jsonl"
    path = os.path.join(RAW_DIR, mid, name)
    head = open(path, "rb").read(200)
    if not head.startswith(b"version https://git-lfs"):
        return
    oid = [s.split(b":")[1] for s in head.split(b"\n") if s.startswith(b"oid")][0].decode()
    data = urllib.request.urlopen(LFS_URL.format(COMMIT, mid, name)).read()
    if hashlib.sha256(data).hexdigest() != oid:
        raise ValueError(f"{mid}: sha256 mismatch")
    open(path, "wb").write(data)


def main(lfs):
    os.makedirs(OUT_DIR, exist_ok=True)
    for mid in sorted(os.listdir(RAW_DIR)):
        if lfs:
            fetch_lfs(mid)
            continue
        out = os.path.join(OUT_DIR, f"{mid}.npz")
        if not os.path.exists(out):
            m = skillcorner.load(os.path.join(RAW_DIR, mid))
            m.save(out)
            print(mid, m.n_frames, "frames", flush=True)


if __name__ == "__main__":
    main("--lfs" in sys.argv)
