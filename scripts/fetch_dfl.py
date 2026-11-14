"""Download the seven IDSSE / Sportec open DFL matches and store them in the common format.

Source: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data 12:195, CC-BY 4.0. kloppy fetches the files.

    python scripts/fetch_dfl.py                # all matches -> data/external/dfl_v3/<id>.npz
    python scripts/fetch_dfl.py J03WMX J03WN1  # a subset
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import kloppy.sportec as sportec  # noqa: E402

from poss.match import from_kloppy  # noqa: E402

# Params
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
OUT_DIR = os.path.join(ROOT, "data", "external", "dfl_v3")


def tracab_to_metres(x, y):
    # Tracab: centimetres, origin at the centre spot, y towards the top touchline.
    return x / 100.0, y / 100.0


def main(ids):
    os.makedirs(OUT_DIR, exist_ok=True)
    for mid in ids:
        out = os.path.join(OUT_DIR, f"{mid}.npz")
        if os.path.exists(out):
            print(mid, "exists")
            continue
        t0 = time.time()
        ds = sportec.load_open_tracking_data(mid, coordinates="tracab")
        m = from_kloppy(ds, mid, "dfl", tracab_to_metres)
        m.save(out)
        print(mid, m.n_frames, "frames", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or MATCHES)
