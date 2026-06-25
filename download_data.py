"""Download the Bonn EEG epilepsy dataset into ./data/.

The dataset (five sets Z, O, N, F, S) is distributed as five ZIP files by the
University of Bonn Department of Epileptology. Because hosting URLs change over
time, set the BONN_BASE_URL below (or the environment variable of the same
name) to the location that currently hosts the files, then run:

    python download_data.py

Each archive unzips to a folder of 100 *.txt files (one EEG segment each).
After running, ./data/ should contain folders Z/ O/ N/ F/ S/.

Reference:
    R.G. Andrzejak et al., "Indications of nonlinear deterministic and
    finite-dimensional structures in time series of brain electrical activity",
    Phys. Rev. E 64, 061907 (2001).
"""
import os
import sys
import zipfile
import urllib.request

# Update this to the current host of the Bonn ZIP files (Z.zip, O.zip, ...).
BONN_BASE_URL = os.environ.get("BONN_BASE_URL", "")
SETS = ["Z", "O", "N", "F", "S"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    if not BONN_BASE_URL:
        print(__doc__)
        print("ERROR: set BONN_BASE_URL (env var or edit this file) first.")
        sys.exit(1)
    os.makedirs(DATA_DIR, exist_ok=True)
    for s in SETS:
        url = f"{BONN_BASE_URL.rstrip('/')}/{s}.zip"
        zpath = os.path.join(DATA_DIR, f"{s}.zip")
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(DATA_DIR, s))
        os.remove(zpath)
        print(f"  -> data/{s}/")
    print("Done.")


if __name__ == "__main__":
    main()
