#!/usr/bin/env python3
"""One-time, dev-time download of the pretrained Kraken base recognition model (the app itself never goes online).

Usage: python scripts/get_base_model.py [--doi 10.5281/zenodo.19238205] [--dest data/models/base]

Default: "Rev City Handwritten English Base Model" (Apache-2.0, kraken recognition model, English handwriting,
~16 MB) from the HTRMoPo repository (Zenodo). Files land in <dest>/<name>/ together with a BASE_MODEL.json record
(doi, license, sha256) that the app reads to label the base model. `make base-model` calls this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DOI = "10.5281/zenodo.19238205"
DEFAULT_NAME = "revcity-english-handwriting"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doi", default=DEFAULT_DOI)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--dest", type=Path, default=REPO / "data" / "models" / "base")
    args = ap.parse_args(argv)

    from htrmopo import get_model
    from kraken import repo

    target = args.dest / args.name
    existing = target / "BASE_MODEL.json"
    if existing.is_file():
        print(f"already present: {target} (delete it to download again)")
        return 0
    desc = repo.get_description(args.doi)
    tmp = args.dest / (args.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.doi}: {desc.summary}")
    get_model(args.doi, path=tmp)
    models = sorted(tmp.rglob("*.mlmodel")) + sorted(tmp.rglob("*.safetensors"))
    if not models:
        shutil.rmtree(tmp, ignore_errors=True)
        print("no model file in the download", file=sys.stderr)
        return 1
    model = models[0]
    record = {
        "name": args.name,
        "doi": args.doi,
        "title": desc.summary,
        "license": desc.license,
        "language": list(desc.language or []),
        "reported_cer_percent": (desc.metrics or {}).get("cer"),
        "file": str(model.relative_to(tmp)),
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    (tmp / "BASE_MODEL.json").write_text(json.dumps(record, indent=2))
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    print(f"saved {target / record['file']}\n{json.dumps(record, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
