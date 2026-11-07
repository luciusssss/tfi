#!/usr/bin/env python3
"""Rebuild the sampled corpora from FLORES+ and BOUQuET using the released sentence IDs.

Both corpora are gated on the Hugging Face Hub, so this repository contains only sentence
IDs. Accept the terms of use of both datasets, then run

    HF_TOKEN=... python3 fetch_corpora.py

This writes, under $EXP_DIR (default ./work):

    corpora/<iso>.s<seed>.jsonl    {"eng", "tgt"}

Each file lists sentence pairs in sampling order. A corpus of n pairs is the first n lines,
so corpora of different sizes are nested.

Sentence IDs are "flores:<split>:<id>" or "bouquet:<split>:<uniq_id>". English text for
FLORES+ comes from eng_Latn, and for BOUQuET from the English side of each pair.
"""
import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Dataset revisions the experiments were built from.
FLORES_REV = "b3a5298db5721c8a682e7ef00a37fcc9ab522757"
BOUQUET_REV = "9a6070a9652e350dda1d353c4fd198533199a911"
FLORES_BASE = f"https://huggingface.co/datasets/openlanguagedata/flores_plus/resolve/{FLORES_REV}/"
BOUQUET_BASE = f"https://huggingface.co/datasets/facebook/bouquet/resolve/{BOUQUET_REV}/"
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
EXP_DIR = os.environ.get("EXP_DIR", "work")


def http_get(url):
    """Authenticated download with retries; returns None on 404."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    return None


_eng = {}


def flores_english(split):
    if split not in _eng:
        raw = http_get(f"{FLORES_BASE}{split}/eng_Latn.jsonl")
        _eng[split] = {d["id"]: d["text"]
                       for d in map(json.loads, raw.decode("utf-8").splitlines()) if d}
    return _eng[split]


def load_flores(tag):
    """tag is a FLORES+ config such as apc_Arab_nort3139. Returns {id: (eng, tgt)}."""
    out = {}
    for split in ("dev", "devtest"):
        raw = http_get(f"{FLORES_BASE}{split}/{tag}.jsonl")
        if raw is None:
            continue
        eng = flores_english(split)
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            e = eng.get(d["id"])
            if e is not None and d.get("text"):
                out[f"flores:{split}:{d['id']}"] = (e, d["text"])
    return out


def load_bouquet(tag):
    """tag is a BOUQuET language tag such as aar_Latn; sentence level, dev and test."""
    import pandas as pd

    out = {}
    for split in ("dev", "test"):
        raw = None
        for name in (f"{tag}-eng_Latn", f"eng_Latn-{tag}"):
            raw = http_get(f"{BOUQUET_BASE}benchmark_data/sentence_level/{split}/{name}.parquet")
            if raw is not None:
                break
        if raw is None:
            continue
        for r in pd.read_parquet(io.BytesIO(raw)).itertuples(index=False):
            src_is_eng = getattr(r, "src_lang", "") == "eng_Latn"
            eng = r.src_text if src_is_eng else r.tgt_text
            tgt = r.tgt_text if src_is_eng else r.src_text
            if eng and tgt:
                out[f"bouquet:{split}:{r.uniq_id}"] = (str(eng), str(tgt))
    return out


def load_language(row):
    recs = {}
    for tag in filter(None, row["flores_configs"].split("|")):
        recs.update(load_flores(tag))
    for tag in filter(None, row["bouquet_configs"].split("|")):
        recs.update(load_bouquet(tag))
    return row["iso639_3"], recs


def read_ids(path):
    return open(path).read().split()


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="the data/ directory of this repository")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not TOKEN:
        sys.exit("Set HF_TOKEN: FLORES+ and BOUQuET are gated datasets.")

    langs = list(csv.DictReader(open(os.path.join(args.data_dir, "languages.csv"))))
    flores_english("dev"), flores_english("devtest")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        text = dict(ex.map(load_language, langs))
    print(f"downloaded {len(text)} languages")

    ids_main = os.path.join(args.data_dir, "corpus_ids")
    out = os.path.join(EXP_DIR, "corpora")
    os.makedirs(out, exist_ok=True)
    for fn in sorted(os.listdir(ids_main)):
        iso = fn.split(".")[0]
        write_jsonl(os.path.join(out, fn.replace(".txt", ".jsonl")),
                    [{"eng": text[iso][i][0], "tgt": text[iso][i][1]}
                     for i in read_ids(os.path.join(ids_main, fn))])

    print(f"wrote corpora to {EXP_DIR}/corpora")


if __name__ == "__main__":
    main()
