#!/bin/bash
# Lay out data/ in the directory structure the evaluation script expects.
# Usage: bash scripts/setup_workdir.sh [data_dir] [workdir]   (defaults: data, work)
set -e
DATA=$(realpath "${1:-data}")
W=${2:-work}
mkdir -p "$W/items" "$W/items_expand" "$W/coverage/.rawcache" "$W/evaluations"
ln -sfn "$DATA/core/items.jsonl" "$W/items/items.jsonl"
ln -sfn "$DATA/latin_script/item_ids.txt" "$W/items/latin_items.txt"
ln -sfn "$DATA/extended/items.jsonl" "$W/items_expand/items.jsonl"
ln -sfn "$DATA/latin_script/ciphers" "$W/ciphers"
ln -sfn "$DATA/grambank/grambank_parameters.csv" "$W/coverage/.rawcache/grambank_parameters.csv"
echo "work directory ready: $W"
echo "next: HF_TOKEN=... EXP_DIR=$W python3 fetch_corpora.py --data-dir $DATA"
