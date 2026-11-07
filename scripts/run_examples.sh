#!/bin/bash
# PARALLEL on the three evaluation sets, CIPHER-PARALLEL on the Latin-script set, and an
# API example. Run from the repository root after scripts/setup_workdir.sh and
# fetch_corpora.py. Each command writes <TAG>_summary.json and <TAG>_details.json to
# $EXP_DIR/evaluations (the vLLM script also writes <TAG>_prompts.jsonl).
# N_PAIRS: 16, 32, 64, 128, 256, 512 or 1024. CORPUS_SEED: 0, 1 or 2.
set -e
export EXP_DIR=${EXP_DIR:-work}
W=$EXP_DIR

# Decoding used in the paper: each model's recommended sampling parameters.
GEMMA="MODEL=google/gemma-4-31B-it TOP_K=64"
QWEN="MODEL=Qwen/Qwen3.6-27B TOP_K=20"
COMMON="CONDITION=parallel DO_SAMPLE=1 TEMPERATURE=1.0 TOP_P=0.95 SAMPLE_SEED=0 MAX_NEW_TOKENS=12288 TP=4"

# Core set (195 languages, 4,084 items)
env $GEMMA $COMMON N_PAIRS=16 CORPUS_SEED=0 TAG=gemma_core_n16_s0 python3 eval_corpus_vllm.py

# Latin-script set (127 languages, 2,657 items)
env $QWEN $COMMON N_PAIRS=128 CORPUS_SEED=0 ONLY_ITEMS=$W/items/latin_items.txt \
    TAG=qwen_latin_n128_s0 python3 eval_corpus_vllm.py

# Extended set (24 languages, 3,775 items)
env $GEMMA $COMMON N_PAIRS=16 CORPUS_SEED=0 ITEMS_FILE=$W/items_expand/items.jsonl \
    TAG=gemma_extended_n16_s0 python3 eval_corpus_vllm.py

# CIPHER-PARALLEL: Latin-script set with the target side enciphered.
# CONDITION=cipher_mono and cipher_shuf give CIPHER-MONO and CIPHER-MISALIGNED.
env $GEMMA $COMMON CONDITION=cipher N_PAIRS=128 CORPUS_SEED=0 ONLY_ITEMS=$W/items/latin_items.txt \
    TAG=gemma_cipher_n128_s0 python3 eval_corpus_vllm.py

# Any OpenAI-compatible API (OpenAI, Azure OpenAI v1 endpoint, a local vLLM server, ...).
# Needs OPENAI_API_KEY, and OPENAI_BASE_URL for endpoints other than OpenAI.
env MODEL=gpt-5-mini CONDITION=parallel N_PAIRS=16 CORPUS_SEED=0 \
    TAG=api_core_n16_s0 python3 eval_corpus_api.py
