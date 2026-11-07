# Typological Feature Induction (TFI)

Data and evaluation code for

> Chen Zhang, Seth Aycock and Yusuke Miyao. 2026. *Typological Feature Induction as a Benchmark
> for Inductive Metalinguistic Reasoning.* [arXiv:2611.04421](https://arxiv.org/abs/2611.04421)

In TFI a model sees English–X sentence pairs for an unnamed language X and answers a Grambank
question about X, such as the order of numeral and noun. Items are built automatically from
Grambank and the FLORES+ and BOUQuET parallel corpora, and gold answers are Grambank values.

This repository covers the PARALLEL setting on the paper's three evaluation sets, and the
cipher conditions (CIPHER-PARALLEL, CIPHER-MONO, CIPHER-MISALIGNED) on the Latin-script set:

| set | languages | features | items |
|---|---|---|---|
| core | 195 | 22 | 4,084 |
| Latin-script | 127 | 22 | 2,657 |
| extended | 24 | 195 | 3,775 |

The Latin-script set is the part of the core set whose target side can be enciphered: a fixed
character-level substitution per language that hides the language's identity while keeping
its grammar. The extended set contains every Grambank feature coded for the 24 languages that
none of the models in the paper identifies.

## Repository layout

```
data/                     items, language metadata and sampled sentence IDs (see data/README.md)
item_prompts.py           prompt construction, corpus formatting, answer extraction
eval_corpus_vllm.py       runs an open model on vLLM and scores the answers
eval_corpus_api.py        the same evaluation through any OpenAI-compatible API
cipher.py                 loads the cipher tables and enciphers the target side
fetch_corpora.py          downloads FLORES+ and BOUQuET and rebuilds the sampled corpora
scripts/setup_workdir.sh  links data/ into the working directory the scripts expect
scripts/run_examples.sh   example commands for the three evaluation sets
```

## Setup

Python 3.12. `eval_corpus_vllm.py` needs a GPU machine that can run vLLM, while
`eval_corpus_api.py` needs only the `openai` package.

```bash
pip install -r requirements.txt
bash scripts/setup_workdir.sh            # creates ./work
export EXP_DIR=work
HF_TOKEN=... python3 fetch_corpora.py    # writes work/corpora/<iso>.s<seed>.jsonl
```

FLORES+ and BOUQuET are gated datasets, so this repository contains sentence IDs only. Accept
their terms of use on the Hugging Face Hub before running `fetch_corpora.py`. It downloads the
revisions used in the paper and rebuilds the exact corpora. A corpus of n pairs is the first n
lines of a file, so corpora of different sizes are nested.

## Running

`eval_corpus_vllm.py` is configured through environment variables:

| variable | meaning |
|---|---|
| `MODEL` | Hugging Face model ID, e.g. `google/gemma-4-31B-it` or `Qwen/Qwen3.6-27B` |
| `CONDITION` | `parallel`, or `cipher`, `cipher_mono`, `cipher_shuf` (Latin-script set only) |
| `N_PAIRS` | number of sentence pairs: 16, 32, 64, 128, 256, 512 or 1024 |
| `CORPUS_SEED` | which sample of sentences: 0, 1 or 2 |
| `ITEMS_FILE` | item file; defaults to the core set |
| `ONLY_ITEMS` | restrict to a list of item IDs, e.g. the Latin-script set |
| `DO_SAMPLE`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `SAMPLE_SEED` | decoding |
| `MAX_NEW_TOKENS`, `ENABLE_THINKING`, `TP` | output budget, thinking mode, tensor parallelism |
| `TAG` | name of the output files |

Settings used in the paper: sampling with temperature 1.0, top-p 0.95, top-k 64 (Gemma) or 20
(Qwen), `SAMPLE_SEED=0`, `MAX_NEW_TOKENS=12288`, thinking off, and three corpus seeds per
setting. `scripts/run_examples.sh` runs one setting on each evaluation set and one cipher
setting.

The cipher conditions read their tables from `data/latin_script/ciphers/<iso>.json`. Each
table maps the lowercase letters of one language. Variants of one base letter (c, č, ć) map
to variants of one cipher letter, vowels map to vowels, consonants to consonants, and no
letter maps to itself. The same table is used for every corpus size and seed, and the
English side is never changed. CIPHER-MONO drops the English side and CIPHER-MISALIGNED
misaligns the pairs, as in the paper.

Each run writes three files to `$EXP_DIR/evaluations`:

- `<TAG>_summary.json`: accuracy and run metadata
- `<TAG>_details.json`: per item, the raw output, the parsed answer and correctness
- `<TAG>_prompts.jsonl`: the prompts

## Running through an API

`eval_corpus_api.py` sends the same prompts to any OpenAI-compatible chat completions
endpoint: OpenAI, Azure OpenAI (v1 endpoint), a local vLLM or SGLang server, OpenRouter, and so
on. It takes the same variables as above except the decoding and vLLM ones, plus:

| variable | meaning |
|---|---|
| `OPENAI_API_KEY` | API key |
| `OPENAI_BASE_URL` | endpoint; defaults to OpenAI, e.g. `https://<resource>.openai.azure.com/openai/v1/` for Azure or `http://localhost:8000/v1` for a local server |
| `MODEL` | model or deployment name |
| `MAX_NEW_TOKENS` | sent as `max_completion_tokens` (default 12288) |
| `TEMPERATURE`, `TOP_P`, `REASONING_EFFORT` | sent only if set |
| `CONCURRENCY` | parallel requests (default 16) |

```bash
EXP_DIR=work OPENAI_API_KEY=... MODEL=gpt-5-mini CONDITION=parallel N_PAIRS=16 \
    CORPUS_SEED=0 TAG=api_core_n16_s0 python3 eval_corpus_api.py
```

Finished items are saved to `<TAG>_partial.jsonl` as they come in, so rerunning an
interrupted command continues where it stopped. Requests that still fail after retries are
recorded with an `error` field and left out of the accuracy, and a rerun requests them again.
The output files have the same format as those of `eval_corpus_vllm.py`.

## Licenses

The code is released under the MIT License. The items in `data/` are derived from Grambank and
are released under CC BY 4.0. The data come from Grambank v1.0.3 and Glottolog v5.3 (both
CC BY 4.0), and from FLORES+ and BOUQuET (both CC BY-SA 4.0) at fixed Hugging Face revisions,
which are referenced by sentence ID only. See `data/README.md` for the exact versions.

## Citation

```bibtex
@misc{zhang2026tfi,
  title         = {Typological Feature Induction as a Benchmark for Inductive Metalinguistic Reasoning},
  author        = {Zhang, Chen and Aycock, Seth and Miyao, Yusuke},
  year          = {2026},
  eprint        = {2611.04421},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.04421}
}
```
