#!/usr/bin/env python3
"""Corpus conditions: the language name is withheld and the model gets only N parallel sentence pairs.

Implementation notes:
  1. No language name in the prompt (item_prompts.build_prompt omits it automatically
     when a corpus is passed).
  2. enable_prefix_caching is on, and items are sent to vLLM **sorted by language** --
     the 22 items of one language share the same corpus prefix. Without caching, Gemma
     at 1024 pairs would need 314 GB of KV cache.
  3. max_model_len is set per size rather than a blanket 262k, since the reserved KV
     cache is computed from it.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import item_prompts as ip  # noqa: E402

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ip.EXP_DIR, "evaluations"))
CORPUS_DIR = os.path.join(ip.EXP_DIR, "corpora")
PIVOT_DIR = os.path.join(ip.EXP_DIR, "corpora_pivot")
MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it")
TAG = os.environ.get("TAG", "corpus_smoke")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
NUM_ITEMS = int(os.environ.get("NUM_ITEMS", "0"))
# Run only the given item_ids (one per line). Used to re-run items truncated by
# max_new_tokens: present_options seeds per item_id, so a subset run yields the same
# option order as the full run and can be merged back directly.
ONLY_ITEMS = os.environ.get("ONLY_ITEMS", "")
TP = int(os.environ.get("TP", "4"))
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.90"))
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"
LIMIT_MM_IMAGES = (int(os.environ["LIMIT_MM_IMAGES"])
                   if os.environ.get("LIMIT_MM_IMAGES", "") != "" else None)
N_PAIRS = int(os.environ["N_PAIRS"])          # 64/128/256/512/1024
CORPUS_SEED = int(os.environ.get("CORPUS_SEED", os.environ.get("SEED", "0")))
# 0 = set from the **measured** longest prompt before building the LLM (see main). A
# heuristic formula overestimates by about 1.5x, and max_model_len directly determines
# the per-sequence KV reservation: overestimating at 1024 pairs cuts concurrency from 3 to 2.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "0"))
# Model context limit; 0 = read max_position_embeddings from the config
CTX_CAP = int(os.environ.get("CTX_CAP", "0"))
# Ablation conditions: parallel=C1 (n correctly aligned pairs) / monolingual=C2 (n target-language sentences only)
#           shuffled=C3 (n pairs but TL side deranged, breaking only alignment)
# Language of the known side. Default eng = use the old corpus directory, byte-comparable with all earlier results.
# For non-eng, read corpora_pivot/ (a pool resampled from the multi-parallel data); **the eng arm
# must then also be re-run on that pool**, since the two pools are sampled differently and the old
# eng results cannot serve as the control. PIVOT_POOL=1 forces eng onto the new pool too.
PIVOT = os.environ.get("PIVOT", "eng")
PIVOT_NAMES = {"eng": "English", "cmn": "Chinese", "jpn": "Japanese",
               "vie": "Vietnamese", "djc": "Dar Daju Daju", "knc": "Kanuri",
               "fra": "French", "ind": "Indonesian", "swh": "Swahili",
               "rus": "Russian", "heb": "Hebrew", "tur": "Turkish",
               "hun": "Hungarian", "kor": "Korean", "hin": "Hindi",
               "pes": "Persian", "cym": "Welsh", "arz": "Egyptian Arabic"}
PIVOT_TAGS = {"eng": "EN", "cmn": "ZH", "jpn": "JA",
              "vie": "VI", "djc": "DJ", "knc": "KR",
              "fra": "FR", "ind": "ID", "swh": "SW", "rus": "RU", "heb": "HE",
              "tur": "TR", "hun": "HU", "kor": "KO", "hin": "HI", "pes": "FA",
              "cym": "CY", "arz": "AR"}
assert PIVOT in PIVOT_NAMES, PIVOT
# Only the English pivot's corpora can be rebuilt with fetch_corpora.py.
USE_PIVOT_POOL = PIVOT != "eng" or os.environ.get("PIVOT_POOL") == "1"

CONDITION = os.environ.get("CONDITION", "parallel")
assert CONDITION in ("parallel", "monolingual", "shuffled",
                     "parallel_named", "cipher",
                     "cipher_mono", "cipher_shuf"), CONDITION
# Decoding: greedy by default. With DO_SAMPLE=1 the three parameters below are used
# (defaults taken from the two models' generation_config.json; Gemma top_k=64,
# Qwen top_k=20, so TOP_K must be passed per model)
DO_SAMPLE = os.environ.get("DO_SAMPLE", "0") == "1"
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TOP_K = int(os.environ.get("TOP_K", "64"))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "0"))


def load_corpus(iso):
    """Take the first N pairs for this language and seed -- nesting is achieved by taking prefixes."""
    if USE_PIVOT_POOL:
        path = os.path.join(PIVOT_DIR, f"{iso}.{PIVOT}.s{CORPUS_SEED}.jsonl")
    else:
        path = os.path.join(CORPUS_DIR, f"{iso}.s{CORPUS_SEED}.jsonl")
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
            if len(pairs) >= N_PAIRS:
                break
    return pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    descs = ip.load_descriptions()
    items = [json.loads(l) for l in open(ip.ITEMS_FILE)]
    if ONLY_ITEMS:
        keep = {l.strip() for l in open(ONLY_ITEMS) if l.strip()}
        items = [it for it in items if it["item_id"] in keep]
        missing = keep - {it["item_id"] for it in items}
        if missing:
            sys.exit(f"[fatal] ONLY_ITEMS has {len(missing)} ids not in the item set: "
                     f"{sorted(missing)[:5]}")
        print(f"[info] ONLY_ITEMS={ONLY_ITEMS} -> {len(items)} items", flush=True)
    elif NUM_ITEMS > 0:
        items = ip.stratified_sample(items, NUM_ITEMS, seed=ip.SEED)
    # Sort by language: prefix caching matches prefixes, so items of the same language must be sent together to hit the cache
    items.sort(key=lambda it: (it["iso639_3"], it["item_id"]))

    print(f"[info] condition={CONDITION} pivot={PIVOT} "
          f"pool={'corpora_pivot' if USE_PIVOT_POOL else 'corpora'}")
    print(f"[info] model={MODEL} tp={TP} n_pairs={N_PAIRS} corpus_seed={CORPUS_SEED} "
          f"max_model_len={MAX_MODEL_LEN} thinking={ENABLE_THINKING}")
    print(f"[info] items={len(items)} max_new_tokens={MAX_NEW_TOKENS}", flush=True)

    # Check the corpus files **before** building the LLM: model loading takes minutes,
    # and discovering a missing file only when fetching the corpus wastes the whole run.
    # The pivot pool covers only the 127 cipher languages while the item set has 195, so
    # not restricting the items is bound to hit this.
    missing = sorted({it["iso639_3"] for it in items
                      if not os.path.exists(
                          os.path.join(PIVOT_DIR,
                                       f'{it["iso639_3"]}.{PIVOT}.s{CORPUS_SEED}.jsonl')
                          if USE_PIVOT_POOL else
                          os.path.join(CORPUS_DIR, f'{it["iso639_3"]}.s{CORPUS_SEED}.jsonl'))})
    if missing:
        sys.exit(f"[fatal] {len(missing)} languages have no corpus file (pool="
                 f"{'corpora_pivot' if USE_PIVOT_POOL else 'corpora'}"
                 f"{', pivot=' + PIVOT if USE_PIVOT_POOL else ''}): "
                 f"{missing[:8]}{' …' if len(missing) > 8 else ''}\n"
                 f"        the pivot pool covers only the 127 cipher languages; restrict the items with "
                 f"ONLY_ITEMS=$EXP_DIR/items/latin_items.txt")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    corpus_cache, n_pairs_used = {}, {}
    prepared = []
    for it in items:
        iso = it["iso639_3"]
        if iso not in corpus_cache:
            pairs = load_corpus(iso)
            # The derangement seed is tied to language + CORPUS_SEED + n, so the same
            # language gets the same derangement across repeated runs at one size
            # (nesting does not hold for C3: the n=32 derangement does not extend the
            # n=16 one, since it is computed over that size's sentence set -- do not
            # assume nesting in analysis)
            corpus_cache[iso] = ip.format_corpus(
                pairs, condition=CONDITION, iso=iso,
                seed=f"{CORPUS_SEED}:{iso}:{N_PAIRS}",
                src_tag=PIVOT_TAGS[PIVOT])
            n_pairs_used[iso] = len(pairs)
        shown = ip.present_options(it)
        body = ip.build_prompt(it, descs[it["feature_id"]], shown,
                               corpus=corpus_cache[iso], condition=CONDITION,
                               src_name=PIVOT_NAMES[PIVOT],
                               src_tag=PIVOT_TAGS[PIVOT])
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = tok.apply_chat_template([{"role": "user", "content": body}],
                                           enable_thinking=ENABLE_THINKING, **kwargs)
        except TypeError:
            text = tok.apply_chat_template([{"role": "user", "content": body}], **kwargs)
        prepared.append((it, shown, text))

    # The model's own context limit. vLLM raises VLLMValidationError for a prompt above
    # it, and does so inside generate() -- a single over-limit prompt kills the whole
    # job. Such prompts must therefore be removed before building the LLM and accounted
    # for separately, not just warned about.
    # Qwen3.6-27B tokenises scripts such as Burmese/Lao about half as efficiently as
    # Gemma (longest prompt at 1024 pairs: 317862 tok vs 149568), so the margin must be
    # re-checked whenever the model changes.
    ctx_cap = CTX_CAP
    if not ctx_cap:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
            cfg = getattr(cfg, "text_config", cfg)
            ctx_cap = int(getattr(cfg, "max_position_embeddings", 0)) or 262144
        except Exception as e:                                  # noqa: BLE001
            print(f"[warn] cannot read max_position_embeddings ({e}), falling back to 262144")
            ctx_cap = 262144

    tok_len = [len(tok(p, add_special_tokens=False)["input_ids"]) for *_, p in prepared]
    budget = ctx_cap - MAX_NEW_TOKENS
    runnable = [pr for pr, n in zip(prepared, tok_len) if n <= budget]
    excluded = [(pr, n) for pr, n in zip(prepared, tok_len) if n > budget]
    if not runnable:
        sys.exit(f"[fatal] all {len(prepared)} prompts exceed {budget} tok; nothing to run")

    lens = sorted(n for n in tok_len if n <= budget)
    max_len = MAX_MODEL_LEN or min(ctx_cap, lens[-1] + MAX_NEW_TOKENS + 512)
    print(f"[info] prompt tokens: median {lens[len(lens)//2]} max {lens[-1]} "
          f"-> max_model_len={max_len} (model limit {ctx_cap})", flush=True)
    if excluded:
        from collections import Counter as _C
        by_lang = _C(pr[0]["iso639_3"] for pr, _ in excluded)
        print(f"[warn] {len(excluded)} prompts exceed {budget} tok; excluded and recorded as "
              f"context_overflow (not generated, but kept in the results)", flush=True)
        print(f"[warn]   languages affected: {dict(by_lang.most_common())}", flush=True)
        print(f"[warn]   longest {max(n for _, n in excluded)} tok", flush=True)
    prepared = runnable

    prompt_path = os.path.join(OUT_DIR, f"{TAG}_prompts.jsonl")
    with open(prompt_path, "w") as f:
        for it, shown, text in prepared:
            f.write(json.dumps({
                "item_id": it["item_id"], "iso639_3": it["iso639_3"],
                "feature_id": it["feature_id"],
                "n_pairs": n_pairs_used[it["iso639_3"]],
                "presented_order": [o["value"] for o in shown],
                "gold_value": it["gold_value"],
                "prompt_templated": text,
            }, ensure_ascii=False) + "\n")
    print(f"[info] prompts -> {prompt_path}", flush=True)

    llm_kwargs = dict(model=MODEL, tensor_parallel_size=TP, dtype="bfloat16",
                      max_model_len=max_len, gpu_memory_utilization=GPU_UTIL,
                      trust_remote_code=True, enable_prefix_caching=True)
    if LIMIT_MM_IMAGES is not None:
        llm_kwargs["limit_mm_per_prompt"] = {"image": LIMIT_MM_IMAGES}
    llm = LLM(**llm_kwargs)
    # Greedy by default (reproducible, a prerequisite for paired tests). DO_SAMPLE=1
    # switches to the operating point recommended by the authors in the model's
    # generation_config.json -- both models specify do_sample:true, so greedy is
    # off-spec, and the Qwen documentation explicitly warns that greedy causes endless
    # repetition. This is used to test whether the loop rate rising with corpus size is
    # a model property or a decoding artefact. Under sampling results are no longer
    # deterministic, so several seeds are needed to estimate accuracy.
    if DO_SAMPLE:
        sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                            max_tokens=MAX_NEW_TOKENS, seed=SAMPLE_SEED)
        print(f"[info] sampling decode: temperature={TEMPERATURE} top_p={TOP_P} "
              f"top_k={TOP_K} seed={SAMPLE_SEED}", flush=True)
    else:
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_NEW_TOKENS)
        print("[info] greedy decode", flush=True)

    t0 = time.time()
    outs = llm.generate([p for *_, p in prepared], sp)
    elapsed = time.time() - t0

    results = []
    for (it, shown, _), o in zip(prepared, outs):
        text = o.outputs[0].text
        codes = {x["value"] for x in shown}
        pred = ip.extract_choice(text, codes)
        lbl = {x["value"]: x["label"] for x in shown}
        results.append({
            "item_id": it["item_id"], "iso639_3": it["iso639_3"],
            "iso_name": it["iso_name"], "feature_id": it["feature_id"],
            "feature_group": it["feature_group"], "corpus_group": it["corpus_group"],
            "n_pairs": n_pairs_used[it["iso639_3"]],
            "presented_order": [x["value"] for x in shown],
            "gold_value": it["gold_value"], "gold_label": it["gold_label"],
            "pred_value": pred, "pred_label": lbl.get(pred),
            "correct": pred == it["gold_value"], "unparsed": pred is None,
            "n_output_tokens": len(o.outputs[0].token_ids),
            "model_output": text,
        })

    # Items exceeding the model context: not run, but kept in the results, otherwise the
    # item counts do not match and cross-model comparisons would silently compute
    # accuracy on different item sets. Counted as wrong (same convention as unparsed),
    # with failure=context_overflow so they can be excluded separately in analysis.
    for (it, shown, _), n_tok in excluded:
        results.append({
            "item_id": it["item_id"], "iso639_3": it["iso639_3"],
            "iso_name": it["iso_name"], "feature_id": it["feature_id"],
            "feature_group": it["feature_group"], "corpus_group": it["corpus_group"],
            "n_pairs": n_pairs_used[it["iso639_3"]],
            "presented_order": [x["value"] for x in shown],
            "gold_value": it["gold_value"], "gold_label": it["gold_label"],
            "pred_value": None, "pred_label": None,
            "correct": False, "unparsed": True, "failure": "context_overflow",
            "n_prompt_tokens": n_tok, "n_output_tokens": 0,
            "model_output": "",
        })
    results.sort(key=lambda r: r["item_id"])

    n = len(results)
    correct = sum(r["correct"] for r in results)
    gold_by_feat = defaultdict(Counter)
    for r in results:
        gold_by_feat[r["feature_id"]][r["gold_value"]] += 1
    majority = sum(max(c.values()) for c in gold_by_feat.values()) / n

    def group(key):
        b = defaultdict(list)
        for r in results:
            b[r[key]].append(r["correct"])
        return {k: {"n": len(v), "correct": sum(v), "accuracy": round(sum(v)/len(v), 4)}
                for k, v in sorted(b.items())}

    out_toks = [r["n_output_tokens"] for r in results]
    summary = {
        "tag": TAG, "condition": f"corpus ({N_PAIRS} pairs, language name withheld)",
        "backend": "vllm", "model": MODEL, "tensor_parallel_size": TP,
        "n_pairs_requested": N_PAIRS, "corpus_seed": CORPUS_SEED,
        "max_model_len": max_len, "enable_thinking": ENABLE_THINKING,
        "n_items": n, "n_correct": correct, "accuracy": round(correct/n, 4),
        "majority_baseline": round(majority, 4),
        "delta_over_majority": round(correct/n - majority, 4),
        "n_unparsed": sum(r["unparsed"] for r in results),
        "n_langs_capped": sum(1 for v in set(n_pairs_used.values()) if v < N_PAIRS),
        "ctx_cap": ctx_cap,
        "n_context_overflow": len(excluded),
        "context_overflow_items": [pr[0]["item_id"] for pr, _ in excluded],
        "elapsed_sec": round(elapsed, 1), "items_per_sec": round(n/elapsed, 2),
        "prompt_tokens": {"median": lens[len(lens)//2], "max": lens[-1]},
        "output_tokens": {"median": sorted(out_toks)[n//2], "max": max(out_toks),
                          "total": sum(out_toks)},
        "condition": CONDITION, "pivot": PIVOT,
        "corpus_pool": "corpora_pivot" if USE_PIVOT_POOL else "corpora",
        "max_new_tokens": MAX_NEW_TOKENS, "shuffle_options": ip.SHUFFLE_OPTIONS,
        "decoding": ({"mode": "sample", "temperature": TEMPERATURE, "top_p": TOP_P,
                      "top_k": TOP_K, "sample_seed": SAMPLE_SEED}
                     if DO_SAMPLE else {"mode": "greedy"}),
        "seed": ip.SEED,
        "accuracy_by_feature": group("feature_id"),
        "accuracy_by_corpus_group": group("corpus_group"),
        "accuracy_by_language": group("iso639_3"),
    }
    with open(os.path.join(OUT_DIR, f"{TAG}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, f"{TAG}_details.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n===== Corpus condition (vLLM) =====")
    print(json.dumps({k: v for k, v in summary.items()
                      if not k.startswith("accuracy_by")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
