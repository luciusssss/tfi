#!/usr/bin/env python3
"""Run TFI items against any OpenAI-compatible chat completions API.

Works with OpenAI, Azure OpenAI (v1 endpoint), a vLLM or SGLang server, OpenRouter and other
compatible services. Prompts, answer extraction and scoring are the same as in
eval_corpus_vllm.py, and the output files have the same format.

Configured through environment variables, like eval_corpus_vllm.py:

    OPENAI_API_KEY      API key (read from the environment only)
    OPENAI_BASE_URL     endpoint, e.g. https://api.openai.com/v1 (the default),
                        https://<resource>.openai.azure.com/openai/v1/ for Azure,
                        or http://localhost:8000/v1 for a local vLLM server
    MODEL               model or deployment name
    CONDITION, N_PAIRS, CORPUS_SEED, ITEMS_FILE, ONLY_ITEMS, NUM_ITEMS, TAG
                        as in eval_corpus_vllm.py
    MAX_NEW_TOKENS      sent as max_completion_tokens (default 12288)
    TEMPERATURE, TOP_P  sent only if set; reasoning models often reject them
    REASONING_EFFORT    sent only if set, e.g. low / medium / high
    CONCURRENCY         parallel requests (default 16)

Finished items are appended to <TAG>_partial.jsonl as they complete, so rerunning the same
command after an interruption skips them. Failed requests are retried with backoff and, if
they still fail, recorded with an error and left out of the accuracy.
"""
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import item_prompts as ip  # noqa: E402

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ip.EXP_DIR, "evaluations"))
CORPUS_DIR = os.path.join(ip.EXP_DIR, "corpora")
MODEL = os.environ.get("MODEL", "")
TAG = os.environ.get("TAG", "api_smoke")
CONDITION = os.environ.get("CONDITION", "parallel")
assert CONDITION in ("parallel", "monolingual", "shuffled",
                     "parallel_named", "cipher",
                     "cipher_mono", "cipher_shuf"), CONDITION
N_PAIRS = int(os.environ.get("N_PAIRS", "0"))
CORPUS_SEED = int(os.environ.get("CORPUS_SEED", "0"))
ONLY_ITEMS = os.environ.get("ONLY_ITEMS", "")
NUM_ITEMS = int(os.environ.get("NUM_ITEMS", "0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "12288"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
MAX_RETRIES = 5
# Optional sampling parameters: only sent when set, since several reasoning models
# reject temperature and top_p.
EXTRA = {}
if os.environ.get("TEMPERATURE"):
    EXTRA["temperature"] = float(os.environ["TEMPERATURE"])
if os.environ.get("TOP_P"):
    EXTRA["top_p"] = float(os.environ["TOP_P"])
if os.environ.get("REASONING_EFFORT"):
    EXTRA["reasoning_effort"] = os.environ["REASONING_EFFORT"]


def load_corpus(iso):
    pairs = []
    with open(os.path.join(CORPUS_DIR, f"{iso}.s{CORPUS_SEED}.jsonl")) as f:
        for line in f:
            pairs.append(json.loads(line))
            if len(pairs) >= N_PAIRS:
                break
    return pairs


def load_items():
    items = [json.loads(l) for l in open(ip.ITEMS_FILE)]
    if ONLY_ITEMS:
        keep = {l.strip() for l in open(ONLY_ITEMS) if l.strip()}
        items = [it for it in items if it["item_id"] in keep]
        missing = keep - {it["item_id"] for it in items}
        if missing:
            sys.exit(f"[fatal] ONLY_ITEMS has {len(missing)} ids not in the item set: "
                     f"{sorted(missing)[:5]}")
    elif NUM_ITEMS > 0:
        items = ip.stratified_sample(items, NUM_ITEMS, seed=ip.SEED)
    items.sort(key=lambda it: (it["iso639_3"], it["item_id"]))
    return items


def main():
    if not MODEL:
        sys.exit("[fatal] set MODEL to the model or deployment name")
    if N_PAIRS <= 0:
        sys.exit("[fatal] set N_PAIRS")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[fatal] set OPENAI_API_KEY")
    from openai import OpenAI
    client = OpenAI(max_retries=0)       # base_url and key come from the environment

    descs = ip.load_descriptions()
    items = load_items()
    missing = sorted({it["iso639_3"] for it in items if not os.path.exists(
        os.path.join(CORPUS_DIR, f'{it["iso639_3"]}.s{CORPUS_SEED}.jsonl'))})
    if missing:
        sys.exit(f"[fatal] {len(missing)} languages have no corpus file: {missing[:8]}; "
                 f"run fetch_corpora.py first")
    print(f"[info] model={MODEL} base_url={client.base_url} condition={CONDITION} "
          f"n_pairs={N_PAIRS} corpus_seed={CORPUS_SEED} items={len(items)} "
          f"concurrency={CONCURRENCY} extra={EXTRA}", flush=True)

    corpus_cache, n_pairs_used, tasks = {}, {}, []
    for it in items:
        iso = it["iso639_3"]
        if iso not in corpus_cache:
            pairs = load_corpus(iso)
            corpus_cache[iso] = ip.format_corpus(
                pairs, condition=CONDITION, iso=iso,
                seed=f"{CORPUS_SEED}:{iso}:{N_PAIRS}")
            n_pairs_used[iso] = len(pairs)
        shown = ip.present_options(it)
        body = ip.build_prompt(it, descs[it["feature_id"]], shown,
                               corpus=corpus_cache[iso], condition=CONDITION)
        tasks.append((it, shown, body))

    usage_total = Counter()
    lock = threading.Lock()
    abort = threading.Event()            # set on an authentication error

    def work(task):
        it, shown, body = task
        text, err, usage = "", None, None
        for attempt in range(MAX_RETRIES):
            if abort.is_set():
                err = "aborted after an authentication error"
                break
            try:
                r = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": body}],
                    max_completion_tokens=MAX_NEW_TOKENS, **EXTRA)
                text = r.choices[0].message.content or ""
                if r.usage is not None:
                    usage = {"prompt": r.usage.prompt_tokens,
                             "completion": r.usage.completion_tokens}
                    with lock:
                        usage_total.update(usage)
                err = None
                break
            except Exception as e:                        # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                if type(e).__name__ == "AuthenticationError":
                    abort.set()
                    break
                if type(e).__name__ == "BadRequestError" or attempt == MAX_RETRIES - 1:
                    break                                 # retrying will not help
                time.sleep(2 ** attempt * 2)
        pred = ip.extract_choice(text, {x["value"] for x in shown})
        lbl = {x["value"]: x["label"] for x in shown}
        return {
            "item_id": it["item_id"], "iso639_3": it["iso639_3"],
            "iso_name": it["iso_name"], "feature_id": it["feature_id"],
            "feature_group": it["feature_group"], "corpus_group": it["corpus_group"],
            "n_pairs": n_pairs_used[it["iso639_3"]],
            "presented_order": [x["value"] for x in shown],
            "gold_value": it["gold_value"], "gold_label": it["gold_label"],
            "pred_value": pred, "pred_label": lbl.get(pred),
            "correct": pred == it["gold_value"], "unparsed": pred is None,
            "usage": usage, "model_output": text, "error": err,
        }

    # Resume: items already answered in <TAG>_partial.jsonl are not requested again.
    os.makedirs(OUT_DIR, exist_ok=True)
    partial = os.path.join(OUT_DIR, f"{TAG}_partial.jsonl")
    results = []
    if os.path.exists(partial):
        want = {t[0]["item_id"] for t in tasks}
        done = {}
        for line in open(partial):
            if line.strip():
                r = json.loads(line)
                if r["item_id"] in want and not r.get("error"):
                    done[r["item_id"]] = r
        for r in done.values():
            usage_total.update(r.get("usage") or {})
        results = list(done.values())
        tasks = [t for t in tasks if t[0]["item_id"] not in done]
        print(f"[info] resuming: {len(done)} items done, {len(tasks)} to go", flush=True)

    t0 = time.time()
    with open(partial, "a", encoding="utf-8") as ckpt, \
            ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if not r["error"]:
                ckpt.write(json.dumps(r, ensure_ascii=False) + "\n")
                ckpt.flush()
            if k % 100 == 0 or k == len(futs):
                ok = [x for x in results if not x["error"]]
                acc = sum(x["correct"] for x in ok) / len(ok) if ok else 0
                print(f"  [{k}/{len(futs)}] acc={acc:.3f} errors={len(results) - len(ok)}",
                      flush=True)
    if abort.is_set():
        sys.exit(f"[fatal] authentication failed; finished items are kept in {partial}, "
                 f"rerun the same command after fixing the key")
    results.sort(key=lambda r: r["item_id"])

    ok = [r for r in results if not r["error"]]
    n_ok = len(ok)
    gold_by_feat = defaultdict(Counter)
    for r in ok:
        gold_by_feat[r["feature_id"]][r["gold_value"]] += 1
    majority = (sum(max(c.values()) for c in gold_by_feat.values()) / n_ok) if n_ok else None

    def group(key):
        b = defaultdict(list)
        for r in ok:
            b[r[key]].append(r["correct"])
        return {k: {"n": len(v), "correct": sum(v), "accuracy": round(sum(v) / len(v), 4)}
                for k, v in sorted(b.items())}

    correct = sum(r["correct"] for r in ok)
    summary = {
        "tag": TAG, "backend": "openai-compatible", "model": MODEL,
        "base_url": str(client.base_url), "condition": CONDITION,
        "n_pairs_requested": N_PAIRS, "corpus_seed": CORPUS_SEED,
        "max_new_tokens": MAX_NEW_TOKENS, "request_params": EXTRA,
        "shuffle_options": ip.SHUFFLE_OPTIONS, "seed": ip.SEED,
        "n_items": len(results), "n_errors": len(results) - n_ok,
        "n_correct": correct,
        "accuracy": round(correct / n_ok, 4) if n_ok else None,
        "majority_baseline": round(majority, 4) if majority is not None else None,
        "n_unparsed": sum(r["unparsed"] for r in ok),
        "usage_tokens": dict(usage_total),
        "elapsed_sec": round(time.time() - t0, 1),
        "accuracy_by_feature": group("feature_id"),
        "accuracy_by_corpus_group": group("corpus_group"),
        "accuracy_by_language": group("iso639_3"),
    }
    with open(os.path.join(OUT_DIR, f"{TAG}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, f"{TAG}_details.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if not summary["n_errors"]:
        # Keep the checkpoint for reference, but stop a later run with the same TAG
        # from resuming from it.
        os.replace(partial, partial.replace("_partial.jsonl", "_partial.done.jsonl"))
    print(json.dumps({k: v for k, v in summary.items()
                      if not k.startswith("accuracy_by")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
