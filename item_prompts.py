#!/usr/bin/env python3
"""Pure item -> prompt logic, used by the evaluation script.

Deliberately does not import torch/transformers, so prompts can be inspected from any
environment.

The feature description uses only the **Summary** section of the Grambank Description, and
the prompt does not reveal that it comes from Grambank. **Never use the Examples section**:
it lists per-language examples with their coding (e.g. "Skolt Saami is coded 1"), which
would leak the answer if the target language appears there.

Answers use Grambank's own numeric codes, and the display order is deterministically
shuffled: the number carries the answer identity and the position only affects
presentation, which removes position bias without an extra mapping layer.
"""
import csv
import os
import random
import re

EXP_DIR = os.environ.get("EXP_DIR", "work")
PARAMS_CACHE = os.path.join(EXP_DIR, "coverage", ".rawcache", "grambank_parameters.csv")
ITEMS_FILE = os.environ.get("ITEMS_FILE", os.path.join(EXP_DIR, "items", "items.jsonl"))

SHUFFLE_OPTIONS = os.environ.get("SHUFFLE_OPTIONS", "1") == "1"
INCLUDE_PROCEDURE = os.environ.get("INCLUDE_PROCEDURE", "0") == "1"
# Without the feature description (Grambank's official definition), the model only sees
# a one-line question and bare option labels (mostly absent/present). This tests whether
# the model knows **how Grambank defines** the feature, which is a different axis from
# whether it knows the language.
# It is not a clean knowledge test: for 9 of the 22 features the summary contains an
# explicit contrast/exclusion (GB044 states it is opposed to the free forms of GB318),
# so without it the item is itself ambiguous, and a wrong answer may mean "does not know"
# or "applies a different criterion than Grambank". Analyse the two groups separately.
INCLUDE_SUMMARY = os.environ.get("INCLUDE_SUMMARY", "1") == "1"
SEED = int(os.environ.get("SEED", "0"))

# The URL of a markdown link may itself contain parentheses (e.g. .../Core-arguments-(S,-A-and-P)),
# so \([^)]+\) would stop early and leave a stray ")" -- one level of nested parentheses must be allowed
LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*(?:[^()\s]|\([^()]*\))*\s*\)")


def clean_markdown(s):
    s = LINK_RE.sub(r"\1", s)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)
    s = s.replace("*", "")
    s = re.sub(r"^\s*[-–]\s+", "", s, flags=re.M)
    s = re.sub(r"\s*\n\s*", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def load_descriptions(path=None):
    """feature_id -> {"summary", "procedure"}, both with markdown cleaned."""
    out = {}
    with open(path or PARAMS_CACHE) as f:
        for r in csv.DictReader(f):
            desc = r.get("Description") or ""
            sec = {}
            for name in ("Summary", "Procedure"):
                m = re.search(rf"##\s*{name}\s*(.*?)(?=\n##|\Z)", desc, re.S)
                sec[name.lower()] = clean_markdown(m.group(1)) if m else ""
            if sec["summary"]:
                out[r["ID"]] = sec
    return out


def present_options(item, shuffle=None, seed=None):
    """Option list for display. Answer identity is the Grambank numeric code; display order is deterministically shuffled."""
    opts = [{"value": o["value"], "label": o["label"].rstrip(".")} for o in item["options"]]
    if SHUFFLE_OPTIONS if shuffle is None else shuffle:
        random.Random(f"{SEED if seed is None else seed}:{item['item_id']}").shuffle(opts)
    return opts


_CIPHER_CACHE = {}


def _cipher_map(iso):
    """Lazily load the cipher table for this language (see cipher.py)."""
    if iso not in _CIPHER_CACHE:
        import cipher
        _CIPHER_CACHE[iso] = cipher.load(iso)
    return _CIPHER_CACHE[iso]


def format_corpus(pairs, condition="parallel", seed=0, pair_sep="\n", iso=None,
                  src_tag="EN"):
    """Sentence pairs -> corpus block in the prompt. TL = target language.

    Only the eng/tgt fields are used. Corpus files also have id/domain/topic/register;
    the id (e.g. "bouquet:dev:P001-S1") would reveal the corpus source to the model and
    is useless for judging grammar.

    Three conditions (ablation design, removing one kind of information at a time):
      parallel     C1  n correctly aligned pairs                -- main condition
      monolingual  C2  n target-language sentences only, no English -- is parallelism useful
      shuffled     C3  n pairs, but the TL side is misaligned   -- is sentence-level alignment useful

    src_tag is the line prefix of the known side (default "EN"). For a non-English pivot
    pass "FR"/"ZH"/...; sentences are read from the "src" field of pairs (old corpus
    files use "eng"; both are accepted).

    pair_sep controls the separator **between pairs**: "\n" (default, as in all runs so
    far) or "\n\n" (blank line between pairs). The default must not change, otherwise
    all earlier results become incomparable. The blank line tests whether the absence of
    one makes pair boundaries hard for the model to see.

    C3 shuffles the TL side with a **derangement** (no fixed points), which only breaks
    the pairing: the sentence set, token count and language composition are identical to
    C1, so any C1 vs C3 difference can only come from alignment. The prompt text is kept
    the same as C1 (the model is not told the corpus was shuffled), so this measures
    whether the model **actually uses** alignment, not what it does once told.
    """
    src = lambda p: p.get("src", p.get("eng", ""))   # noqa: E731

    if condition == "monolingual":
        return pair_sep.join(f"TL: {p['tgt']}" for p in pairs)
    if condition in ("cipher_mono", "cipher_shuf"):
        # The C2/C3 counterparts of C5: rearrange as in C2/C3, then encrypt the TL side.
        # The order of the two steps does not matter (the cipher is a per-character
        # bijection, independent of sentence order), but the **same** cipher table must
        # be used, otherwise C5 and its ablations are not the same "language".
        if iso is None:
            raise ValueError(f"condition={condition} requires iso")
        import cipher
        m = _cipher_map(iso)
        enc = [{"src": src(p), "tgt": cipher.apply_cipher(p["tgt"], m)}
               for p in pairs]
        base = "monolingual" if condition == "cipher_mono" else "shuffled"
        return format_corpus(enc, condition=base, seed=seed, pair_sep=pair_sep,
                             src_tag=src_tag)
    if condition == "cipher":
        # C5: the TL side is scrambled with a fixed per-language character-level
        # bijection; the EN side is untouched (otherwise the task is unsolvable).
        # The cipher table is built from the whole corpus, so n=16 and n=1024 use the
        # same one -- C1 and C5 are thus strictly paired: same language, sentences and
        # model; only the orthography changes.
        if iso is None:
            raise ValueError("condition=cipher requires iso")
        import cipher
        m = _cipher_map(iso)
        return pair_sep.join(
            f"{src_tag}: {src(p)}\nTL: {cipher.apply_cipher(p['tgt'], m)}"
            for p in pairs)
    if condition == "shuffled":
        tgts = _derange([p["tgt"] for p in pairs], seed)
        return pair_sep.join(f"{src_tag}: {src(p)}\nTL: {t}"
                             for p, t in zip(pairs, tgts))
    # The corpus body of C4 (parallel_named) must be **byte-identical** to C1 -- the
    # difference between them may only come from whether the prompt names the language,
    # not from the corpus layout, hence this shared branch.
    if condition not in ("parallel", "parallel_named", "cipher"):
        raise ValueError(f"unknown condition: {condition}")
    return pair_sep.join(f"{src_tag}: {src(p)}\nTL: {p['tgt']}" for p in pairs)


def _derange(xs, seed):
    """Derangement: return a permutation of xs in which every position gets a different element.

    The naive approach reshuffles until there are no fixed points; for large n this takes
    about e (~2.7) tries in expectation but has no worst-case bound. Here, after one
    shuffle, each remaining fixed point is swapped with the previous position, which is
    done in one pass and stays random. For n<2 a derangement is impossible and xs is
    returned unchanged.
    """
    if len(xs) < 2:
        return list(xs)
    idx = list(range(len(xs)))
    random.Random(seed).shuffle(idx)
    for k in range(len(idx)):
        if idx[k] == k:
            j = k - 1 if k > 0 else k + 1
            idx[k], idx[j] = idx[j], idx[k]
    assert all(idx[k] != k for k in range(len(idx))), "derangement failed"
    return [xs[i] for i in idx]


def build_prompt(item, desc, shown, corpus=None, condition="parallel",
                 src_name="English", src_tag="EN"):
    """corpus=None is the closed-book condition.

    When a corpus is given it goes at the **very beginning**: prompt caching is prefix
    matching and the 22 items of one language share the same corpus, so the cache is only
    hit if the corpus comes first and the question after.

    In corpus conditions the **language name is not given**: the aim is to test whether
    typological features can be inferred from the corpus, and naming the language would
    let the model fall back on closed-book knowledge. This anonymisation is weak for
    high-resource languages (the model recognises e.g. Turkish from a few sentences), so
    langid_control.py measures identification rates and results are analysed by group.

    The exception is C4 (condition="parallel_named"): both the corpus **and** the
    language name are given. It is the missing cell of the 2x2 design -- C0 gives the
    name without a corpus and C1 a corpus without the name, so their difference mixes
    "adding the corpus" and "removing the name". With C4:
      C4 - C1 = what the language name is still worth given the corpus
      C4 - C0 = what the corpus is still worth given the language name
    The difference between the sum of the two and (C1 - C0) is the interaction term.
    """
    named = condition == "parallel_named"
    lines = []
    if corpus:
        # C3 (shuffled) deliberately reuses the C1 header -- the model is not told the corpus is shuffled; this tests whether it really uses alignment
        if condition in ("monolingual", "cipher_mono"):
            head = ("Below are sentences in an unnamed target language, one per"
                    " line (TL = target language).")
        elif named:
            # Word-for-word identical to C1 except "an unnamed target language" is
            # replaced by the language name; no sentences are added or removed -- the
            # C1/C4 difference must come only from knowing the language name
            head = (f"Below is a parallel corpus between {src_name} and"
                    f" {item['iso_name']}. Each pair is one sentence: {src_tag} is"
                    f" {src_name}, TL is {item['iso_name']}.")
        else:
            # C5 (cipher) deliberately reuses the C1 header -- the model is not told the
            # orthography was scrambled, as with C3: this measures what the model can
            # actually get out of it, not how it copes once told "this is a cipher".
            head = (f"Below is a parallel corpus between {src_name} and an unnamed"
                    f" target language. Each pair is one sentence: {src_tag} is"
                    f" {src_name}, TL is the target language.")
        lines += [head, "", corpus, "", "---", ""]
    lines += [
        "You are a linguist answering a question about the grammar of a specific language.",
        "",
    ]
    if not corpus or named:
        lines += [f"Language: {item['iso_name']} (ISO 639-3: {item['iso639_3']})", ""]
    lines += [f"Question: {item['question']}"]
    if INCLUDE_SUMMARY:
        lines += ["", "What this question means:", desc["summary"]]
    if INCLUDE_PROCEDURE and desc.get("procedure"):
        lines += ["", "How this is decided:", desc["procedure"]]
    lines += ["", "Options:"]
    lines += [f"  {o['value']} = {o['label']}" for o in shown]
    codes = "/".join(o["value"] for o in shown)
    lines += [
        "",
        (("Answer based only on the sentences above."
          if condition in ("monolingual", "cipher_mono") else
          "Answer using the corpus above together with your knowledge of this"
          " language." if named else
          "Answer based only on the corpus above.") if corpus
         else "Answer based on your knowledge of this language.")
        + " You must choose exactly one option.",
        f"End your reply with a final line in the form: Answer: <{codes}>",
    ]
    return "\n".join(lines)


ANSWER_RE = re.compile(r"Answer\s*[:：]\s*\(?([0-9])\)?")


def extract_choice(text, valid_codes):
    """First look for 'Answer: N'; fall back to a valid code in the last non-empty line only.

    No full-text fallback: the body contains many numbers (GB430-GB433, cited years, ...),
    so a full-text fallback would misparse.
    """
    for m in reversed(list(ANSWER_RE.finditer(text))):
        if m.group(1) in valid_codes:
            return m.group(1)
    tail = [ln for ln in text.strip().splitlines() if ln.strip()]
    if tail:
        for m in reversed(list(re.finditer(r"\b([0-9])\b", tail[-1]))):
            if m.group(1) in valid_codes:
                return m.group(1)
    return None


def stratified_sample(items, n, seed=0):
    """Stratified sampling by feature, so that a small smoke test covers all features."""
    from collections import defaultdict
    by_feat = defaultdict(list)
    for it in items:
        by_feat[it["feature_id"]].append(it)
    rng = random.Random(seed)
    for v in by_feat.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < min(n, len(items)):
        for fid in sorted(by_feat):
            if i < len(by_feat[fid]) and len(picked) < n:
                picked.append(by_feat[fid][i])
        i += 1
    return picked
