#!/usr/bin/env python3
"""Character-level cipher for the Latin-script set (the CIPHER-* conditions in the paper).

Each language has a fixed bijection over the lowercase letters of its target-side text,
stored in data/latin_script/ciphers/<iso>.json. Letters are mapped in groups that share a
base letter: if c maps to f, then č and ć map to other variants of f, so variants of one
letter stay variants of one letter. Vowels map to
vowels and consonants to consonants, no letter maps to itself, and images are drawn from
ASCII, Latin-1 Supplement and Latin Extended-A. Uppercase follows from lowercase.

The tables were built once from the full corpus of each language, so every corpus size and
seed uses the same table. The English side is never changed.
"""
import json
import os

import item_prompts as ip

CIPHER_DIR = os.environ.get("CIPHER_DIR", os.path.join(ip.EXP_DIR, "ciphers"))


def load(iso):
    """The lowercase mapping {plain: cipher} for one language."""
    with open(os.path.join(CIPHER_DIR, f"{iso}.json")) as f:
        return json.load(f)["map"]


def apply_cipher(text, mapping):
    """Replace code point by code point, without Unicode normalisation.

    Combining marks, digits, punctuation and unmapped characters are kept as they are, so
    the ciphertext has the same length, word count and word-length distribution as the
    plaintext.
    """
    out = []
    for ch in text:
        rep = mapping.get(ch.lower())
        if rep is None:
            out.append(ch)
        elif ch.isupper():
            out.append(rep.upper())
        else:
            out.append(rep)
    return "".join(out)
