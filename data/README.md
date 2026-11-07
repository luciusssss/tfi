# Data

Items for typological feature induction (TFI): given a sample of English–X sentence pairs, a
model answers a Grambank question about language X. Gold answers are Grambank values.

## Contents

```
languages.csv                 195 target languages (ISO 639-3, Glottocode, family, script,
                              source corpus configs, number of core items, set membership)
grambank/grambank_parameters.csv
                              Grambank feature definitions, used to build the prompts
core/items.jsonl              Core set: 22 features x 195 languages, 4,084 items
latin_script/items.jsonl      Latin-script set: core items of the 127 Latin-script languages
                              the cipher applies to, 2,657 items
latin_script/item_ids.txt     the same item IDs, one per line
latin_script/ciphers/<iso>.json
                              cipher table for each of the 127 languages, {"map": {plain: cipher}}
extended/items.jsonl          Extended set: every Grambank feature coded for the 24 languages
                              none of the models identifies, 3,775 items over 195 features
corpus_ids/<iso>.s<seed>.txt  sampled sentence IDs per language and corpus seed (0, 1, 2),
                              in sampling order, up to 1,024 per file
```

A corpus of n pairs is the first n IDs of a file, so corpora of different sizes are nested.

## Item format

Each line of `items.jsonl` is one item:

```json
{"item_id": "aar-GB024", "iso639_3": "aar", "glottocode": "afar1241", "script": "Latn",
 "feature_id": "GB024", "feature_group": "Word order",
 "question": "What is the order of numeral and noun in the NP?",
 "options": [{"key": "A", "value": "1", "label": "Num-N"}, ...],
 "gold_value": "1", "gold_key": "A", "gold_label": "Num-N", ...}
```

`options` lists every value Grambank defines for the feature, including values unattested in
our sample. Options are shuffled at prompt time.

## Cipher tables

Each table is a bijection over the lowercase letters of one language's target-side text,
built from its full corpus. Uppercase letters follow from lowercase, and digits, punctuation
and combining marks are left unchanged. Seven Latin-script languages (Vietnamese, Duala, Ewe,
Fon, Kxoe, Hassaniyya and Dar Daju Daju) are not in the set because their orthographies need
more letters than the cipher's character pool holds.

## Corpus text

FLORES+ and BOUQuET are gated datasets, so this release contains sentence IDs only. IDs have
the form `flores:<split>:<id>` or `bouquet:<split>:<uniq_id>`. After accepting the terms of
use of both datasets on the Hugging Face Hub, `fetch_corpora.py` in the repository root
downloads them at the revisions we used and rebuilds the exact corpora.

## Sources, versions and licenses

| source | version | used for | license |
|---|---|---|---|
| Grambank | v1.0.3 (CLDF, [grambank/grambank@7ae000c](https://github.com/grambank/grambank/tree/v1.0.3)) | feature definitions and gold values | CC BY 4.0 |
| Glottolog | v5.3 (CLDF, [glottolog/glottolog-cldf@072ca0d](https://github.com/glottolog/glottolog-cldf/tree/v5.3)) | Glottocode to ISO 639-3 mapping, families | CC BY 4.0 |
| FLORES+ | Hugging Face revision `b3a5298db5721c8a682e7ef00a37fcc9ab522757` | parallel sentences, referenced by ID only | CC BY-SA 4.0 |
| BOUQuET | Hugging Face revision `9a6070a9652e350dda1d353c4fd198533199a911` | parallel sentences, referenced by ID only | CC BY-SA 4.0 |

The Grambank and Glottolog files were downloaded on 2026-07-27 and are identical to the
tagged releases above. Language names follow the ISO 639-3 code tables downloaded the same day.

The items are derived from Grambank and are released under CC BY 4.0, for research use.
