# Data card: ESCI (Shopping Queries Dataset)

All numbers below come from `reports/data_stats.json` (written by `psr prepare`, 2026-11-02, data
sha `b09cb98146bfc0f25509bb3e2c7f518430e73b1cf6f7168eca61b3451aa4e177`).

## Source and licence
Amazon Shopping Queries Dataset (ESCI), https://github.com/amazon-science/esci-data, folder
`shopping_queries_dataset/`. Licence: Apache-2.0. The files are downloaded by `psr fetch` and are
not redistributed in this repository; only `tests/fixtures/esci_sample.parquet` (200 queries, text
cut to 300 characters) is committed.

Citation: Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
Product Search", arXiv:2206.06588, 2022.

## Filters
- `small_version == 1` (the reduced set that ESCI Task 1 uses).
- Locales `us` and `es`. Japanese is left out: BM25 would need a Japanese tokenizer, and two
  locales are enough to show cross-language behaviour.
- Examples are joined to products on `(product_id, product_locale)`. HTML tags are removed from
  text fields and whitespace is collapsed; raw lengths are kept (`*_raw_len`).

## Splits
The official `split` column gives train and test. Validation is the train queries with
`((query_id * 2654435761) % 4294967296) // 65536 % 10 == 0`, so a query is in exactly one split
(tested in `tests/test_data.py`).

| Split | Locale | Queries | Judgements |
|---|---|---|---|
| train | us | 18,807 | 378,691 |
| train | es | 5,039 | 136,879 |
| val | us | 2,081 | 40,962 |
| val | es | 593 | 16,012 |
| test | us | 8,956 | 181,701 |
| test | es | 2,417 | 65,883 |

Totals match the esci-data README within 0.5% (SPEC 4.2): us 29,844 queries / 601,354
judgements, es 8,049 queries / 218,774 judgements. Unique products kept: us 482,105, es 167,761.

## Label distribution (judgements)
| Split | Locale | E | S | C | I |
|---|---|---|---|---|---|
| train | us | 164,590 | 133,115 | 17,114 | 63,872 |
| train | es | 56,363 | 45,686 | 11,210 | 23,620 |
| val | us | 17,229 | 14,513 | 1,976 | 7,244 |
| val | es | 6,768 | 5,114 | 1,278 | 2,852 |
| test | us | 79,708 | 63,563 | 8,099 | 30,331 |
| test | es | 27,074 | 22,458 | 4,902 | 11,449 |

## Known issues found during prepare
- Products with an empty title: 0.
- Products with an empty brand: 35,643; with an empty description: 306,846.
- Duplicate (query, product) pairs: 0.
- Products whose bullet points are longer than 2,000 characters: 10,555.
- Query texts that appear in two splits of the same locale: 0.
