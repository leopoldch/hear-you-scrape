# Detailed tables

`results/` holds the tables and figures made by the notebooks.
This folder holds the score tables of **every** experiment listed in [RESULTS.md](../../RESULTS.md), including
those whose code is not in the notebooks. One folder per section of RESULTS.md, one sub-folder per experiment.

Not included: the recordings, the lists of recordings, and the values measured by the authors of the datasets
(for example the friction of each surface). Get them from the datasets themselves, see `1_data.ipynb`.

| Folder | What is inside |
|---|---|
| `2_known_setup/` | first models on Cluster and LMT108 (SVM, CNN), same textures touched by other people |
| `3_new_setup/` | every attempt with no label of the new setup: first models, domain adaptation, pretrained encoders, simulated devices, CLAP with text, setup-to-setup transfer, same-name materials, the locked test on building materials, our phones |
| `4_few_labels/` | a few labeled surfaces: the locked test on HaTT, choosing the surfaces to label, adding taps, merging datasets |
| `5_fine_tuning/` | fine-tuning CLAP: leave one dataset out, weight mixing and edits, one source at a time, augmentations, number of setups, early stopping, recall per family |
| `6_synthetic/` | synthetic scrapes: 200, 2000, with the real datasets, and the volume curve |
| `7_friction/` | friction and other physical quantities between setups, and what the friction model reads |

Most tables have one row per test, with the score and, when we computed it, a 95 % interval (`low`, `high`).
