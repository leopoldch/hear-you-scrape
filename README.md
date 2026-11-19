# hear-you-scrape

<p align="center"><img src="docs/poster.png" width="500" alt="Parody of the Alien (1979) poster: the egg is the spectrogram of a real scrape"></p>

*Contact: In Space No One Can Hear You Scrape.*

A surface can look like wood and be plastic. Sliding something on it gives another clue: its sound and its
vibration. We try to say if a surface is **wood, metal, textile or polymer** from a scrape.

The hard part is a **recording setup the model has never seen**: another tool, another sensor, another lab.
We use eight public datasets (steel tips held by hand, a bare finger, a machine, a drumstick in videos) and a
few recordings made with our phones.

Scores are balanced accuracy on surfaces never used for training. Chance is 0.25.
Every experiment, with its number and whether it passed, is in [RESULTS.md](RESULTS.md).

## 1. The datasets differ more than the families

Before any model we looked at the spectra. A PCA separates the two first datasets (Cluster and LMT108) much
better than the four families. Inside LMT108 the families do differ (wood has a bump at 150-400 Hz), in Cluster
they look almost the same. The Cluster vibration also contains lines from the machine (f = 3.5 x speed), and the
level rises 4-7 dB each time the speed doubles.

![PCA of the spectra, colored by family, by dataset and by speed](results/signals_pca.png)

## 2. A setup the model knows: it works

On new surfaces of the same setup, an SVM on spectral shapes and a small CNN both reach about 0.74 with
sound + vibration. Pretrained encoders (BEATs, CLAP) give 0.72-0.76.

## 3. A new setup with no label: it does not work reliably

Trained on Cluster and tested on LMT108 (or the reverse), every model is at chance. We tried about 40 ideas:
domain adaptation, random fake devices, 250 simulated devices, pretrained encoders, an audio language model.
All stay near chance.

The best one needs no training: CLAP compares the sound with sentences like "the sound of scraping wood".
It gives 0.43 on average, fine on some setups and at chance on others.

![CLAP with text prompts on each dataset](results/clap_text_scores.png)

We locked one test in advance: 184 building materials recorded with a tool we had never looked at.
We got 0.38 for a pass mark of 0.50.

What transfers depends on the **kind of contact**. Families pass only between the two steel tips held by hand.
Materials with the same name find each other between two steel-tip datasets (0.94, chance 0.5), but not between a
steel tip and the machine.

## 4. Training on other setups makes it worse

We fine-tuned CLAP on the other datasets. It gets better on setups it has seen (0.45 to 0.70) and worse on the
one it has not (0.43 to 0.31). The score falls from the first training steps and never comes back.

![Score on the new setup while fine-tuning on the others](results/fine_tuning_curve.png)

Nothing we tried fixes it: mixing the old and new weights, editing the weights, many augmentations, unfreezing
the layers step by step, training on 1 to 4 setups, adding 200 to 20000 synthetic scrapes, a lower learning rate.

## 5. Two labeled surfaces per family are enough

If we label two surfaces per family on the new setup and train a small classifier on frozen BEATs vectors, we get
0.64 on a locked test (0.42 with no label). Data from the other setups does not help here either: 0.61 alone,
0.56 with the others.

![A few labeled surfaces of the new setup](results/few_labels_locked_test.png)

Two things help more: choosing which surfaces to label with k-means, and adding taps to the scrapes. With 8 to 12
labeled surfaces we reach 0.61 to 0.91 depending on the dataset. This last part still needs a locked test.

## 6. Friction crosses setups better than families do

Instead of a family we predict a measured quantity. A model trained on steel-tip vibrations ranks the surfaces
of a bare-finger dataset by friction (0.59 and 0.52, rank correlation), and again on a second one (0.87 on 10
textures). No label of the new setup is used.

![Friction of a bare finger predicted by steel-tip models](results/friction_steel_tip_to_finger.png)

It has clear limits: it fails with sound (0.28 and 0.03), and on the machine dataset the sign is inverted (-0.52).
The model mostly follows the vibration level and the speed of the gesture. This part is exploratory.

## Notebooks

| Notebook | What it does |
|---|---|
| `1_data.ipynb` | downloads the datasets and cuts the recordings into short windows |
| `2_signals.ipynb` | spectra of sound and vibration, machine noise, effect of speed |
| `3_known_setup.ipynb` | new surfaces of a setup the model knows |
| `4_new_setup.ipynb` | a new setup with no label |
| `5_few_labels.ipynb` | a new setup with a few labeled surfaces |
| `6_fine_tuning.ipynb` | fine-tuning CLAP on the other setups |
| `7_synthetic.ipynb` | synthetic scrapes as training data |
| `8_friction.ipynb` | predicting friction instead of a family |

Heavy steps save their output in `cache/`, so a notebook is fast the second time.
Tables and figures go to `results/`.

## Data

Not in the repo. `1_data.ipynb` gets it from the authors:
[Cluster](https://doi.org/10.6084/m9.figshare.29438288),
[LMT108 and LMT69](https://zeus.lkn.ei.tum.de/downloads/texture/),
[HaTT](https://repository.upenn.edu/handle/20.500.14332/30085),
[TUM building materials](https://mediatum.ub.tum.de/1474014),
[SENS3](https://www.sens3.net/surfaces),
[Devillard 2025](https://doi.org/10.6084/m9.figshare.26965267),
[Greatest Hits](https://andrewowens.com/vis/hits.html).

## Run

```
uv sync
uv run jupyter lab
```
