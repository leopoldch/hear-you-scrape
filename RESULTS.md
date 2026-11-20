# All our experiments, one line each

Task: say if a surface is **wood, metal, textile or polymer** from the sound or the vibration of a scrape.
The hard part is a recording setup the model has never seen.

How to read the tables:

- Scores are balanced accuracy on surfaces never used for training. Chance is 0.25 (0.33 with three families).
  Friction results are rank correlations (0 = nothing, 1 = perfect).
- **k** = number of labeled surfaces per family taken from the new setup. k = 0 means no label at all.
- **Rule first?** `locked` = pass mark written before, on data never scored before, read once.
  `yes` = pass mark written before the run. `no` = exploratory.
- **Worked?** is about the pass mark when there was one.
- The score tables behind every line are in [results/details/](results/details/README.md).

Datasets: Cluster (machine), LMT108 and LMT69 (steel tip held by hand), HaTT (steel tip), TUM building materials
(steel tooltip), SENS3 (bare finger), Devillard (bare finger), Greatest Hits (drumstick, video sound), YCB (robot),
and our own phone recordings.

## 1. What the signals look like (`signals.ipynb`)

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| Do the spectra group by family or by dataset? (PCA, three clustering methods) | By dataset (ARI 0.80-0.86 with four groups), almost not by family (about 0.1) | - | no |
| Clustering inside one dataset | Cluster: wood and most textiles are isolated (ARI 0.63-0.74). LMT108: only wood (about 0.3) | - | no |
| What explains the variance of the Cluster spectra? | Sound: surface 33 %, speed 32 %, direction 8 %. Vibration: speed 26 %, direction 23 %, surface 17 %. Family 5-10 % | - | no |
| Do families differ inside one dataset? | Yes in LMT108 (wood has a 150-400 Hz bump), almost not in Cluster | - | no |
| Is there machine noise in the Cluster vibration? | Lines at multiples of 3.5 x speed hold 45-73 % of the 10-600 Hz energy | - | no |
| Can NMF isolate the machine? | Two narrow templates are active on all 8 surfaces: a candidate, not validated | partly | no |
| Does speed only shift the frequencies? | No. Spectra at 20 and 60 mm/s match better without rescaling (0.29-0.45 vs about 0.05). Level about +5.5 dB per doubling | - | no |
| Do sound and vibration tell the same thing? | Coherence about 0.01 in LMT108, 0.10 in Cluster | - | no |

## 2. A setup the model knows (`known_setup.ipynb`)

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| New surfaces of a known setup (Cluster), SVM and small CNN | About 0.74 with sound + vibration | yes | no |
| Same with pretrained encoders (BEATs, CLAP) | 0.72-0.76 | yes | no |
| Easier label sets inside one setup | Three families 0.84, metal vs textile 0.96 | yes | no |
| CLAP fine-tuned on one setup, new surfaces of it | 0.75 (0.45 before fine-tuning) | yes | yes |
| Same 10 textures touched by other people (Devillard, chance 0.10) | New person 0.70-0.77; other sensor of the same kind 0.49-0.70; vibration to sound 0.17-0.21 | yes | yes |

## 3. A new setup, no label (`new_setup.ipynb`)

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| Train on Cluster, test on LMT108 (and the reverse) | Near chance for every model (0.25-0.31) | no | no |
| Train on both, test on SENS3 | 0.30-0.35 | no | no |
| Usual fixes: more layers, AdaBN, mixup, noise, DANN | 0.24-0.34 | no | no |
| Two setups for training, test on the third; add Greatest Hits | 0.16-0.35 | no | no |
| Remove the mean spectrum of each setup | 0.26 on average (best case 0.39) | no | no |
| Temporal and impulsiveness features | 0.19-0.30 | no | no |
| Probes on BEATs, CLAP and every layer of Qwen2-Audio | 0.13-0.38. Qwen answers "polymer" 99.8 % of the time | no | yes |
| Train with random fake devices | 0.23-0.26 | no | yes |
| Domain generalization (DANN, CORAL, contrastive, model soup) | 0.23-0.25 | no | yes |
| Editing or adapting Qwen2-Audio | At most 0.30; final test on SENS3 0.23 (mark 0.35) | no | yes |
| One shared space for sound and vibration | 0.29 (mark 0.35) | no | yes |
| Why: is the family direction shared between setups? | No. A probe gives 0.57-0.68 inside a setup and 0.19-0.25 across | - | no |
| 250 simulated devices | 0.70 inside the simulation, 0.22-0.40 on real setups | no | yes |
| About 35 more ideas (transduction, optimal transport, co-training, self-training, other encoders...) | None passes. N29 passes on Greatest Hits (0.49) but equals CLAP text alone | no | no |
| Five more ideas (gain-free features, naming clusters, tap sound...) | All fail. If the clusters were named right: 0.68-0.80 | no | yes |
| CLAP with text prompts, vectors centered on the new setup | LMT108 0.53, LMT69 0.65, Greatest Hits 0.48, our phone 0.83; TUM 0.34, SENS3 0.32, Cluster 0.24. Mean of the six public datasets 0.43 | partly | yes |
| Same on our phone, 5 objects | 21 of 31 takes right; survives noise, fails with a 1 kHz low-pass | yes | no |
| Same on a colleague's phone | 0.45 [0.33-0.58], mark 0.50 | no | locked |
| Same on robot pokes (YCB) | 0.31 | no | yes |
| Train on one setup, test on another, all 12 pairs (vibration) | Only LMT108 and HaTT transfer (0.43 and 0.35). The ten other pairs 0.11-0.30 | partly | yes |
| Do materials with the same name find each other? (0.5 = chance) | LMT69 and LMT108 0.94; HaTT and LMT108 0.74; LMT108 and TUM 0.66; with Cluster 0.38-0.45 | yes | yes |
| Same lab, other tool: LMT69 to LMT108 on new surfaces | 0.64 [0.50-0.76]; the reverse 0.46 | partly | no |
| Locked test on 184 building materials (TUM) | 0.38 [0.27-0.48], mark 0.50. Wood recall 0.15 | no | locked |
| More training materials with the same contact | HaTT 0.48 to 0.65, LMT108 0.40 to 0.47, TUM stays 0.38 | partly | no |
| Many more materials, splitting polymer, a learned projection | No gain (plateau at 4-8 materials per family) | no | no |
| Translate one contact into another with about 30 shared materials | HaTT 0.24 to 0.51; needs 24-32 shared materials | partly | no |
| Taps from one tool to another | 0.09 | no | yes |
| Without polymer | CLAP 0.58; BEATs 0.66-0.75 between hand-held setups, fails on TUM and Cluster. Decided after seeing the results | partly | no |

## 4. A few labels of the new setup (`few_labels.ipynb`)

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| Labels of the new setup alone, or pooled with the other setups? | Alone is better: 0.49 / 0.56 at k = 1 / 2, pooled 0.39 / 0.48 (mean of three setups) | yes | no |
| Domain generalization + 4 labels | 0.63-0.67 (mark 0.68) | no | yes |
| Locked test on HaTT: 2 labeled surfaces per family | **0.64 [0.54-0.76]** (k = 1: 0.53; no label: 0.42). Mark 0.45 | yes | locked |
| Same on TUM (three families) | k = 2: 0.61 [0.44-0.77]; k = 8: 0.76 | yes | no |
| CLAP: 2 labels alone, or with the other datasets? | Alone 0.61, with the others 0.56, frozen probe 0.51 | no | yes |
| CLAP: one dataset alone, or the six merged? (new surfaces) | Alone 0.75, merged 0.70 | no | yes |
| Choose which surfaces to label with k-means | +0.15 to +0.17 at 8 labels on HaTT, LMT108, TUM; +0.04 Cluster; -0.05 SENS3. About as good as 2 per family, without knowing the families. Greatest Hits +0.08 [0.00-0.15] (earlier run) | yes | yes |
| Add taps to the scrapes, k = 2 | SENS3 0.42 to 0.66, TUM 0.63 to 0.69. Earlier run: LMT108 0.62 to 0.76, Greatest Hits +0.06 (their taps are not in the notebook) | yes | yes |
| Both together, 8 then 12 chosen surfaces | SENS3 0.58 / 0.65, TUM 0.64 / 0.72; earlier run LMT108 0.86 / 0.91. No locked test | yes | no |
| Taps inside one setup | TUM: taps 0.89, scrape 0.71. SENS3: scrape + taps 0.72, scrape 0.51 | yes | no |

## 5. Fine-tuning CLAP on the other setups (`fine_tuning.ipynb`)

The original CLAP (text prompts, no training) gives 0.43 on average on a setup never seen.

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| Light adaptation on frozen vectors | 0.47 to 0.48 at best, 0.36 at full strength | no | yes |
| Fine-tune the whole sound encoder on the other datasets | Unseen setup **0.43 to 0.31**. New surfaces of seen setups 0.45 to 0.70 | no | yes |
| 20 ways to edit the fine-tuned weights without gradients | Best 0.68 seen / 0.37 unseen. None keeps both | no | yes |
| Mix original and fine-tuned weights (0, 25, 50, 75, 100 %) | Unseen 0.43 / 0.41 / 0.37 / 0.33 / 0.31. Every edit sits on this curve | no | yes |
| Which dataset hurts which? | No dataset helps another. SENS3 -0.14, TUM -0.10, Cluster -0.10 | no | yes |
| Route between the two models with a "known setup" detector | 10 right of 12: 0.72 seen / 0.39 unseen | no | yes |
| Equalize each setup's frequency bands | 0.32 to 0.34 | no | yes |
| Many augmentations, layers unfrozen step by step | 0.30 with either or both, 0.38 with neither (original 0.49 on 7 targets) | no | yes |
| Which families does fine-tuning destroy? | Metal and textile first; each source destroys a different family | - | no |
| Fine-tune on 1, 2, 3, 4 setups | 0.35 / 0.32 / 0.32 / 0.31 | no | yes |
| Stop early, or use a 10 times lower learning rate | Falls from step 50 and never rises. Low rate ends at 0.36 | no | yes |
| Best "robust" model on the colleague's phone | 0.53 [0.37-0.69], mark 0.55 | no | yes |

## 6. Synthetic scrapes (`synthetic.ipynb`)

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| 200 scrapes from a small physics model, no real data | Probe 0.37, fine-tuned CLAP 0.41 (original 0.43) | no | yes |
| 2000 scrapes, much more variety | Probe 0.39, fine-tuned 0.37-0.39 | no | yes |
| Real datasets + 2000 synthetic scrapes | Unseen public datasets 0.38 (0.36 without). Colleague's phone 0.65, mark 0.55, but weight mixing alone already gives 0.60 | partly | yes |
| 200, 2000 or 20000 scrapes, half of every batch | 0.34 / 0.35 / 0.34 (0.31 with none). An earlier run with 3 seeds, and one trained 4 times longer, gave the same | no | yes |

## 7. Friction and other physical quantities (`friction.ipynb`)

Here the model predicts a measured quantity instead of a family. Vibration only, no label of the new setup.

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| Friction between steel-tip setups | HaTT to LMT69 0.50, LMT69 to HaTT 0.39. Holds in 4 directions of 6 | yes | no |
| Steel tip to bare finger | HaTT to SENS3 **0.59 [0.35-0.76]**, LMT69 to SENS3 0.52. But TUM to SENS3 -0.55 | partly | no |
| Second bare-finger dataset (Devillard, 10 textures) | 0.87 and 0.89 | yes | no |
| Does it depend on our settings? | Positive in all 9 settings (0.31-0.62). 96-100 % of 24 settings | yes | no |
| Same with sound instead of vibration | 0.28 and 0.03, mark 0.30 | no | yes |
| Same on the machine dataset (Cluster) | **-0.52 [-0.65, -0.36]**: the sign is inverted | no | yes |
| Same texture, ten people (Devillard) | 0.36 [-0.15, 0.70] | no | yes |
| What does the model read? | At a fixed material it follows the speed of the gesture (0.81), not the friction. Steel tip to finger goes through the vibration level | - | no |
| Density learned on TUM | Ranks metal first elsewhere (AUC 0.81 and 0.84) | yes | no |
| Rigidity from taps, heat flow | Rigidity 0.35 and 0.42. Heat flow does not transfer | partly | no |
| Four measured quantities to family, inside SENS3 | 0.74 (vibration alone 0.51) | yes | no |
| Predicted quantities to family, across setups | 0.19-0.38 | no | yes |

## 8. The proposal: velocity consistency (`velocity_consistency.ipynb`)

Cluster only, five folds of surfaces. 95 % intervals from drawing the surfaces again.

| Question | Result | Worked? | Rule first? |
|---|---|---|---|
| Frozen encoders + logistic regression, new surfaces | BEATs: sound 0.72, vibration 0.63, both 0.74. CLAP: 0.67, 0.58, 0.69 | - | no |
| BEATs fine-tuned, new surfaces, without / with the consistency penalty | 0.75 [0.68-0.82] / 0.75 [0.68-0.82] | no | no |
| Same, test at a speed never seen in training (60 mm/s) | 0.68 [0.62-0.74] / 0.68 [0.62-0.74] | no | no |
| Per family (fine-tuned, new surfaces) | F1: wood 0.96, textile 0.92, metal 0.65, polymer 0.50 | - | no |

## In short

- The penalty of our proposal (same answer at two speeds) changes nothing, and an unseen speed costs only 0.07.
- No label of the new setup: nothing works reliably. Training on other setups makes CLAP worse (0.43 to 0.31).
- Two labeled surfaces per family of the new setup: 0.64 on a locked test. Other setups do not help.
- What transfers depends on the kind of contact. Friction crosses from steel tip to finger, but not to the machine
  and not with sound.
