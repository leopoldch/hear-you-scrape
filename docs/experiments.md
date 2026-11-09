# Transfer to a new setup: everything we tried

The question: can a model recognize the material family on a setup (probe, sensor, gesture) it has never seen?
Development setups: Cluster, LMT108, SENS3. Score: balanced accuracy per recording, chance 0.25, 95% interval
from resampling whole surfaces. **k** = labeled surfaces per family of the new setup (k = 0: no label).
Each experiment had its prediction and success criterion written before it ran.

Scripts in `scripts/hub/` run on the GPU JupyterHub (`hub_job.py` uploads and starts them); the others run
locally. Figures and tables are in `analysis/classification/<folder>/`.

## No label of the new setup (k = 0): nothing transfers between our setups

| Why we tried it | What we did | Result | Code → results |
|---|---|---|---|
| Large pretrained models may already know materials | Linear probe on frozen BEATs, CLAP and every Qwen2-Audio layer, trained on 2 setups, tested on the 3rd | 0.13–0.38, almost all 0.20–0.30 | `hub/setup_transfer_probes.py` → `qwen_audio/` |
| Qwen may only be biased toward one answer | Qwen answers directly, raw, re-centered or re-calibrated | Says "polymer" 99.8% of the time, even on silence or noise: it does not listen | `hub/qwen_baseline_check.py` → `qwen_audio/` |
| Make Qwen answer from the audio | Edit its output head, in-context examples, QLoRA, hand edits of its weights | Learns the training setup (up to 0.70), stays at chance on a new one | dead end, code not kept; tables in `qwen_audio/` |
| Remove what identifies the setup | Linear erasure of the setup (INLP, mean) in Qwen | Setup no longer readable, family gone with it: 0.29 | dead end, code not kept; `qwen_audio/` |
| Simulate many setups | Random filters, levels, speed and noise on training audio | 0.24–0.26; known-setup score drops | `hub/augmented_beats_transfer.py` → `augmentation/` |
| Standard domain generalization | DANN, CORAL, contrastive across setups, synthetic probes, model soup on BEATs | 0.20–0.27 | `hub/domain_generalization_beats.py` → `domain_generalization/` |
| Audio and acceleration see the same contact | Shared audio ↔ acceleration space learned by contrast | Setup readability 0.97 → 0.42, but family lost too: 0.29 | `classification/cross_modal_contrastive.py` → `cross_modal/` |
| Maybe the labels differ in meaning | Easy pairs only (metal vs textile) across setups | 0.96 inside a setup, 0.53 across | `classification/label_consistency.py` → `label_consistency/` |
| Maybe the geometry between families is shared | Family distance matrices (RSA), Gromov-Wasserstein, rank-normalized descriptors | Geometry not shared; GW at chance; only acceleration flatness keeps its order | `classification/relational_transfer.py` → `relational_transfer/` |
| The web knows scraping sounds | Ask CLAP with text ("the sound of scraping wood"), balanced on unlabeled target recordings | LMT108 0.49 and Greatest Hits 0.48; Cluster and SENS3 at chance | `classification/clap_text_zero_shot.py` → `clap_text_zero_shot/` |
| A physics simulation can contain hundreds of setups | 250 simulated devices (sensor filter, probe modes, tip, speed, force, motor lines) | Works inside the simulation with 1 device (0.70); real setups at chance; calibration learned in simulation hurts | `hub/simulated_devices*.py` → `simulated_devices/` |

The night runs N1–N38 of a second agent (code on the hub, not in this repo) combined CLAP text, a probe trained
on AudioLDM synthetic scrapes and a hand-off to the accelerometer. LMT108 reaches about 0.60 and Greatest Hits
0.49 (fixed before the run; the same as CLAP text alone). Cluster stays near 0.33, SENS3 and a robot setup (YCB)
at chance. Its "surface vote" scores use the identity of test surfaces, so they are not strict k = 0. **The web
prior helps only when the recording sounds like web audio** (air microphone, hand gesture).

## A few labels of the new setup: this works

| Why we tried it | What we did | Result | Code → results |
|---|---|---|---|
| Accept a short calibration | Probe on k labeled surfaces of the new setup, pooled with the other setups or alone | Alone beats pooled by +0.06 to +0.10; fusing encoders adds nothing | `hub/fusion_few_shot_probes.py` → `qwen_audio/fusion_few_shot.png` |
| Final method, tested once | Frozen BEATs, acceleration, probe on the new setup only; locked test on HaTT | **k = 2: 0.64 [0.54–0.76]**, k = 1: 0.53; the k = 0 pooled reference is 0.42 | `classification/calibrated_new_setup.py` → `calibrated_new_setup/` |
| Get more from k surfaces | Window-level probe + self-training; fine-tuning BEATs | +0.03 to +0.05 (development only) | `classification/few_label_classifiers.py`, `hub/few_shot_finetune_beats.py` |

HaTT has only 3 wood surfaces, and polymer stays weak (0.33). HaTT and Greatest
Hits are now used up: no unseen setup is left for a family test.

## Why k = 0 fails

| Question | What we did | Result | Code → results |
|---|---|---|---|
| Is the family direction the same in every setup? | Probe from 2 labeled surfaces of one setup, applied to another; cosine of family directions | Cosine ≈ 0 across setups, ≈ 0.5 inside one | `classification/calibration_relation.py` → `calibrated_new_setup/` |
| Is the gap just acoustics (filter, probe, speed)? | Compare transfer in the 250 simulated devices with real setups | Simulation transfers well internally; the same probe is weak on real setups, suggesting the simulation misses relevant differences | `hub/simulated_devices_calibration.py` → `simulated_devices/` |
| Would tapping read the material better than scraping? | Greatest Hits, hits vs scratches on the same objects | 0.54 vs 0.49; a probe trained on hits reads scratches as well | `hub/tap_versus_scratch.py` → `tap_versus_scratch/` |
| Does transfer work when the surfaces are the same? | Devillard 2025: same 10 textures, 10 people, 4 sensors at once; recognize the texture (chance 0.10) | New person 0.70–0.77, other sensor of the same kind 0.49–0.70; acceleration ↔ audio at chance | `classification/devillard_same_surfaces.py` → `devillard_same_surfaces/` |

**Reading.** No surface appears in two of our setups, so a change of setup always comes with a change of
surface collection. When the surfaces are shared (Devillard), transfer works without labels. The failure between
our setups therefore does not prove that k = 0 is impossible. Devillard has textures, not families, and only bare
fingers, so it does not settle the question.

**Next experiment.** Record the same surfaces with different kinds of contact: 12 surfaces (3 per family) ×
3 contacts (metal tip, plastic tip, finger) × 2 sensors at once.
