# Ground Reaction Force Estimation from a Single Wrist-Worn IMU Using Biomechanics-Informed Deep Learning

Code accompanying the paper. This repository contains the sagittal-plane three-link
upper-body kinematics model, the proposed Biomechanics-Informed Neural Network (BINN),
the ablation variants, the CNN and Transformer baselines, the training and evaluation
scripts, and the scripts and complete results of the CoM-pathway corruption and
channel-removal analyses.

---

## 1. Repository layout

```
matlab/
  1_preprocessing/
    segment_strides.m   Stride extraction from wrist gyroscope
                                                    local minima; resampling to T = 101
  2_upper_body_model/
    fit_model_parameters.m        Dataset A: segment peak-to-peak
                                                    amplitudes and angular offsets
                                                    -> ratios r_ES/WE, r_SC/WE and
                                                    offsets theta_0,i   (Eq. 1-2, Fig. 3)
    upper_body_model.m                        Three-link model; forearm angle from
                                                    gyroscope integration; CoM-relative and
                                                    absolute CoM acceleration (Eq. 3-4, Fig. 4)
    estimate_com_acceleration.m
                                                    CoM acceleration estimation, stride-wise
    perturb_torso_offset.m                       Regenerates the CoM input with a
                                                    perturbed torso angular offset
                                                    (Section III-B sensitivity analysis)
  3_dataset_build/
    build_dataset.m                        Merges GRF / wrist IMU / model output
                                                    into the .mat structs used by Python
    subsample_training_data.m                       Random stride-wise subsampling to
                                                    5 / 10 / 20 / 50 % (3 seeds each)
  4_metrics/
    paper_data_processing.m                         Per-stride NRMSE(range) and Pearson r
                                                    from the prediction CSVs; produces the
                                                    per-subject error tables behind every
                                                    number reported in the paper

python/                       Flat package; all modules import each other by name,
                              so run scripts from inside this directory.
  main.py                     Entry point: builds dataset, runs LOSO, trains, evaluates
  sweep.py                    Cartesian product of data_variants x dataset_variants
  utils_and_modules.py        Dataset assembly, LOSO loaders, scaler, training loop, NRMSE
  matlab_data_prep.py         Loads the MATLAB .mat structs
  models_*.py                 BINN, ablation variants, CNN, Transformer, ANN, LSTM
  tune_*.py                   Random hyperparameter search (30 trials per model)
  eval_day1_on_full.py        Evaluates the trained folds and writes the per-stride
  eval_percent_models_on_full.py  prediction CSVs consumed by paper_data_processing.m
  eval_noise_r_targeted.py    CoM-pathway corruption analysis
  eval_offset_perturbation.py Torso offset perturbation at inference
  make_aux_ablation_table.py  CoM channel-removal statistics and tables
  configs/                    One config per row of Table II, plus revision configs

results/                      Result tables (CSV)
```

All tables in `results/` were produced with the **final-epoch** checkpoints, matching the
evaluation protocol stated in Section II-B.3 and used for every number in the paper.
They contain model-accuracy metrics (NRMSE and Pearson r) only. No force-plate, IMU or
motion-capture recording is included; see Section 8.

---

## 2. Reproducing the paper

### 2.1 MATLAB pipeline

Run in order. Paths at the top of each script must be set to the local data location.

1. `segment_strides.m` — segment continuous walking into
   strides using wrist gyroscope local minima; resample each stride to 101 samples.
2. `fit_model_parameters.m` — derive the population-level amplitude
   ratios and angular offsets from Dataset A.
3. `upper_body_model.m` — apply the three-link model to Dataset B and produce the
   CoM-related acceleration used as the biomechanical input.
4. `build_dataset.m` — assemble `stride_grf`, `stride_kinematics_arm`, and
   `stride_modeling`.
5. `subsample_training_data.m` — generate the reduced-data variants.

Output `.mat` files are consumed directly by the Python code.

### 2.2 Paths

The scripts take repository-relative defaults and read four environment variables, so
nothing has to be edited to point them at a local copy of the data:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `../data` | `.mat` files written by `build_dataset.m` |
| `RUNS_ROOT` | `../runs` | training output root |
| `OUTPUT_ROOT` | `../eval_outputs` | per-stride prediction CSVs; input of `paper_data_processing.m` |
| `PERTURB_DIR` | `../data/perturb` | perturbed `stride_modeling` files |

`BASE_DIR` / `OUT_DIR` at the top of the MATLAB scripts and `target_root_folder` in
`paper_data_processing.m` play the same role and must match.

### 2.3 Python training

```bash
cd python
python main.py  --config configs/proposed.yaml      # single condition
python sweep.py --config configs/proposed.yaml      # all sweep combinations
```

Each run writes `metrics.json`, `test_summary.json`, per-stride prediction CSVs, and
`config_used.yaml` to `runs/<model>/<timestamp>_<name>/`.

### 2.4 Evaluation protocol

Leave-one-subject-out cross-validation over 19 subjects, seed 98, 200 epochs, AdamW,
MSE loss, **weights from the final epoch** (no early stopping, no scheduler). The
z-score scaler is fit on the training subjects of each fold only.

Seven training configurations are defined:

| Configuration | Train | Test | Regime |
|---|---|---|---|
| `tv_te_ss1` | slow | slow | ID |
| `tv_te_ss2` | moderate | moderate | ID |
| `tv_te_ss3` | fast | fast | ID |
| `tv_te_ss123` | all three | all three | ID |
| `tv_ss1__te_ss2ss3` | slow | moderate, fast | OOD |
| `tv_ss2__te_ss1ss3` | moderate | slow, fast | OOD |
| `tv_ss3__te_ss1ss2` | fast | slow, moderate | OOD |

Splitting the evaluation by test session yields **6 ID and 6 OOD evaluation conditions**.
Session labels map to walking speed as `ss1 = 1.0`, `ss2 = 1.25`, `ss3 = 1.5 m/s`.

Training data are reduced to 5 / 10 / 20 / 50 / 100 % by random stride-wise subsampling;
fractions below 100 % are averaged over three independent subsamplings.

Stride counts by speed (Dataset B, 19 subjects): 1,658 at 1.0 m/s, 1,906 at 1.25 m/s,
and 2,089 at 1.5 m/s, for 5,653 in total. Two subjects each lack one session (one at
1.25 m/s, one at 1.5 m/s), so 17 subjects have data at all three speeds.

---

## 3. Configuration to paper mapping

Every config in `python/configs/` was extracted from the actual run that produced the
reported numbers; hyperparameter blocks for model types the config does not select were
dropped. Parameter counts were verified against Table II.

| Config file | Paper | Model type | Parameters |
|---|---|---|---|
| `proposed.yaml` | Proposed BINN | `binn`, mode `proposed` | **111,788** |
| `ablation_wo_biomech_arch.yaml` | w/o Biomechanics-informed Architecture | `cnn`, CoM channels concatenated at the input | 268,418 |
| `ablation_wo_cross_attention.yaml` | w/o Cross-Attention | `binn`, mode `no_attn` | **111,800** |
| `ablation_wo_com_to_grf_mapper.yaml` | w/o CoM-to-GRF Mapper | `binn`, mode `attn_only` | **111,732** |
| `baseline_cnn.yaml` | CNN baseline | `cnn`, wrist IMU only | **268,418** |
| `baseline_transformer.yaml` | Transformer baseline | `transformer`, wrist IMU only | **398,722** |

Ablation variants use `decoder_channels_by_mode` so that all variants keep a parameter
count comparable to BINN, as stated in Section II-B.2. The three variants are matched to
within 0.06 % (111,788 / 111,800 / 111,732).

`ablation_mode: proposed` selects the proposed full architecture, and `proposed` is the
name used for its run output directory. `binn` and the historical internal name
`no_physics` are accepted as aliases and produce an identical model, so configs and
checkpoints written before this renaming still load unchanged. The evaluation scripts
look for a `proposed/` run directory first and fall back to `no_physics/`.

### Input channels

| | Wrist IMU | CoM-related acc. | Stride duration | `in_dim` |
|---|---|---|---|---|
| BINN and ablation variants | acc x,y,z + gyro x,y,z | AP, VT | 1 | 9 |
| CNN, Transformer baselines | acc x,y,z + gyro x,y,z | — | 1 | 7 |
| Sacrum baseline (Section III-B) | sacrum acc x,y,z + gyro x,y,z | — | 1 | 7 |

The sacrum baseline uses the **IMU** signals (`imu_acc_local_*`, `imu_gyro_local_*`)
only; no marker-derived kinematics are used.

---

## 4. Revision analyses

All commands are run from inside `python/`.

### 4.1 CoM-pathway corruption (Section III-B)

```bash
python eval_noise_r_targeted.py --runs_root <trained_runs> --out_csv noise_results.csv
```

Trained models are held fixed and only the inference-time CoM channel is corrupted:

```
z' = r * z + sqrt(1 - r^2) * n
```

where `z` is the stride-standardised channel and `n` is low-frequency noise (Gaussian
smoothing, sigma = 10 frames), **centred to zero mean and orthogonalised against `z`
within each stride**. Both steps are required:

- Without orthogonalisation the finite-sample `corr(z, n)` is not zero — smoothing
  reduces the effective degrees of freedom from 101 to about 10, so `|corr(z, n)|`
  averages 0.42 — and the achieved correlation scatters widely across strides
  (SD 0.43 at a target of 0.36).
- Without centring, `mean(n)` is not zero and a DC offset of up to 0.78 stride-SD is
  introduced, growing as the corruption becomes more severe and confounding the
  dose-response.

With both applied, the achieved correlation equals the target exactly for every stride
(SD < 1e-14 over 5,653 strides) and the per-stride mean and standard deviation of the
original channel are preserved. The same orthogonalised noise waveform is reused at
every corruption level so that adjacent-level differences reflect only the change in
mixing coefficient. Random seeds are deterministic across processes (`blake2s`).

Note that `r` here is the correlation between the corrupted channel and the uncorrupted
**model-derived** CoM channel. It is not the correlation reported in Table III, which
compares the model-derived CoM estimate with the measured reference.

Results: `results/noise_final_all.csv` (207,360 rows).

### 4.2 CoM channel removal (Section III-B)

```bash
python sweep.py --config configs/config_aux_ablation.yaml
python make_aux_ablation_table.py --runs_root <trained_runs>
```

Removes the AP or VT CoM channel from the input and retrains with identical folds, seed
and protocol. 5 data fractions x 7 training configurations x 3 input combinations = 105
runs. Results: `results/ch_all.csv` (per channel), `results/test_all.csv` (channel mean).

`make_aux_ablation_table.py` produces the reported statistics from the final-epoch test
NRMSE(range).

### 4.3 Torso offset perturbation (Section III-B)

```matlab
MODE='add'; SEG='sc'; DELTA_LIST=[-30,-15,15,30];
run('matlab/2_upper_body_model/perturb_torso_offset.m')
```

```bash
PERTURB_DIR=../data/perturb python eval_offset_perturbation.py
```

Applies an additive perturbation to the torso angular offset at inference only, with the
trained model fixed. `PERTURB_DIR` points at the perturbed `stride_modeling` `.mat` files
written by `perturb_torso_offset.m`; it defaults to `../data/perturb`.
Results: `results/scadd_all.csv`.

### 4.4 Sacrum IMU baseline (Section III-B)

```bash
python sweep.py --config configs/config_sacrum_baseline.yaml
```

Parameter-matched to BINN (111,788), same folds, seed and protocol, using the sacrum IMU
signals recorded simultaneously from the same subjects. 5 data fractions x 7 training
configurations = 35 runs. Results: `results/chB_all.csv`.

---

## 5. Result files

| File | Contents |
|---|---|
| `noise_final_all.csv` | CoM corruption, all conditions and corruption levels |
| `ch_all.csv`, `test_all.csv` | CoM channel-removal retraining |
| `chB_all.csv` | Sacrum IMU baseline |
| `scadd_all.csv` | Torso offset perturbation (additive, +-15 and +-30 deg) |
| `ablation_models_bych.csv` | Per-channel NRMSE and r for all models, ID conditions at 100 % |
| `supp_dose_response.csv` | Corruption dose-response, all 60 baseline-to-corruption comparisons |
| `supp_corruption_example_deg.csv` | NRMSE increase per corruption level, AP-CoM corrupted |

In `ablation_models_bych.csv` the `model` column uses the config names:
`proposed`, `wo_biomech_arch`, `wo_cross_attention`, `wo_com_to_grf_mapper`, `cnn`,
`transformer`.

---

## 6. Metrics and statistics

NRMSE is the stride-level RMSE normalised by the peak-to-peak range of the measured GRF
within that stride, expressed as a percentage. Pearson `r` is computed per stride. Both
metrics are averaged per subject before any statistical test; the subject is the only
independent sampling unit.

All statistical tests are paired t-tests at the subject level with Benjamini-Hochberg
FDR correction. Four families are corrected independently: the 128 primary
GRF-performance comparisons reported in Tables IV-V and Figs. 7-9, the 60
baseline-to-corruption comparisons, the 48 adjacent-corruption-level comparisons, and
the eight channel-removal comparisons.

Per-condition performance is reported for all 19 subjects. Analyses that require data at
all three walking speeds - the subject-level paired comparisons and the subject-specific
speed-peak regressions - use the 17 subjects with complete data. Comparisons that
contrast two conditions within the same subject, such as the corruption, channel-removal
and offset-perturbation analyses, do not require all three speeds and use all 19.

---

## 7. Environment

```bash
conda env create -f environment.yml
conda activate binn-grf
```

MATLAB R2021b or later is required for the `.m` scripts (`tiledlayout`, `readtable`,
string arrays).

---

## 8. Data

The datasets generated and analyzed during the current study are not publicly available
due to participant privacy and institutional review board (IRB) restrictions, but are
available from the corresponding author on reasonable request.
