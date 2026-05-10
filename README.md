# MSA feature extraction prototype

Prototype for checking whether a downstream music structure analysis path is
feasible with:

- STM rhythm descriptors.
- MFCC timbre/texture descriptors.
- Chroma and CENS harmonic descriptors.
- Predominant melody F0 using Essentia/MELODIA.
- Beat-synchronous summaries.
- Self-similarity matrices per descriptor and a simple fused SSM.

## Environment

The environment used in this workspace is:

```bash
conda activate codex_msa_features
```

To recreate it from scratch:

```bash
conda env create -f environment.yml
conda activate codex_msa_features
```

## Smoke test

The script can generate a synthetic A/B/A audio file and process it:

```bash
python extract_msa_features.py --demo --out-dir feature_outputs
```

This writes:

- `feature_outputs/demo_aba_song.wav`
- `feature_outputs/demo_aba_song_features.npz`
- `feature_outputs/demo_aba_song_summary.json`
- `feature_outputs/demo_aba_song_preview.png`

## Real audio

```bash
python extract_msa_features.py path/to/song.wav --out-dir feature_outputs
```

The `.npz` contains raw frame-level features, beat-synchronous features and
SSMs. The preview image is meant only as a quick sanity check before building a
proper segmentation/evaluation pipeline.

## Downstream structure baseline

Run the current unsupervised downstream benchmark on the 20-track RWC-P pilot:

```bash
/Users/pcancela/miniforge3/envs/emilio_msa_features/bin/python run_structure_baseline.py \
  --features-dir feature_outputs/rwc_p_20 \
  --annotation-dir data/rwc-annotations-archive \
  --out-dir structure_outputs/rwc_p_20_iter2 \
  --limit 20 \
  --plot-experiment E11_anchor_mfcc_multiscale \
  --ranked-track-figures 3
```

Useful scale-up flags:

- `--feature-glob "*_features.npz"` selects feature files.
- `--track-regex "RWC_P0(1|2)"` filters tracks by filename.
- `--experiments E1_mfcc,E11_anchor_mfcc_multiscale` runs a subset.
- `--experiments E17_learned_ssm_loo` runs a supervised leave-one-out SSM
  fusion baseline.
- `--experiments E18_product_mfcc_cens,E19_product_stm_mfcc_cens` runs
  non-linear geometric SSM fusion checks.
- `--experiments E20_poly_mfcc_cens,E21_poly_stm_mfcc_cens` runs polynomial
  SSM fusion checks with individual terms and pairwise products.
- `--learned-weight-step 0.1` controls the supervised SSM weight grid.
- `--ranked-track-figures 3` writes plots only for the top/bottom tracks.
- `--skip-dashboard` avoids global plots for quick smoke tests.

Main outputs:

- `experiment_metrics.csv`: one row per track and experiment.
- `aggregate_metrics.csv`: average metrics by experiment.
- `best_experiment_by_track.csv`: strongest experiment for each track.
- `learned_ssm_loo_weights.csv`: fold-specific weights for supervised SSM
  fusion when `E17_learned_ssm_loo` is run.
- `run_manifest.json`: reproducibility metadata for the run.
- `figures/`: dashboard and selected infographics.

Example supervised SSM run:

```bash
/Users/pcancela/miniforge3/envs/emilio_msa_features/bin/python run_structure_baseline.py \
  --features-dir feature_outputs/rwc_p_20 \
  --annotation-dir data/rwc-annotations-archive \
  --out-dir structure_outputs/rwc_p_20_supervised_ssm \
  --limit 20 \
  --experiments E1_mfcc,E3_content,E6_stm_content,E8_adaptive_stm_content,E10_mfcc_multiscale,E11_anchor_mfcc_multiscale,E13_anchor_dense,E17_learned_ssm_loo \
  --plot-experiment E17_learned_ssm_loo \
  --ranked-track-figures 2
```
