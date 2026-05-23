# MSA Feature Extraction Prototype

Prototype for extracting interpretable and SSL-based features for music
structure analysis, then converting them into beat-synchronous descriptors and
self-similarity matrices.

## Environment

The environment used in this workspace is:

```bash
conda activate msa_features
```

To recreate it:

```bash
conda env create -f environment.yml
conda activate msa_features
```

## Modules

- `data_loading.py`: loads YAML config, audio files, section annotations, and demo audio.
- `beat_tracking.py`: estimates beats/downbeats and regularizes beat boundaries.
- `feature_types.py`: computes descriptor streams and SSM-ready feature matrices, including SSL embeddings.
  - _IMPORTANT NOTE:_ currently this code takes all the SSL models within the same folder as a "grandparent" directory. For example, if running the script: 
  ```python 
  python extract_msa_features.py
  ``` 
  make sure to have the SSL models repos ([MusicFM](https://github.com/minzwon/musicfm), [MuQ](https://github.com/tencent-ailab/MuQ/), [MATPAC](https://github.com/aurianworld/matpac/), or any other you want) at ``bash MSA-with-Interpretable-Features/../../SSL_models``
- `pipeline.py`: orchestrates extraction, beat-synchronous aggregation, preview rendering, and output writing.

## Config File

The extraction scripts use `msa_feature_config.template.yaml`.

Important fields:

- `audio_path`: single-file mode.
- `audio_dir`: batch mode.
- `out_dir`: output directory for `.npz`, preview images, and summaries.
- `feature_config.computing_features`: only these features are computed.
- `feature_config.preview_features`: only these computed features are previewed.
- `feature_config.beat_tracking_method`: `librosa`, `beat_this`, or `madmom`.

Example feature section:

```yaml
feature_config:
  computing_features:
    - stm
    - mfcc
    - chroma
    - matpac
    - muq
    - musicfm
  preview_features:
    - stm
    - mfcc
    - chroma
    - matpac
```

## Usage

Single-file extraction with config:

```bash
python extract_msa_features.py --config msa_feature_config.template.yaml
```

Single-file extraction with an explicit audio path:

```bash
python extract_msa_features.py path/to/song.wav --config msa_feature_config.template.yaml
```

Batch extraction with config:

```bash
python batch_extract_msa_features.py --config msa_feature_config.template.yaml
```

The internal orchestration happens through `pipeline.py`, while
`extract_msa_features.py` and `batch_extract_msa_features.py` are the public
CLI entrypoints.

## Smoke Test

Generate a synthetic A/B/A example:

```bash
python extract_msa_features.py --demo --out-dir feature_outputs
```

This writes files such as:

- `feature_outputs/demo_aba_song.wav`
- `feature_outputs/demo_aba_song_features.npz`
- `feature_outputs/demo_aba_song_summary.json`
- `feature_outputs/demo_aba_song_preview.png`

## Outputs

Each run can produce:

- raw feature arrays for the requested features
- beat-synchronous feature arrays
- SSMs for the requested computed features
- preview images for the requested preview features
- a JSON summary with shapes, diagnostics, and output paths

## Downstream Structure Baseline

Run the current unsupervised downstream benchmark on extracted features:

```bash
/Users/pcancela/miniforge3/envs/emilio_msa_features/bin/python run_structure_baseline.py \
  --features-dir feature_outputs/rwc_p_20 \
  --annotation-dir data/rwc-annotations-archive \
  --out-dir structure_outputs/rwc_p_20_iter2 \
  --limit 20 \
  --plot-experiment E11_anchor_mfcc_multiscale \
  --ranked-track-figures 3
```
