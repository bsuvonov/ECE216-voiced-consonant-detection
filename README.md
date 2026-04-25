# ECE216 Voiced Consonant Detection

This repository contains the report and experiment code for position-independent detection of voiced consonants in isolated words. The study focuses on the consonants `d`, `g`, `n`, and `r`, using real recordings from Speech Commands v0.02.

## Contents

- `report/main.pdf`: final compiled report.
- `report/main.tex`: LaTeX source for the report.
- `matlab/run_fullword_topic2.m`: MATLAB FFT/STFT spectrum-based detectors.
- `python/run_fullword_experiments.py`: full experiment pipeline, including MFCC/DTW and AI MLP comparisons.
- `python/run_experiments.py`: wrapper for the full Python experiment pipeline.
- `results/summary.txt`: Python experiment summary.
- `results/matlab_fullword_classical_metrics.txt`: MATLAB spectrum-detector summary.
- `results/matlab_fullword_classical_model.mat`: exported MATLAB classical detector templates and parameters.

## Dataset

The dataset is not committed because it is several GB. Download Speech Commands v0.02 and extract it under `data/speech_commands_v0.02`:

```bash
mkdir -p data
wget -O data/speech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/speech_commands_v0.02
tar -xzf data/speech_commands_v0.02.tar.gz -C data/speech_commands_v0.02
```

## Run Experiments

Python:

```bash
python3 python/run_experiments.py
```

MATLAB:

```matlab
cd matlab
run_fullword_topic2
```

The report can be rebuilt from `report/main.tex` with a standard LaTeX distribution.
