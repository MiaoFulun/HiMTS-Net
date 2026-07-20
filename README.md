# HiMTS-Net

Official implementation of **HiMTS-Net: A Native-Axis-Preserving Hierarchical
Multi-Domain Time--Spectral Network for Small Target Detection in Sea
Clutter**.

## Overview

HiMTS-Net preserves the native temporal axis while learning hierarchical
representations from complementary time--spectral inputs. Sample-adaptive
gating combines the four branches for small target detection in sea clutter.

This repository provides the model architecture, the four input
representations, data preparation for IPIX and SDRDSP2022, and the experiment
pipeline described in the paper. Raw datasets, pretrained weights, comparison
baselines, and experimental result files are not redistributed.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data layout

Download the datasets from their official sources and arrange them as follows:

```text
data/
|-- ipix/
|   |-- 19931118_023604_starea280.cdf
|   `-- ...
`-- sdrdsp2022/
    |-- 20221114140049_stare_HH.mat
    `-- ...
```

The repository does not redistribute either dataset.

## Usage

IPIX:

```bash
python train.py --config configs/ipix.yaml
```

SDRDSP2022:

```bash
python train.py --config configs/sdrdsp2022.yaml
```

Each run writes a locally trained checkpoint, normalization parameters, and
training history under the configured output directory. These generated files
are ignored by Git and are not part of the public source release.

## Method configuration

The supplied configurations use the settings stated in the paper:

- hidden dimension: 32
- encoder kernels: 15, 21, and 31
- two residual blocks per encoder stage
- AdamW learning rate: 1e-3
- batch size: 256
- training epochs: 30
- mini-batch hard-negative fraction: 0.25
- validation model-selection operating point: Pfa = 1e-3
- random seed: 42

## Citation

Please cite the accompanying paper if this code is useful in your research.

## License

This project is released under the MIT License.
