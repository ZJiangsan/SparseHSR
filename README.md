# SparseHSR

**SparseHSR: From RGB Guidance to Sparse Visible–NIR Sensing for Full-Range Hyperspectral Super-Resolution**

Research code and experiment records for the SparseHSR study by Jiangsan Zhao, Jakob Geipel and Krzysztof Kusnierek.

## Scientific question
Given a spectrally complete low-resolution hyperspectral image (LR-HSI), how do the number and wavelengths of high-resolution visible/NIR guidance bands affect full-range hyperspectral super-resolution? The work compares an abundance-Gram reconstruction procedure with HySure-SVD over 16 predefined guidance configurations on Chikusei and ICVL and a focused airborne HySpex Mjolnir evaluation.

**Scope:** A controlled empirical study of measurement configurations, *not* a new reconstruction algorithm, automatic sensor co-optimization, or validation of an independently acquired dual-resolution sensor pair.

## Contents
- `scripts/hysure_full_16_configurations.py`: full 16-configuration HySure baseline, with preprocessing, data-fidelity solver, convergence diagnostics and results export.
- `scripts/gram_multiseed_robustness_seeds1_2.py`: abundance-Gram robustness runner for five selected configurations across seeds 1 and 2, including the full model and loss implementation.
- `results/10_manuscript_baseline_table.csv` and `results/11_RGB_comparison.csv`: existing numerical summaries (not newly computed).
- `docs/REPRODUCIBILITY.md`: experiment conventions, data requirements, limitations and instructions.

## Reproduction
Install dependencies from `requirements.txt`; obtain the source datasets separately and set input paths/checkpoints in the scripts. See [reproducibility instructions](docs/REPRODUCIBILITY.md).

**Release limitation:** The original seed-0 abundance-Gram master script, frozen encoder/decoder checkpoints, large source data and full experiment logs are not included in this initial repository snapshot. Consequently this code archive is not yet a complete one-command recreation of every number in the paper. The robustness script refers to the existing seed-0 results directory for the combined three-seed analysis. Do not mistake a fresh seed-1/2 run for the original seed-0 sweep.

## Data and code attribution
Chikusei and ICVL are third-party research datasets; follow their distributors' licenses. The HySure GPU baseline follows the numerical formulation from Simões et al. (IEEE TGRS 2015); the original HySure implementation is at https://github.com/alfaiate/HySure. The abundance-Gram formulation is inherited from the authors' previous work.

This repository includes source scripts and compact analysis summaries only; it does not redistribute large datasets, pretrained checkpoints, manuscript review correspondence or institution-internal files.

## Citation
Please cite the SparseHSR article once the final publication metadata is available.
