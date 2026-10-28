# Reproducing SparseHSR experiments

## Source files

- `scripts/gram_multiseed_robustness_seeds1_2.py` is the **actual submitted seed-1/seed-2 robustness runner** derived from the established master code. It includes model class definitions, the preprocessing protocol, fixed RGB response, 16-configuration manifest, loss components, checkpoint selection and five-condition robustness workflow. Its training flag is enabled by default and expects GPU-accessible input files.
- `scripts/hysure_full_16_configurations.py` is the **full 16-configuration HySure baseline**, including the predefined wavelength configurations, exact spatial degradation, observed-only RGB response fitting, SALSA/ADMM optimization, convergence checks and evaluation/export procedures.
- `results/*.csv` are existing output tables from completed runs, not new analyses generated for this repository upload.

## Datasets and preparation

**Chikusei**: 1000 × 1000 × 40 full-range representation; 31-band visible product for synthetic RGB. The 40-band source wavelengths are embedded in both scripts, although the preparation function preferentially loads `selected_wavelengths_nm` from the source HDF5 when present.

**ICVL**: 1083 × 1392 × 40 full-range source and 1083 × 1392 × 31 visible product, both transformed identically: rotate 90°, horizontal mirror, then `[392:1192, 100:900]` crop. The exact ICVL 40-band wavelength vector is read from the source MAT file. The actual source `ICVL_40band.mat` supplied during the revision has values below.

The two full/visible cubes for each scene are normalized by a **single shared maximum**: `max(full_40.max(), visible_31.max())`. The fixed 3×31 response is stored as `RGB_SRF_BGR` in each script, in B/G/R order.

Generate the LR-HSI using an 8×8 Gaussian spatial kernel (σ=0.5), valid convolution, and 8× spatial subsampling. Narrowbands use nearest-source-wavelength selection. All target wavelength labels in paper tables are approximate nominal anchors.

### ICVL 40-band wavelength values (nm)
```
400.00000, 415.38461, 430.76923, 446.15384, 460.99000,
476.92307, 492.30768, 507.69232, 523.07690, 538.46155,
548.67000, 569.23077, 584.61536, 600.00000, 615.38464,
630.76923, 651.82000, 661.53845, 676.92310, 692.30770,
707.69230, 723.07690, 738.46155, 753.84610, 769.23080,
784.61536, 801.39000, 815.38464, 830.76920, 846.15387,
861.53845, 876.92310, 892.30770, 907.69230, 923.07690,
938.46155, 953.84610, 969.23080, 984.61536, 1000.00000
```

Note: the **nominal 714-nm ICVL guide anchor selects 707.6923 nm**, using nearest-band mapping.

## Abundance-Gram architecture and optimization

The frozen encoder consists of bias-free linear layers 40→5, 45→5, and 50→40 with concatenation and ReLU. A 40→40 bias-free linear decoder follows. The sparse-guide mapper is a residual bias-free linear map to 40 bands with repeated input channels. The composite objective uses 0.1× reconstruction L₂,₁ + 10× diagonal loss + 100× centered Gram loss (the exact internal terms/weights are in `compute_training_loss`). Adam uses LR=1e−4 and weight decay=1e−4. Checkpoint selection is by minimum unsupervised composite objective. The targeted seed study uses seeds 1 and 2 on five specified configurations and combines those with original seed-0 outcomes.

## HySure baseline

The accompanying GPU port uses SVD subspace dimension p=10, λφ=5e−4, λm=1, ADMM μ=0.05 and float64 computations; max 20,000 iterations and a repeated optimization-only convergence test. Both methods take the same LR-HSI and HR guide for each configuration, but the methods have distinct objectives and reconstruction conventions.

## Running locally

1. Install requirements: `pip install -r requirements.txt`.
2. Obtain the Chikusei and ICVL data from their original distributors, and prepare the full-range and visible cubes in the expected script formats.
3. Update the paths under **USER SETTINGS** at the beginning of each script (the original machine-relative paths are preserved for provenance).
4. For Gram robustness, provide the separately trained dataset-specific frozen encoder/decoder `.pth` checkpoints and the original seed-0 result directory before computing three-seed summaries.
5. For HySure, run the baseline on a sufficiently provisioned CUDA GPU; output files are created under its named output directory.

## Known limits of this initial archive

The original seed-0 master runner, checkpoint binaries, source MAT/HDF5 cubes, Mjolnir-specific field runner, and complete historical logs are not in this snapshot. Reproducing the published quantitative tables from scratch therefore requires those additional artifacts. Do not claim that the included robustness runner regenerates all 16 seed-0 results.

The benchmark data are not redistributed here. Please use the datasets' original distribution terms. No automated sensor optimization or independent dual-sensor spatial-resolution validation was performed as part of these baseline scripts.

## Metric notes
The supplied scripts contain their **original evaluation implementations**. In particular, the ERGAS-like number printed by their evaluation functions uses the code's existing scale convention. If comparing against a convention with an explicit 1/r factor (r=8), the reporting definition must be stated and applied consistently. Do not silently replace source CSV values with differently normalized quantities.
