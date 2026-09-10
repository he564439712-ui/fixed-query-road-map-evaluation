# Fixed-query route outcomes beyond aggregate road-map scores

This repository contains the code and frozen numerical evidence for the study
**“Fixed-query route outcomes beyond aggregate road-map scores: a controlled
evaluation of remote-sensing road products.”** It evaluates what prespecified
start–goal queries add beyond aggregate road-product metrics when the road
product, query set, and planning protocol are held fixed.

Version `v1.0.0` is the archival release prepared for the associated manuscript.
The version-specific Zenodo DOI will be added to the GitHub release description
after Zenodo finishes archiving the tag.

## Release contents

```text
configs/                 Frozen experiment configuration
src/                     Data, model, evaluation, and planning modules
scripts/                 Training, inference, evaluation, audit, and reporting CLIs
tests/                   Unit and regression tests
artifacts/raw_metrics/   Frozen numerical outputs used by the manuscript
metadata.csv             Massachusetts image manifest
label_class_dict.csv     Dataset label mapping
environment.yml          Recorded conda environment
requirements.txt         Portable Python dependency list
```

The archive deliberately excludes source imagery, masks, checkpoints,
prediction rasters, local logs, manuscript files, and copied third-party source.
These exclusions keep the release small, avoid redistributing datasets, and
remove machine-specific paths and contact details that are unnecessary for
software reuse.

## Installation

Create the recorded environment:

```bash
conda env create -f environment.yml
conda activate road-risk-planning
```

Install a PyTorch and torchvision build appropriate for the available CPU or
GPU by following the official PyTorch installation instructions. For a lighter
environment used only to inspect frozen outputs and run non-model utilities:

```bash
python -m pip install -r requirements.txt
```

## Validation

Run the source-level test suite from the repository root:

```bash
python -m pytest tests -q
```

Verify the archived APLS identity checks and regenerate the active manuscript
tables from the frozen metrics:

```bash
python scripts/audit_apls_identity.py
python scripts/make_paper_tables.py
```

`make_paper_tables.py` creates `paper/tables/` locally. The numerical panels of
the paper figures read the same frozen metrics. Image-based panels also require
the original datasets and prediction rasters, which are not redistributed.

## Full experiment inputs

The Massachusetts Roads data are available from the [original dataset
page](https://www.cs.toronto.edu/~vmnih/data/). DeepGlobe Road Extraction data
must be obtained from an authorized distributor. Place inputs according to
`configs/data.yaml` and the conventions in `src/data/deepglobe.py`.

The SpaceNet APLS audit uses the public SpaceNet reference implementation. Its
source is not copied into this archive; the expected local path and the audit
scope are documented in `scripts/eval_massachusetts_apls_reference.py` and
`scripts/eval_deepglobe_apls_reference.py`.

## Reproduction order

The principal workflow is:

1. Train and predict with `train.py`, `predict_ensemble.py`, and the corresponding
   DeepGlobe scripts.
2. Calibrate probabilities with `calibrate.py`.
3. generate the fixed query manifests with `generate_queries.py` or
   `generate_deepglobe_queries.py`.
4. Evaluate segmentation, topology, APLS, and route outcomes with the `evaluate_*`
   and `eval_*` scripts.
5. Run the image-level summaries, factorial analysis, search-cap sensitivity,
   bridge audit, and negative metric-rank audit with the corresponding analysis
   scripts.

Random seeds, thresholds, split rules, and query counts are fixed in the
configuration files and command-line defaults. The archived CSV and JSON files
permit inspection of reported values without downloading large raster products.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff). For reproducible scholarly
citation, cite the Zenodo **version DOI for `v1.0.0`** once it appears in the
GitHub release description. The Zenodo concept DOI identifies the evolving
software project and may resolve to a later version.

## Licensing and third-party material

Original software in this repository is released under the MIT License. Frozen
numerical results in `artifacts/raw_metrics/` are released under CC BY 4.0; see
[`DATA_LICENSE.md`](DATA_LICENSE.md). Dataset imagery, labels, model weights, and
third-party software are not part of this release. See
[`THIRD_PARTY.md`](THIRD_PARTY.md) for provenance and upstream terms.
