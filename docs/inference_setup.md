# Project Prometheus — Inference Setup

## Objective

Run an official pretrained Vesuvius Challenge ink-detection model on a real Herculaneum papyrus surface using NVIDIA GPU resources managed by CIQ Fuzzball.

## Prerequisites

- [ ] GPU job executes successfully in Fuzzball
- [ ] NVIDIA GPU model and memory identified
- [ ] CPU architecture identified (`x86_64` or `aarch64`)
- [X] Compatible Python and PyTorch environment confirmed
- [X] Official Vesuvius repository available
- [ ] Pretrained model checkpoint accessible
- [ ] Compatible papyrus surface-volume dataset accessible
- [ ] Persistent output storage available

## First inference experiment

**Experiment ID:** `baseline`

**Input:** One compatible papyrus surface volume

**Model:** Official pretrained ink-detection checkpoint

**Output:** `predictions/depth_default.tif`

**Measurements:** Inference runtime, GPU memory, GPU utilization, output image dimensions

## Success criteria

The inference job completes without errors and produces a valid two-dimensional ink-probability image that our existing `compare_predictions.py` can eventually read.

## Experimental controls

Keep the model checkpoint, input region, preprocessing, output alignment, and inference settings consistent when comparing depth configurations.

## Current blockers

Record Fuzzball access issues, unavailable datasets, incompatible dependencies, and model-loading errors here.
