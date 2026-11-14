# ciq-hackathon-team5

# Project Prometheus
**Accelerating Ancient Scroll Discovery Through Distributed, Uncertainty-Aware AI**

### The Problem: Recovering Lost History at Scale

Nearly 2,000 years after Mount Vesuvius buried the ancient city of Herculaneum, hundreds of carbonized papyrus scrolls remain largely unreadable. Physically opening them risks destroying their contents, so researchers rely on high-resolution CT scans and artificial intelligence to detect hidden ink.

However, ink-detection models are computationally expensive to run, and their predictions can vary depending on model checkpoints, scan depths, and classification thresholds. Researchers need scalable infrastructure to process these scans and reproducible methods to understand where AI predictions are reliable or uncertain.

### Our Solution

**Project Prometheus** combines CIQ Fuzzball, distributed GPU inference, and uncertainty-aware analysis to make ancient scroll research more scalable, reproducible, and computationally targeted.

Using pretrained Vesuvius Challenge ink-detection models, we orchestrated inference across two NVIDIA GB10 GPU nodes with VAST shared storage. We systematically varied model checkpoints and CT depth windows, measured prediction disagreement, and evaluated an adaptive strategy that prioritizes additional analysis in uncertain regions.

### Measurable Results

| Metric | Result |
|---|---|
| Distributed inference workloads | 18 GPU jobs across 2 nodes |
| Estimated parallel speedup | **1.61×** versus summed sequential task runtimes |
| Prediction pixels analyzed | **86.8 million** across 2 scroll sections |
| Checkpoint-induced disagreement | **32.00% / 29.77%** |
| Depth-induced disagreement | **14.65% / 12.36%** |
| Adaptive improvement over random selection | **16.5% / 20.5%** lower reference disagreement at the six-pass budget |
| Independent reproducibility | Local analysis reproduced distributed experiment metrics |

### Key Scientific Findings

**1. Model checkpoint selection matters more than scan depth.** Changing the model checkpoint produced approximately 2.2–2.4 times as much binary prediction disagreement as changing the CT depth window in our controlled comparisons.

**2. Uncertainty can guide computational priorities.** Our adaptive-selection experiment concentrated additional analysis on regions with conflicting predictions. At the six-pass compute budget, it covered approximately 73–75% of contested pixels while refining about 57% of usable tile area.

**3. Scientific results can be independently reproduced.** We validated the distributed inference outputs through a separate local analysis pipeline, confirming the reported disagreement and uncertainty metrics.

### Why CIQ Fuzzball Matters

CIQ Fuzzball enabled us to coordinate multiple AI inference configurations across GPU resources, transforming a computationally intensive research task into a repeatable, distributed experimental workflow.

Our experiments demonstrate how orchestration infrastructure can support not only faster scientific computation, but also systematic experimentation, uncertainty analysis, and reproducibility.

### Impact and Future Potential

Project Prometheus does not claim to have deciphered ancient text or improved ground-truth ink-detection accuracy. Our adaptive results are based on offline comparisons with a full-model reference, and real-world GPU savings remain to be validated.

Instead, we demonstrate a promising foundation for **intelligent scientific compute allocation**: identify where AI predictions are unstable, prioritize those regions for further analysis, and make more effective use of limited computational resources.

**Our vision:** Help researchers analyze more scroll material, investigate uncertain regions more efficiently, and accelerate the effort to recover lost knowledge from antiquity.