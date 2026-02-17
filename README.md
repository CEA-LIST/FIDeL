# Failure Identification in Imitation Learning via Statistical and Semantic Filtering (FIDeL)

This repository hosts the **project webpage** and supplementary materials for the paper:

**Failure Identification in Imitation Learning via Statistical and Semantic Filtering**  
Quentin Rolland, Fabrice Mayran de Chamisso, Jean-Baptiste Mouret  
*IEEE International Conference on Robotics and Automation (ICRA), 2026*

link to the site : **https://cea-list.github.io/FIDeL/**
---

## Overview

Imitation Learning (IL) policies are brittle to rare or out-of-distribution events in real-world robotic deployments.  
We introduce **FIDeL**, a policy-agnostic failure identification framework that combines:

- Vision-based anomaly detection
- Optimal transport alignment with expert demonstrations
- Spatio-temporal thresholding via conformal prediction
- Semantic filtering using Vision-Language Models (VLMs)

FIDeL detects, localizes, and semantically filters failures in real time, without interfering with policy execution.

---

## BotFails Dataset

We also introduce **BotFails**, a multimodal dataset for robotic failure detection:

- Vision, proprioception, and language instructions
- 646 video sequences
- 414,359 annotated frames
- Real-world manipulation and interaction tasks
- Explicit failure and benign anomaly annotations

---

## Results

FIDeL outperforms state-of-the-art anomaly detection baselines on BotFails, achieving:

- **+5.30% AUROC** in anomaly detection
- **+17.38% accuracy** in failure identification

Qualitative results and videos are available on the project webpage.

---



## Acknowledgments
Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
