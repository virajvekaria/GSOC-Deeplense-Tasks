# DeepLense GSoC Evaluation - Solution Repository

This repository contains my solutions for the **DeepLense GSoC evaluation tasks**, including the **Common Test (I)** and the selected **Specific Tests II, V, VI, VII, and VIII**.

Most tasks are implemented in **PyTorch** and documented through task-specific Jupyter notebooks. For convenience, each task folder also includes its own local `README.md` with setup notes, approach, and result snapshots.

---

## Contents

| Task | Folder | Main File | Title | Model / Method |
|------|--------|-----------|-------|----------------|
| I | `Common task/` | `task_common.ipynb` | Multi-Class Classification | ResNet-18 |
| II | `Test 2/` | external fork + `deeplense_agent` package | Agentic AI | Pydantic AI + DeepLenseSim tools |
| V | `Test 5/` | `task_5.ipynb` | Lens Finding & Data Pipelines | Custom CNN + Spatial Attention |
| VI.A | `Test 6/` | `task_6a.ipynb` | Super-Resolution (Synthetic) | RCAN |
| VI.B | `Test 6/` | `task_6b.ipynb` | Super-Resolution (Real Pairs) | Augmented / Fine-tuned RCAN |
| VII | `Test 7/` | `task_7.ipynb` | Physics-Guided Classification | PhysicsInformedFusionNet + EfficientNet-B0 |
| VIII | `Test 8/` | `ldm.ipynb` | Diffusion Models | VAE + Latent Diffusion |

---

## Repository Layout

- [Common task](Common%20task): Common Test I deliverables
- [Test 2](Test%202): Test II summary plus copied sample run artifacts
- [Test 5](Test%205): Test V deliverables
- [Test 6](Test%206): Test VI.A and VI.B deliverables
- [Test 7](Test%207): Test VII deliverables
- [Test 8](Test%208): Test VIII deliverables

---

## Task I - Multi-Class Classification

- **Objective:** Classify lensing images into three categories: `no_sub`, `sphere/subhalo`, and `vortex`.
- **Architecture:** Pretrained `ResNet-18` fine-tuned for three classes.
- **Pipeline:** `.npy` image loading, resize to `224 x 224`, data augmentation, classification head replacement, ROC analysis.
- **Reported Result:** `0.95` test accuracy with `0.95` macro precision / recall / F1.
- **Artifacts:** notebook, best checkpoint, and saved result image in [Common task](Common%20task).

---

## Task II - Agentic AI

- **Objective:** Build a natural-language interface around DeepLenseSim for validated, human-in-the-loop strong lensing simulation.
- **Implementation:** A concrete `deeplense_agent` package built with **Pydantic AI** and preserved in the linked fork:
  <https://github.com/virajvekaria/DeepLenseSim>
- **Workflow:** prompt parsing, typed request validation, capability lookup, preview-before-run, explicit confirmation, artifact generation.
- **Supported Configurations:** `Model_I`, `Model_II`, `Model_III` with `no_sub`, `cdm`, and `axion` substructure.
- **Artifacts:** sample run, metadata, preview images, and contact sheet in [Test 2](Test%202).

---

## Task V - Lens Finding & Data Pipelines

- **Objective:** Binary classification of lens versus non-lens observations under heavy class imbalance.
- **Architecture:** Custom CNN with a `SpatialAttention` block.
- **Training Strategy:** focal loss, hard-negative mining, and a second-stage fine-tuning "bootcamp".
- **Inference Strategy:** test-time augmentation over flips and rotation.
- **Reported Result:** best reported test AUC `0.9650` after fine-tuning and TTA.

---

## Task VI - Image Super-Resolution

### Task VI.A - Synthetic Dataset

- **Objective:** Upscale simulated low-resolution lensing images using paired high-resolution targets.
- **Model Used:** `RCAN` (Residual Channel Attention Network).
- **Evaluation:** MSE, PSNR, and SSIM.
- **Reported Result:** `PSNR 42.32 dB`, `SSIM 0.9745`.

### Task VI.B - Real HR/LR Pairs

- **Objective:** Super-resolve a much smaller real paired dataset.
- **Method:** reuse the same RCAN family with extra augmentation and limited-data adaptation.
- **Techniques Used:** flips, rotations, LR-noise augmentation, and fine-tuning on expanded pairs.
- **Reported Result:** `PSNR 29.95 dB`, `SSIM 0.8217`.

---

## Task VII - Physics-Guided ML

- **Objective:** Improve three-class classification with a physics-informed architecture.
- **Integration:** a physics-informed inverse-lensing encoder is fused with an `EfficientNet-B0` classifier.
- **Transformer Component:** includes local self-attention and a physics-guided reconstruction branch before fusion.
- **Reported Result:** final mean test AUC `0.9933`.
---

## Task VIII - Diffusion Models

- **Objective:** Generate realistic strong gravitational lensing images.
- **Approach:** first train a grayscale VAE, then train a latent diffusion model over the learned latent space.
- **Sampling:** EMA checkpoint plus DDIM-style fast sampling.
- **Evaluation:** qualitative generations, FID, and SSIM.
- **Reported Result:** `FID 5.8864` on `5000` generated samples and `SSIM 0.8699`.

---

## Evaluation Summary

| Task | Metrics | Best Reported Values |
|------|---------|----------------------|
| I | Accuracy / Macro F1 | `0.95 / 0.95` |
| II | Structured run output | sample run + metadata produced |
| V | AUC | `0.9650` |
| VI.A | PSNR / SSIM | `42.32 / 0.9745` |
| VI.B | PSNR / SSIM | `29.95 / 0.8217` |
| VII | Mean AUC | `0.9933` |
| VIII | FID | `5.8864` |

---

## Running The Work

- Most task folders are notebook-driven and can be run independently.
- Dataset paths may need to be updated to match your local machine.
- Main Python dependencies used across tasks include `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `torchmetrics`, `efficientnet_pytorch`, etc.
- Test II additionally uses `pydantic`, `pydantic-ai`, and model-provider integrations for Gemini / Ollama.

---

## Notes

- Some large checkpoints could not be pushed directly in GitHub if they exceeded size limits.
- Task-specific details, result previews, and any extra download links are documented inside each task folder's README.
