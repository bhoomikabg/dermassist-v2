# DermAssist

Image-only skin-lesion classifier (ResNet18, 7 HAM10000 classes) with post-hoc calibration, Grad-CAM explanations and a
rule-based decision layer that turns calibrated probabilities into a clinical action.

```
image ──► predict_single_image.py ──► JSON (calibrated probabilities + Grad-CAM) ──► decision_layer.py ──► action + explanation
```

## Repository layout

```
DermAssist/
├── README.md  requirements.txt  .gitignore
├── .gitattributes              # Git LFS rules for *.pt and *.npz
├── data/                       # splits.csv (committed); train_extended.csv and images are not committed
├── models/                     # P2b_seed0_best.pt, ENS_logits.npz, temperature_calibration.pt (Git LFS)
├── outputs/                    # Grad-CAM images, calibration plots, stability results (not committed)
├── testdata/                   # your own test images (optional)
└── src/
    ├── data/prepare_data.py                      # lesion-grouped split (+ ISIC 2019 extension of train)
    ├── training/train_resnet18.py                # trains the model, writes checkpoint + ENS_logits.npz
    ├── calibration/
    │   ├── calibrate_temperature.py              # fits T (with melanoma bias) -> temperature_calibration.pt
    │   ├── compare_calibration_methods.py        # temperature / vector / Dirichlet / isotonic comparison
    │   ├── logit_adjustment_analysis.py          # HAM-prior correction, tau logit adjustment, thresholds
    │   └── melanoma_bias_sweep.py                # recall vs accuracy for the melanoma logit bias
    ├── explainability/
    │   ├── gradcam_utils.py                      # Grad-CAM on layer4, heatmap + overlay + lesion focus score
    │   └── stability_analysis.py                 # Grad-CAM SSIM under noise / brightness / JPEG
    ├── inference/
    │   ├── dermassist_model.py                   # model loading, preprocessing, TTA, calibration
    │   ├── predict_single_image.py               # one image -> prediction JSON (+ Grad-CAM)
    │   └── verify_pipeline.py                    # checks the pipeline reproduces the training logits
    └── decision/decision_layer.py                # probabilities -> risk level + action (reads JSON on stdin)
```

## 1. Setup

Python 3.10 to 3.13 is recommended. Very new versions (3.15) may lack wheels for some packages.

**Windows (PowerShell):**
```powershell
git lfs install
git clone <your-repo-url> DermAssist
cd DermAssist
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
git lfs install
git clone <your-repo-url> DermAssist && cd DermAssist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-image` is optional (only for the stability check: `--stability-check` and `stability_analysis.py`): `pip install scikit-image`.
It needs a Python version it ships wheels for (3.12 / 3.13).

## 2. Model files (Git LFS)

These are stored with Git LFS, so they arrive with the clone:

| File | Location | Needed for |
|---|---|---|
| `P2b_seed0_best.pt` | `models/` | running the model, verify, stability |
| `ENS_logits.npz` | `models/` | calibration scripts, verify |
| `temperature_calibration.pt` | `models/` | running the model (regenerate with `calibrate_temperature.py` if missing) |
| `splits.csv` | `data/` | verify, stability (or regenerate with `prepare_data.py`) |

Install Git LFS **before** cloning (`git lfs install`), or run `git lfs pull` inside an existing clone. Without it you get tiny
pointer files and `torch.load` fails. Do not unzip `.pt` / `.npz` files: they are zip archives and must stay single files.

Only for training, verify and the stability check:
- **HAM10000** (images + `HAM10000_metadata.csv`), Kaggle "Skin Cancer MNIST: HAM10000".
- **ISIC 2019** (images + `ISIC_2019_Training_GroundTruth.csv`), optional, adds training images only.

To just run predictions you need the checkpoint and `temperature_calibration.pt`, nothing else.

## Quick setup check (no datasets needed)

```powershell
python -c "import torch; ck=torch.load('models/P2b_seed0_best.pt',map_location='cpu'); print(list(ck))"
python src/calibration/calibrate_temperature.py --logits models/ENS_logits.npz --out models/temperature_calibration.pt
python src/inference/verify_pipeline.py --skip-images
```
Expected: `['backbone', 'head']`, then `T = 1.9620`, then `calibrated TEST accuracy: 0.8501`.

## 3. Run the model on one image

Run from the repo root. Put each command on **one line** (PowerShell does not understand the bash `\` line break;
if you want several lines in PowerShell, end each line with a backtick `` ` ``).

```powershell
# only if temperature_calibration.pt is missing (T = 1.9620 with mel-bias 0.5)
python src/calibration/calibrate_temperature.py --logits models/ENS_logits.npz --out models/temperature_calibration.pt

python src/inference/predict_single_image.py --image lesion.jpg --checkpoint models/P2b_seed0_best.pt --temperature-file models/temperature_calibration.pt --gradcam | python src/decision/decision_layer.py --pretty
```

The short form `python predict_single_image.py ... | python decision_layer.py --pretty` works if you `cd` to a folder that holds both files.

Useful options: `--stability-check` (re-runs on 4 mild perturbations and adds a `stability` block; needs scikit-image), `--no-tta`,
`--gradcam-target-class mel`, `--gradcam-output-dir outputs/explainability`, `--lesion-mask mask.png`, `--device cpu|cuda`.
Grad-CAM images are written to `outputs/explainability/`. Logs go to stderr, stdout is JSON only.

## 4. Reproduce from scratch

```powershell
python src/data/prepare_data.py --ham-dir <ham10000-folder> --isic-dir <isic2019-folder>
python src/training/train_resnet18.py --splits data/splits.csv --extended data/train_extended.csv --out-dir models
python src/calibration/calibrate_temperature.py --logits models/ENS_logits.npz --out models/temperature_calibration.pt
python src/calibration/compare_calibration_methods.py --logits models/ENS_logits.npz --out-dir outputs
python src/calibration/logit_adjustment_analysis.py --logits models/ENS_logits.npz
python src/calibration/melanoma_bias_sweep.py --logits models/ENS_logits.npz
python src/explainability/stability_analysis.py --checkpoint models/P2b_seed0_best.pt --splits data/splits.csv --ham-dir <ham10000-folder>
python src/inference/verify_pipeline.py --image-root <ham10000-folder>
```
Training caches all resized images in RAM (about 4.3 GB with the ISIC extension); a GPU is strongly recommended.
`verify_pipeline.py` with the images should report a max logit difference of about 2.5e-5 and calibrated test accuracy 0.8501.

## 5. Reference results (HAM10000 test split, fold 0)

| | accuracy | NLL | Brier | ECE | melanoma recall |
|---|---|---|---|---|---|
| raw | 0.8543 | 0.4447 | 0.2114 | 0.0605 | 0.4589 |
| calibrated (T = 1.9620, mel bias 0.5) | 0.8501 | 0.3969 | 0.2078 | 0.0315 | 0.5479 |


## Method notes

- Split: `StratifiedGroupKFold` (7 folds, seed 42) grouped by `lesion_id`: fold 0 = test, fold 1 = validation. ISIC 2019 images (HAM10000 duplicates removed) are added to the training set only.
- Model: ImageNet ResNet18, 256x256, head `Dropout(0.3) + Linear(512, 7)`, plain cross-entropy, AdamW with layer-wise LR (2e-5 / 5e-5 / 2e-4 / 1e-3), cosine schedule, 16 epochs, best epoch by validation accuracy, 4-flip TTA.
- Calibration: `softmax((logits + beta * e_mel) / T)`, T fitted on validation NLL.
- Decision layer: rule-based thresholds (0.45 / 0.15 / 0.70), HIGH-risk classes mel / bcc / akiec, six-level escalation ladder.

**Research prototype, not a medical device. Output must not replace clinical judgement.**