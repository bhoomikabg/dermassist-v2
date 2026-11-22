# DermAssist V2

A fresh React/Vite and FastAPI application for academic skin-lesion research and awareness. The new application is separate from both existing project folders.

## Run locally (Windows PowerShell)

Open two PowerShell windows in `C:\Desktop\DermAssist-V2` (or the folder where this deliverable is saved).

Backend:

```powershell
Set-Location C:\Desktop\DermAssist-V2\backend
& 'C:\Users\Rog\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

If Python is registered with the Windows launcher, `py -3.11 -m venv .venv` also works in place of the explicit interpreter path above.

```powershell
& 'C:\Users\Rog\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
```

Frontend (Node.js 22.12 or newer):

```powershell
Set-Location C:\Desktop\DermAssist-V2\frontend
npm ci
npm run dev
```

Open the frontend at <http://localhost:5173>. The API is at <http://localhost:8000>; interactive API documentation is at <http://localhost:8000/docs>. Health check: <http://localhost:8000/api/health>.

If Python, PyTorch or torchvision are unavailable for your Python version, install a supported Python build and the corresponding PyTorch/torchvision wheels first. The backend selects CUDA when available and otherwise uses CPU. The model can take longer to load or analyse an image on CPU.

## Model integration

The backend includes the new repository's actual `ResNet18` architecture, 256 × 256 RGB resize, ImageNet mean/std normalization, four-flip test-time averaging, temperature and melanoma-logit-bias calibration, decision layer and `layer4` Grad-CAM implementation. The seven output labels in their checkpoint order are `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`. The matching `P2b_seed0_best.pt` checkpoint and `temperature_calibration.pt` artifact are kept together under `backend/models/`. `ENS_logits.npz` is not required for prediction; it is used for validation/calibration analysis in the source repository.

Default upload limit is 12 MiB. Accepted types are JPEG, PNG and WebP. The API returns calibrated model probabilities, the repository's rule-based decision output, calibration metadata and URLs for original, heatmap and overlay images. Results are academic research outputs and are not medical diagnoses.

Optional environment settings: `DERMASSIST_CHECKPOINT`, `DERMASSIST_CALIBRATION`, and comma-separated `CORS_ORIGINS`. Their defaults point to the bundled model files and Vite localhost origin.

## Frontend production build

```powershell
cd frontend
npm run build
```

## Verification status

- Repository inspection confirmed full-size checkpoint (44,800,651 bytes), calibration artifact (1,613 bytes), and ensemble logits archive (187,434 bytes); the checkpoint and calibration files begin with ZIP/PyTorch archive signatures rather than Git LFS pointer text.
- The source README documents the architecture, preprocessing, seven class order, four-flip TTA, calibration formula, and decision layer. The repository's own reported metrics are not represented as tests of this application.
- Standalone CPU inference on the repository's `nv.jpg` fixture completed. It produced seven calibrated probabilities summing to 1, an `nv` top prediction, the decision layer's `low` / `low_risk_monitor` result, and a 256 × 256 Grad-CAM map.
- The repository's `verify_pipeline.py --skip-images` check completed against its saved logits and reported temperature 1.9620, melanoma bias 0.5, and calibrated TEST accuracy 0.8501. This is the repository's saved-logit verification result, not a new external clinical performance claim.
- The FastAPI health endpoint returned `ok` with model and calibration loaded. A JPEG upload returned the seven classes, prediction, decision, and all three explainability URLs; HEAD checks for original, heatmap and overlay each returned HTTP 200.
- Upload validation returned HTTP 415 for a text file, HTTP 400 for an unreadable image, and HTTP 413 for a file over 12 MiB. Decision-layer checks produced the expected high / urgent specialist review, moderate / practitioner review, and low / low-risk monitor outputs for representative class-probability inputs.
- Frontend `npm ci` completed with zero reported vulnerabilities and `npm run build` succeeded. Browser review confirmed the homepage and analysis route render, a sample JPEG previews, and an unavailable-backend error is shown clearly.
- The production build emits non-fatal module-level directive warnings from React Router and Lucide packages. Responsive device dimensions were not separately measured.

## Contact and imagery

No configured team contact address was found in the new model repository. The contact control opens the user's email application with a prefilled request and deliberately does not invent a recipient address. Add a verified address to the `mailto:` link if one becomes available. The homepage carousel references professional Unsplash image URLs and therefore needs an internet connection to show the photographs.

## Research disclaimer

DermAssist is an academic research project developed for educational and experimental purposes. Its AI-generated outputs are not medical diagnoses and should not replace professional medical consultation.
