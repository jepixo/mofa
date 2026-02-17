# MOFA

Neutral-identity face reconstruction scaffold that accepts 1–3 images and outputs a neutralized 3D head mesh, UV texture, and export artifacts.

> **Status:** current models are structural placeholders (random/untrained behavior). Topology/pipeline/export flow are implemented first; production quality requires training your own weights and validated data licensing.

## Features
- Procedural base head mesh generation
- UV layout + preview generation
- Nine-stage pipeline orchestration
- GLB/FBX placeholder exporters + identity JSON export
- CLI for end-to-end execution

## Requirements
- Python 3.10–3.12
- `pip`, `venv`
- Optional GPU: CUDA-compatible NVIDIA card + matching PyTorch build

## Dependency Notes (Important)
To avoid `numpy`/`f2py.exe` install churn (especially on Windows), this repo pins NumPy to a known stable line.

Install from `requirements.txt` (already pinned):

```bash
pip install -r requirements.txt
```

If your environment is already messy, create a fresh virtual environment first (recommended below).

---

## Quick Start (Linux)

```bash
# 1) Clone
cd /path/to
# git clone <repo-url>
cd mofa

# 2) Create env
python3 -m venv .venv
source .venv/bin/activate

# 3) Upgrade installer tooling
python -m pip install --upgrade pip setuptools wheel

# 4) Install deps
pip install -r requirements.txt

# 5) Run tests
python -m pytest tests -v

# 6) Run CLI
python cli.py --input front.jpg --output ./output
```

---

## Windows Setup

### Recommended: WSL2 (best parity with cloud/Linux)
Since your cloud target is Linux-first, **WSL2 is the preferred local workflow**.

1. Open WSL Ubuntu.
2. Follow the exact Linux steps above.
3. Keep project files inside the Linux filesystem for best performance (e.g. `~/src/mofa`).

### Native Windows (supported, but more fragile)

```powershell
# In PowerShell
cd J:\dripboard\mofa
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m pytest tests -v
python cli.py --input front.jpg --output .\output
```

If you hit the `f2py.exe.deleteme` / WinError 2 issue:

1. Use a **fresh venv** (most reliable fix).
2. Avoid mixing global Python/site-packages with project env.
3. Repair Python launcher/scripts if needed:
   - Re-run Python installer with **Repair**.
4. Re-run install with no cache:

```powershell
pip install --no-cache-dir -r requirements.txt
```

---

## GCP Deployment Guide (Cloud-Optimized Path)

### Recommended architecture
- **Inference API**: FastAPI service wrapping `MofaPipeline`
- **Container runtime**: Cloud Run (CPU) for baseline, or GKE/GCE GPU VM for accelerated inference
- **Storage**: Cloud Storage bucket for input/output artifacts (`face.glb`, `albedo.png`, `identity.json`)
- **Async jobs**: Cloud Tasks / Pub/Sub for queue-based processing
- **Observability**: Cloud Logging + Cloud Monitoring

### Option A: Cloud Run (CPU-first MVP)
Best for low ops burden and easy autoscaling.

1. Add a minimal API server (FastAPI/Flask) that accepts images and returns artifact URLs.
2. Build container image.
3. Push to Artifact Registry.
4. Deploy to Cloud Run.

High-level commands:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>

gcloud builds submit --tag <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/mofa:latest

gcloud run deploy mofa-api \
  --image <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/mofa:latest \
  --region <REGION> \
  --platform managed \
  --allow-unauthenticated
```

### Option B: GPU inference (recommended for production latency)
Cloud Run GPU is region/feature dependent; otherwise use:
- **GCE VM with NVIDIA T4/L4/A10** + Docker
- or **GKE node pool with GPUs**

Use this when you enable heavier learned models (identity/texture completion) and need predictable latency.

### Data + security checklist for cloud
- Do not store raw user images longer than needed.
- Encrypt storage at rest (default in GCP) + IAM least privilege.
- Add signed URLs for artifact downloads.
- Add retention + deletion policies for biometric outputs.
- Log model version/hash with each job result.

---

## CLI

```bash
python cli.py --input front.jpg [--left left.jpg] [--right right.jpg] --output ./output
```

Options:
- `--device cpu|cuda`
- `--texture-size 1024|2048`
- `--format glb|fbx`

---

## Testing

```bash
python -m pytest tests -v
```

If tests fail during dependency import, verify your venv and reinstall dependencies in a clean environment.
