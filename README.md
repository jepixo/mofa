# MOFA

Neutral-identity face reconstruction scaffold that accepts 1–3 images and outputs a neutralized 3D head mesh, UV texture, and export artifacts.

> **Status:** current models are structural placeholders (random/untrained behavior). Topology/pipeline/export flow are implemented first; production quality requires your own trained weights and properly licensed data.

## TL;DR (Recommended Dev Workflow)
If you are on Windows, run MOFA inside **WSL2 + Conda**. This avoids native-Windows package issues and matches Linux/GCP deployment targets.

---

## 1) WSL2 + Conda Setup (Primary Path)

### Install WSL2 (once, from PowerShell as admin)
```powershell
wsl --install
```
Then reboot if prompted and install Ubuntu from Microsoft Store.

### Inside Ubuntu (WSL shell)

```bash
# 1) Clone
cd ~
# git clone <repo-url>
cd mofa

# 2) Install Miniconda (if you don't have conda)
# https://docs.conda.io/en/latest/miniconda.html
# Example:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

# 3) Create environment from file
conda env create -f environment.yml
conda activate mofa

# 4) Verify imports
python -c "import torch, numpy, cv2, mediapipe; print('ok')"

# 5) Run tests
python -m pytest tests -v

# 6) Run CLI
python cli.py --input front.jpg --output ./output
```

### Update environment later
```bash
conda env update -f environment.yml --prune
```

---

## 2) Linux (Non-WSL)
Use the same Conda workflow as above:

```bash
conda env create -f environment.yml
conda activate mofa
python -m pytest tests -v
```

---

## 3) Native Windows
Not recommended for this project. Use WSL2 instead.

Reason: native Windows often hits Python packaging/script issues (`f2py.exe`, mixed global/venv installs, wheel conflicts). WSL2 gives Linux parity and fewer setup problems.

---

## 4) GCP Hosting (Cloud-Optimized)

### Recommended architecture
- **API layer:** FastAPI service wrapping `MofaPipeline`
- **Compute:**
  - Cloud Run (CPU) for simple MVP
  - GCE/GKE GPU for production latency
- **Artifacts:** Cloud Storage (`face.glb`, `albedo.png`, `identity.json`)
- **Async jobs:** Pub/Sub or Cloud Tasks
- **Observability:** Cloud Logging + Monitoring

### Cloud Run baseline
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>

gcloud builds submit --tag <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/mofa:latest

gcloud run deploy mofa-api \
  --image <REGION>-docker.pkg.dev/<PROJECT_ID>/<REPO>/mofa:latest \
  --region <REGION> \
  --platform managed
```

### GPU path (recommended for production)
Use GCE VM or GKE with NVIDIA GPUs when you enable heavier learned models and need tighter latency/SLA.

### Security/compliance checklist
- Treat outputs as biometric data.
- Minimize raw-image retention.
- Use IAM least privilege.
- Use signed URLs for download.
- Add deletion and retention policies.
- Log model version/hash for each inference.

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

## Repo Files for Environment
- `environment.yml` → **primary** setup for WSL/Linux (Conda)
- `requirements.txt` → optional pip fallback
