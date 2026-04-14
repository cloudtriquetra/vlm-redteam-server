# VLM Red Team Server

A universal inference server for Vision-Language Models (VLMs). Host any VLM model behind a single FastAPI endpoint — no code changes needed to add new models.

Built for AI security researchers and red teamers who need a consistent, scriptable interface for testing VLM models.

---

## Features

- **Any model source** — HuggingFace Hub, local folder, URL download, or AWS S3
- **Multiple tasks** — OCR, image captioning, visual question answering (VQA)
- **Apple Silicon ready** — auto-detects and uses MPS (Metal Performance Shaders)
- **Lazy loading** — models load on first request and stay cached in memory
- **Zero code changes** — add models by editing `models.yaml` only
- **PyRIT compatible** — endpoint format designed for red team scanner integration

---

## Requirements

- Python 3.9+
- macOS with Apple Silicon (MPS) **or** any machine with CPU/CUDA
- ~2GB+ free RAM per model loaded

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For AWS S3 model sources, also install:
```bash
pip install boto3
```

### 2. Configure models

Edit `models.yaml` to register the models you want to host. See [Adding Models](#adding-models) below.

### 3. Start the server

```bash
uvicorn vlm_server:app --host 0.0.0.0 --port 8000
```

### 4. Verify it's running

```bash
curl http://localhost:8000/health
# {"status":"ok","device":"mps"}

curl http://localhost:8000/models
# lists all registered models and whether they are loaded
```

### 5. Run the smoke test

```bash
python3 examples/test_server.py
```

---

## API Reference

### `GET /health`
Returns server status and active compute device.

```json
{"status": "ok", "device": "mps"}
```

### `GET /models`
Lists all models registered in `models.yaml`.

```json
{
  "trocr-base-printed": {
    "source": "hf",
    "task": "ocr",
    "description": "Printed text OCR (0.3B)",
    "loaded": false
  }
}
```

### `POST /inference`
Run inference on a model.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model key from `models.yaml` |
| `image_base64` | string | yes* | Base64-encoded image |
| `prompt` | string | no | Text prompt or question. Required for VQA. Optional for captioning. OCR models also accept this as a smoke-test shortcut. |

*For OCR models only, `prompt` can be used instead of `image_base64` — the server renders it as an image internally.

**Example — OCR:**
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"model": "trocr-base-printed", "prompt": "Hello World"}'
```

**Example — real image:**
```bash
python3 examples/send_image.py /path/to/document.png --model trocr-base-printed
```

**Example — VQA:**
```bash
python3 examples/send_image.py photo.jpg \
  --model vilt-vqa \
  --prompt "What colour is the car?"
```

**Response:**
```json
{
  "model": "trocr-base-printed",
  "task": "ocr",
  "output": "Invoice Total $1,234.56",
  "device": "mps"
}
```

---

## Adding Models

Edit `models.yaml`. No Python knowledge required.

### From HuggingFace Hub

```yaml
models:
  my-model:
    source: hf
    repo: organisation/model-name-on-huggingface
    task: ocr           # ocr | captioning | vqa
    description: What this model does
```

Models are downloaded automatically on first request and cached at `~/.cache/huggingface/hub/`.

### From a local folder

```yaml
models:
  my-local-model:
    source: local
    path: /absolute/path/to/model/folder
    task: captioning
    description: My locally trained model
```

The folder must contain `config.json` and model weights (`model.safetensors` or `.bin`).

### From a URL

```yaml
models:
  partner-model:
    source: url
    url: https://example.com/models/vlm-v2.zip
    task: captioning
    description: Model received from partner
```

Supports `.zip` and `.tar.gz`. Downloaded and extracted once to `~/.cache/vlm_server/<model-key>/`.

### From AWS S3

```yaml
models:
  internal-model:
    source: s3
    bucket: my-ml-models-bucket
    key: models/internal-vlm-v3.tar.gz
    task: vqa
    description: Internal model
```

Requires `boto3` installed and AWS credentials configured:
```bash
pip install boto3
aws configure   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
```

---

## Supported Tasks

| Task | `task` value | Input | Notes |
|---|---|---|---|
| OCR | `ocr` | image (or text prompt for smoke test) | Reads printed/handwritten text from images |
| Image captioning | `captioning` | image + optional prompt | Generates a text description of an image |
| Visual Q&A | `vqa` | image + required prompt | Answers a natural language question about an image |

---

## Tested Models

| Model key | HuggingFace repo | Task | Size |
|---|---|---|---|
| `trocr-base-printed` | microsoft/trocr-base-printed | ocr | 0.3B |
| `trocr-large-printed` | microsoft/trocr-large-printed | ocr | 0.7B |
| `trocr-base-handwritten` | microsoft/trocr-base-handwritten | ocr | 0.3B |
| `blip-base` | Salesforce/blip-image-captioning-base | captioning | 0.4B |
| `blip-large` | Salesforce/blip-image-captioning-large | captioning | 1B |
| `git-base` | microsoft/git-base | captioning | 0.7B |
| `vilt-vqa` | dandelin/vilt-b32-finetuned-vqa | vqa | 0.2B |

---

## Project Structure

```
vlm-redteam-server/
├── vlm_server.py       # FastAPI server — universal model host
├── models.yaml         # Model registry — edit to add models
├── requirements.txt    # Python dependencies
├── examples/
│   ├── test_server.py  # Smoke test — run after starting server
│   └── send_image.py   # Send any image file to the server
└── README.md
```

---

## Troubleshooting

**MPS errors on Apple Silicon**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uvicorn vlm_server:app --host 0.0.0.0 --port 8000
```

**Model not loading**
- Check the model key matches exactly what is in `models.yaml`
- For `source: local`, confirm the path exists and contains `config.json`
- Check server logs — model load errors are printed with full tracebacks

**Slow first request**
- HuggingFace models download on first request (~0.3–5GB depending on model)
- Subsequent requests use the local cache and are fast

**Port already in use**
```bash
uvicorn vlm_server:app --host 0.0.0.0 --port 8001
```

---

## Red Team Integration

This server is designed to be the target for PyRIT-based red team scanners. The `/inference` endpoint accepts `image_base64` + `prompt` which maps directly to PyRIT's `PromptRequestPiece` model, making it straightforward to wrap as a custom `PromptTarget`.

---

## License

MIT
