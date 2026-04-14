# AI Red Team Server

A universal inference server for Vision-Language Models (VLMs) and Audio Models. Host any model behind a single FastAPI endpoint — no code changes needed to add new models.

Built for AI security researchers and red teamers who need a consistent, scriptable interface for testing models before scanning.

---

## Features

- **Any model source** — HuggingFace Hub, local folder, URL download, or AWS S3
- **Vision + Audio** — OCR, image captioning, VQA, speech-to-text, audio classification
- **Apple Silicon ready** — auto-detects MPS (Metal Performance Shaders)
- **EC2 / Linux ready** — auto-detects CUDA for GPU instances, falls back to CPU
- **Lazy loading** — models load on first request and stay cached in memory
- **Zero code changes** — add models by editing `models.yaml` only
- **PyRIT compatible** — endpoint format designed for red team scanner integration

---

## Requirements

- Python 3.9+
- macOS with Apple Silicon **or** any EC2/Linux VM (CPU or GPU)
- ~2GB+ free RAM per model loaded

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For AWS S3 model sources:
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

## EC2 / RHEL Setup

```bash
# Install Python
sudo dnf install -y python3.11 python3.11-pip git

# Clone repo
git clone https://github.com/cloudtriquetra/ai-redteam-server
cd ai-redteam-server

# CPU instance (t3.large)
pip3 install -r requirements.txt

# GPU instance (g4dn.xlarge — CUDA 12.1)
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt

# Start server (persistent after SSH disconnect)
nohup uvicorn vlm_server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Check logs
tail -f server.log
```

Open port **8000** in your EC2 security group inbound rules.

---

## API Reference

### `GET /health`

```json
{"status": "ok", "device": "mps"}
```

### `GET /models`

```json
{
  "trocr-base-printed": {
    "source": "hf",
    "task": "image-to-text",
    "description": "Printed text OCR (0.3B)",
    "custom_loader": false,
    "loaded": false
  }
}
```

### `POST /inference`

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model key from `models.yaml` |
| `image_base64` | string | vision models | Base64-encoded image (PNG, JPG) |
| `audio_base64` | string | audio models | Base64-encoded audio (WAV, MP3, FLAC) |
| `prompt` | string | depends | Question for VQA. Optional for captioning. Text shortcut for OCR smoke testing. |

**Example — OCR:**
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"model": "trocr-base-printed", "prompt": "Hello World"}'
```

**Example — image file:**
```bash
python3 examples/send_image.py /path/to/document.png --model trocr-base-printed
```

**Example — VQA:**
```bash
python3 examples/send_image.py photo.jpg \
  --model vilt-vqa \
  --prompt "What colour is the car?"
```

**Example — speech to text:**
```bash
python3 examples/send_audio.py recording.wav --model whisper-base
```

**Example — against EC2:**
```bash
python3 examples/send_audio.py recording.wav \
  --host http://YOUR_EC2_IP:8000 \
  --model whisper-small
```

**Response:**
```json
{
  "model": "whisper-base",
  "task": "automatic-speech-recognition",
  "output": "The quick brown fox jumps over the lazy dog",
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
    task: image-to-text     # see Supported Tasks below
    description: What this model does
```

Models download automatically on first request and cache at `~/.cache/huggingface/hub/`.

### From a local folder

```yaml
models:
  my-local-model:
    source: local
    path: /absolute/path/to/model/folder
    task: image-to-text
    description: My locally trained model
```

The folder must contain `config.json` and model weights (`model.safetensors` or `.bin`).

### From a URL

```yaml
models:
  partner-model:
    source: url
    url: https://example.com/models/my-model-v1.zip
    task: image-to-text
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
    task: automatic-speech-recognition
    description: Internal model
```

Requires `boto3` and AWS credentials:
```bash
pip install boto3
aws configure   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
```

### Custom loader fallback

Most models load automatically via HuggingFace `pipeline()`. For rare models that need non-standard loading, add `custom_loader: true` and use the legacy task name:

```yaml
models:
  unusual-model:
    source: hf
    repo: some-org/unusual-model
    task: ocr               # legacy names: ocr | captioning | vqa |
    custom_loader: true     #   speech-to-text | audio-classification
    description: Needs custom loading
```

---

## Supported Tasks

The `task` field in `models.yaml` maps directly to HuggingFace pipeline task names. The server uses `pipeline()` automatically — no model-specific code needed.

### Vision

| Task | `task` value | Input | Notes |
|---|---|---|---|
| OCR / image to text | `image-to-text` | image or text prompt | TrOCR, BLIP, GIT, LLaVA all use this |
| Visual Q&A | `visual-question-answering` | image + required prompt | Answers a question about an image |
| Image classification | `image-classification` | image | Returns class label |
| Document Q&A | `document-question-answering` | image + required prompt | Extracts answers from document images |

### Audio

| Task | `task` value | Input | Notes |
|---|---|---|---|
| Speech to text | `automatic-speech-recognition` | audio file | Auto-resamples to 16kHz |
| Audio classification | `audio-classification` | audio file | Returns class label |

Audio supports WAV, MP3, FLAC, OGG, M4A — any sample rate is automatically resampled.

### Legacy task names (custom_loader only)

If `custom_loader: true` is set, these original task names are still supported for backward compatibility: `ocr`, `captioning`, `vqa`, `speech-to-text`, `audio-classification`.

---

## Tested Models

### Vision models

| Model key | HuggingFace repo | Task | Size |
|---|---|---|---|
| `trocr-base-printed` | microsoft/trocr-base-printed | image-to-text | 0.3B |
| `trocr-large-printed` | microsoft/trocr-large-printed | image-to-text | 0.7B |
| `trocr-base-handwritten` | microsoft/trocr-base-handwritten | image-to-text | 0.3B |
| `blip-base` | Salesforce/blip-image-captioning-base | image-to-text | 0.4B |
| `blip-large` | Salesforce/blip-image-captioning-large | image-to-text | 1B |
| `git-base` | microsoft/git-base | image-to-text | 0.7B |
| `vilt-vqa` | dandelin/vilt-b32-finetuned-vqa | visual-question-answering | 0.2B |
| `moondream2` | vikhyatk/moondream2 | image-to-text | 1.8B |
| `llava-1.5-7b` | llava-hf/llava-1.5-7b-hf | image-to-text | 7B — needs ~14GB RAM |

### Audio models

| Model key | HuggingFace repo | Task | Size |
|---|---|---|---|
| `whisper-base` | openai/whisper-base | automatic-speech-recognition | 74M |
| `whisper-small` | openai/whisper-small | automatic-speech-recognition | 244M |
| `wav2vec2-base` | facebook/wav2vec2-base-960h | automatic-speech-recognition | 95M |
| `audio-classifier` | MIT/ast-finetuned-audioset-10-10-0.4593 | audio-classification | 87M |

---

## Project Structure

```
ai-redteam-server/
├── vlm_server.py         # FastAPI server — universal model host
├── models.yaml           # Model registry — edit to add models
├── requirements.txt      # Python dependencies
├── examples/
│   ├── test_server.py    # Smoke test — run after starting server
│   ├── send_image.py     # Send any image file to the server
│   └── send_audio.py     # Send any audio file to the server
└── README.md
```

---

## Infrastructure Guide

| Model size | Examples | Recommended instance | Cost |
|---|---|---|---|
| Small (< 1B) | TrOCR, Whisper-base, BLIP-base | t3.large (CPU) or Mac Air | ~$0.08/hr or free |
| Medium (1–3B) | BLIP-large, Whisper-small, Moondream2 | g4dn.xlarge (T4 GPU) | ~$0.53/hr |
| Large (7B+) | LLaVA-1.5-7b | g4dn.xlarge (16GB VRAM) or g5.xlarge | ~$0.53–$1.00/hr |

Tip: only run GPU instances during active red team sessions to keep costs low.

---

## Troubleshooting

**MPS errors on Apple Silicon**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uvicorn vlm_server:app --host 0.0.0.0 --port 8000
```

**`accelerate` not found / float16 warning**
`pipeline()` with float16 requires `accelerate`. It is in `requirements.txt` but if you see this error:
```bash
pip install accelerate
```

**transformers version conflict**
```bash
pip install "transformers>=4.40.0,<5.0.0" --force-reinstall
```

**Audio sample rate error**
Audio is auto-resampled via librosa — ensure `librosa` and `soundfile` are installed:
```bash
pip install librosa soundfile
```

**Model not loading**
- Check the model key matches exactly what is in `models.yaml`
- For `source: local`, confirm the path exists and contains `config.json`
- Check server logs — model load errors are printed with full tracebacks

**pipeline() fails for a specific model**
A small number of older or non-standard models don't work with `pipeline()`. Add `custom_loader: true` to the model entry in `models.yaml` and use a legacy task name (`ocr`, `captioning`, `vqa`, `speech-to-text`, `audio-classification`).

**Slow first request**
HuggingFace models download on first request (~0.3–14GB depending on model). Subsequent requests use the local cache and are fast.

**Port already in use**
```bash
uvicorn vlm_server:app --host 0.0.0.0 --port 8001
```

---

## Red Team Integration

This server is designed to be the target for PyRIT-based red team scanners. The `/inference` endpoint accepts `image_base64`, `audio_base64`, and `prompt` — making it straightforward to wrap as a custom `PromptTarget` for both vision and audio attack scenarios.

---

## License

MIT
