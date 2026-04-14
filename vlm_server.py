"""
vlm_server.py — Universal AI inference server
================================================
Hosts any Vision-Language or Audio Model behind a unified FastAPI endpoint.
Supports HuggingFace Hub, local folders, URL downloads, and AWS S3.

Usage:
    uvicorn vlm_server:app --host 0.0.0.0 --port 8000

Add models in models.yaml — no code changes needed.
"""

import base64
import io
import logging
import os
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")       # Apple Silicon
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")      # EC2 GPU
else:
    DEVICE = torch.device("cpu")       # EC2 CPU / fallback
log.info(f"Device: {DEVICE}")

# ── Registry ──────────────────────────────────────────────────────────────────

YAML_PATH = Path(__file__).parent / "models.yaml"

if not YAML_PATH.exists():
    raise FileNotFoundError(f"models.yaml not found at {YAML_PATH}")

with open(YAML_PATH) as f:
    REGISTRY: dict = yaml.safe_load(f)["models"]

log.info(f"Registry loaded — {len(REGISTRY)} model(s): {list(REGISTRY.keys())}")

# ── In-memory model cache ─────────────────────────────────────────────────────

CACHE: dict = {}

# ── Source resolvers ──────────────────────────────────────────────────────────

def _extract_archive(archive_path: str, dest: Path) -> str:
    """Extract zip or tar.gz to dest. Returns the model directory."""
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive_path) as t:
            t.extractall(dest)

    os.unlink(archive_path)

    contents = list(dest.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        return str(contents[0])
    return str(dest)


def resolve_model_path(model_key: str, entry: dict) -> str:
    """
    Resolve the model source to a local filesystem path that
    HuggingFace from_pretrained() can load from.
    """
    source = entry.get("source", "hf")

    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    if source == "hf":
        return entry["repo"]

    # ── Local folder ──────────────────────────────────────────────────────────
    if source == "local":
        path = Path(entry["path"]).expanduser()
        if not path.exists():
            raise ValueError(f"[{model_key}] Local path not found: {path}")
        log.info(f"[{model_key}] Using local path: {path}")
        return str(path)

    # ── URL download ──────────────────────────────────────────────────────────
    if source == "url":
        cache_dir = Path.home() / ".cache" / "vlm_server" / model_key
        if cache_dir.exists():
            log.info(f"[{model_key}] Using cached download: {cache_dir}")
            contents = list(cache_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                return str(contents[0])
            return str(cache_dir)

        url = entry["url"]
        log.info(f"[{model_key}] Downloading from {url} ...")
        cache_dir.mkdir(parents=True, exist_ok=True)

        suffix = ".zip" if url.endswith(".zip") else ".tar.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            tmp_path = tmp.name

        log.info(f"[{model_key}] Extracting ...")
        return _extract_archive(tmp_path, cache_dir)

    # ── AWS S3 ────────────────────────────────────────────────────────────────
    if source == "s3":
        cache_dir = Path.home() / ".cache" / "vlm_server" / model_key
        if cache_dir.exists():
            log.info(f"[{model_key}] Using cached S3 download: {cache_dir}")
            contents = list(cache_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                return str(contents[0])
            return str(cache_dir)

        try:
            import boto3
        except ImportError:
            raise RuntimeError(
                f"[{model_key}] boto3 is required for S3 sources. "
                "Run: pip install boto3"
            )

        bucket = entry["bucket"]
        key    = entry["key"]
        log.info(f"[{model_key}] Downloading s3://{bucket}/{key} ...")
        cache_dir.mkdir(parents=True, exist_ok=True)

        suffix   = ".zip" if key.endswith(".zip") else ".tar.gz"
        tmp_path = str(cache_dir / f"download{suffix}")
        boto3.client("s3").download_file(bucket, key, tmp_path)

        log.info(f"[{model_key}] Extracting ...")
        return _extract_archive(tmp_path, cache_dir)

    raise ValueError(f"[{model_key}] Unknown source: '{source}'. "
                     "Must be one of: hf | local | url | s3")


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_key: str) -> tuple:
    """Load and cache a model. Returns (processor, model, task)."""
    if model_key in CACHE:
        return CACHE[model_key]

    if model_key not in REGISTRY:
        raise ValueError(
            f"Model '{model_key}' not found in models.yaml. "
            f"Available: {list(REGISTRY.keys())}"
        )

    entry      = REGISTRY[model_key]
    task       = entry["task"]
    model_path = resolve_model_path(model_key, entry)

    log.info(f"[{model_key}] Loading (task={task}) ...")

    # ── Vision tasks ──────────────────────────────────────────────────────────

    if task == "ocr":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained(model_path)
        model     = VisionEncoderDecoderModel.from_pretrained(model_path).to(DEVICE)

    elif task == "captioning":
        from transformers import BlipProcessor, BlipForConditionalGeneration
        try:
            processor = BlipProcessor.from_pretrained(model_path)
            model     = BlipForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
        except Exception:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_path)
            try:
                from transformers import AutoModelForVision2Seq
                model = AutoModelForVision2Seq.from_pretrained(model_path).to(DEVICE)
            except ImportError:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)

    elif task == "vqa":
        from transformers import ViltProcessor, ViltForQuestionAnswering
        processor = ViltProcessor.from_pretrained(model_path)
        model     = ViltForQuestionAnswering.from_pretrained(model_path).to(DEVICE)

    # ── Audio tasks ───────────────────────────────────────────────────────────

    elif task == "speech-to-text":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(model_path)
        model     = WhisperForConditionalGeneration.from_pretrained(model_path).to(DEVICE)

    elif task == "audio-classification":
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        processor = AutoFeatureExtractor.from_pretrained(model_path)
        model     = AutoModelForAudioClassification.from_pretrained(model_path).to(DEVICE)

    else:
        raise ValueError(
            f"[{model_key}] Unsupported task: '{task}'. "
            "Must be one of: ocr | captioning | vqa | speech-to-text | audio-classification"
        )

    model.eval()
    CACHE[model_key] = (processor, model, task)
    log.info(f"[{model_key}] Ready.")
    return processor, model, task


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def decode_audio(b64: str, target_sr: int = 16000):
    """
    Decode base64 audio and return (audio_array, sample_rate).
    Auto-resamples to target_sr (default 16000 Hz for Whisper/Wav2Vec2).
    Handles WAV, MP3, FLAC, OGG, M4A via librosa.
    """
    import librosa
    audio_bytes = base64.b64decode(b64)
    # librosa handles all formats and resamples in one step
    audio_array, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=target_sr,   # resample to target on load
        mono=True,      # convert stereo to mono
    )
    log.info(f"Audio decoded — sr={sample_rate}Hz, duration={len(audio_array)/sample_rate:.1f}s")
    return audio_array, sample_rate


def render_text_as_image(text: str) -> Image.Image:
    """Render plain text onto a white image. Used for OCR smoke-testing."""
    from PIL import ImageDraw, ImageFont
    img  = Image.new("RGB", (600, 80), "white")
    draw = ImageDraw.Draw(img)
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ]
    font = None
    for candidate in font_candidates:
        try:
            font = ImageFont.truetype(candidate, 32)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((10, 20), text, fill="black", font=font)
    return img


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Universal AI Red Team Server",
    description="Host any Vision-Language or Audio Model and expose it for inference and red teaming.",
    version="2.0.0",
)


class InferenceRequest(BaseModel):
    model:        str
    image_base64: str | None = None   # vision models — base64 image
    audio_base64: str | None = None   # audio models  — base64 WAV/MP3
    prompt:       str | None = None   # text prompt / question


class InferenceResponse(BaseModel):
    model:  str
    task:   str
    output: str
    device: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/models", summary="List all registered models")
def list_models():
    return {
        key: {
            "source":      val.get("source", "hf"),
            "task":        val["task"],
            "description": val.get("description", ""),
            "loaded":      key in CACHE,
        }
        for key, val in REGISTRY.items()
    }


@app.post("/inference", response_model=InferenceResponse, summary="Run inference")
async def infer(req: InferenceRequest):
    # Load model (from cache or fresh)
    try:
        processor, model, task = load_model(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Model load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Run inference
    try:
        with torch.no_grad():

            # ── Vision ────────────────────────────────────────────────────────

            if task == "ocr":
                if req.image_base64:
                    image = decode_image(req.image_base64)
                elif req.prompt:
                    image = render_text_as_image(req.prompt)
                else:
                    raise HTTPException(status_code=400,
                        detail="OCR requires 'image_base64' or 'prompt'.")
                px  = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
                ids = model.generate(px)
                out = processor.batch_decode(ids, skip_special_tokens=True)[0]

            elif task == "captioning":
                if not req.image_base64:
                    raise HTTPException(status_code=400,
                        detail="Captioning requires 'image_base64'.")
                image  = decode_image(req.image_base64)
                inputs = processor(image, req.prompt or "", return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                ids    = model.generate(**inputs, max_new_tokens=64)
                out    = processor.decode(ids[0], skip_special_tokens=True)

            elif task == "vqa":
                if not req.image_base64 or not req.prompt:
                    raise HTTPException(status_code=400,
                        detail="VQA requires both 'image_base64' and 'prompt'.")
                image  = decode_image(req.image_base64)
                inputs = processor(image, req.prompt, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                logits = model(**inputs).logits
                out    = model.config.id2label[logits.argmax(-1).item()]

            # ── Audio ─────────────────────────────────────────────────────────

            elif task == "speech-to-text":
                if not req.audio_base64:
                    raise HTTPException(status_code=400,
                        detail="speech-to-text requires 'audio_base64'.")
                audio_array, sample_rate = decode_audio(req.audio_base64)
                inputs = processor(
                    audio_array,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                ids    = model.generate(**inputs)
                out    = processor.batch_decode(ids, skip_special_tokens=True)[0]

            elif task == "audio-classification":
                if not req.audio_base64:
                    raise HTTPException(status_code=400,
                        detail="audio-classification requires 'audio_base64'.")
                audio_array, sample_rate = decode_audio(req.audio_base64)
                inputs = processor(
                    audio_array,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                logits = model(**inputs).logits
                out    = model.config.id2label[logits.argmax(-1).item()]

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Inference error [{req.model}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return InferenceResponse(
        model=req.model,
        task=task,
        output=out,
        device=str(DEVICE),
    )
