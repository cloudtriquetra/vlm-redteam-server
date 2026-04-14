"""
vlm_server.py — Universal AI inference server
================================================
Hosts any Vision-Language or Audio Model behind a unified FastAPI endpoint.
Supports HuggingFace Hub, local folders, URL downloads, and AWS S3.

Uses HuggingFace pipeline() for all models — no model-specific loader code.
Adding a new model requires only a models.yaml entry, zero Python changes.

For models that do not work with pipeline() (non-standard architectures),
set custom_loader: true in models.yaml to use the legacy specific loaders.

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
from transformers import pipeline as hf_pipeline

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

# ── Task routing ──────────────────────────────────────────────────────────────
# Maps HuggingFace pipeline task names to input modality.
# To support a new task just add it here — no other changes needed.

IMAGE_TASKS = {
    "image-to-text",
    "visual-question-answering",
    "image-classification",
    "zero-shot-image-classification",
    "document-question-answering",
}

AUDIO_TASKS = {
    "automatic-speech-recognition",
    "audio-classification",
    "zero-shot-audio-classification",
}

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
    source = entry.get("source", "hf")

    if source == "hf":
        return entry["repo"]

    if source == "local":
        path = Path(entry["path"]).expanduser()
        if not path.exists():
            raise ValueError(f"[{model_key}] Local path not found: {path}")
        return str(path)

    if source == "url":
        cache_dir = Path.home() / ".cache" / "vlm_server" / model_key
        if cache_dir.exists():
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
        return _extract_archive(tmp_path, cache_dir)

    if source == "s3":
        cache_dir = Path.home() / ".cache" / "vlm_server" / model_key
        if cache_dir.exists():
            contents = list(cache_dir.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                return str(contents[0])
            return str(cache_dir)
        try:
            import boto3
        except ImportError:
            raise RuntimeError(f"[{model_key}] boto3 required. Run: pip install boto3")
        bucket, key = entry["bucket"], entry["key"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix   = ".zip" if key.endswith(".zip") else ".tar.gz"
        tmp_path = str(cache_dir / f"download{suffix}")
        boto3.client("s3").download_file(bucket, key, tmp_path)
        return _extract_archive(tmp_path, cache_dir)

    raise ValueError(f"[{model_key}] Unknown source: '{source}'")


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_via_pipeline(model_key: str, task: str, model_path: str):
    """Generic loader using HuggingFace pipeline(). Works for most models."""
    dtype      = torch.float16 if DEVICE.type in ("cuda", "mps") else torch.float32
    device_arg = "mps" if DEVICE.type == "mps" else DEVICE
    pipe = hf_pipeline(
        task=task,
        model=model_path,
        device=device_arg,
        torch_dtype=dtype,
    )
    return ("pipeline", pipe, task)


def _load_via_custom(model_key: str, task: str, model_path: str):
    """
    Legacy custom loader for models that don't work with pipeline().
    Used when custom_loader: true is set in models.yaml.
    Preserves backward compatibility with the original task names:
    ocr | captioning | vqa | speech-to-text | audio-classification
    """
    log.info(f"[{model_key}] Using custom loader")

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
            f"[{model_key}] custom_loader=true but task '{task}' has no custom handler."
        )

    model.eval()
    return ("custom", processor, model, task)


def load_model(model_key: str) -> tuple:
    """
    Load and cache a model. Returns either:
      ("pipeline", pipe, task)            for pipeline() models
      ("custom", processor, model, task)  for custom-loaded models
    """
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

    if entry.get("custom_loader", False):
        result = _load_via_custom(model_key, task, model_path)
    else:
        result = _load_via_pipeline(model_key, task, model_path)

    CACHE[model_key] = result
    log.info(f"[{model_key}] Ready.")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def decode_audio(b64: str, target_sr: int = 16000):
    """Decode base64 audio, auto-resample to target_sr via librosa."""
    import librosa
    audio_bytes = base64.b64decode(b64)
    audio_array, sample_rate = librosa.load(
        io.BytesIO(audio_bytes), sr=target_sr, mono=True,
    )
    log.info(f"Audio decoded — sr={sample_rate}Hz, duration={len(audio_array)/sample_rate:.1f}s")
    return audio_array, sample_rate


def render_text_as_image(text: str) -> Image.Image:
    """Render plain text onto a white image. Used for OCR smoke-testing."""
    from PIL import ImageDraw, ImageFont
    img  = Image.new("RGB", (600, 80), "white")
    draw = ImageDraw.Draw(img)
    for candidate in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(candidate, 32)
            break
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()
    draw.text((10, 20), text, fill="black", font=font)
    return img


def _extract_output(result) -> str:
    """Normalise HuggingFace pipeline() output to a plain string."""
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        for key in ("generated_text", "text", "answer", "label"):
            if key in result:
                return str(result[key])
    return str(result)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Universal AI Red Team Server",
    description="Host any Vision-Language or Audio Model for inference and red teaming.",
    version="3.0.0",
)


class InferenceRequest(BaseModel):
    model:        str
    image_base64: str | None = None
    audio_base64: str | None = None
    prompt:       str | None = None


class InferenceResponse(BaseModel):
    model:  str
    task:   str
    output: str
    device: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/models")
def list_models():
    return {
        key: {
            "source":        val.get("source", "hf"),
            "task":          val["task"],
            "description":   val.get("description", ""),
            "custom_loader": val.get("custom_loader", False),
            "loaded":        key in CACHE,
        }
        for key, val in REGISTRY.items()
    }


@app.post("/inference", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    try:
        cached = load_model(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Model load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    loader_type = cached[0]
    task        = cached[-1]

    try:
        with torch.no_grad():

            # ── Generic pipeline() path ───────────────────────────────────────
            if loader_type == "pipeline":
                _, pipe, _ = cached

                if task in IMAGE_TASKS:
                    if req.image_base64:
                        image = decode_image(req.image_base64)
                    elif req.prompt and task == "image-to-text":
                        image = render_text_as_image(req.prompt)
                    else:
                        raise HTTPException(400,
                            "Image tasks require 'image_base64'. "
                            "image-to-text also accepts 'prompt' for smoke testing.")

                    kwargs = {}
                    if req.prompt and task in (
                        "visual-question-answering",
                        "document-question-answering",
                    ):
                        kwargs["question"] = req.prompt

                    out = _extract_output(pipe(image, **kwargs))

                elif task in AUDIO_TASKS:
                    if not req.audio_base64:
        raise HTTPException(400, f"'{task}' requires 'audio_base64'.")
    audio_array, sample_rate = decode_audio(req.audio_base64)
    kwargs = REGISTRY[req.model].get("pipeline_kwargs", {})
    out = _extract_output(
        pipe({"array": audio_array, "sampling_rate": sample_rate}, **kwargs)
    )
                else:
                    raise HTTPException(400,
                        f"Task '{task}' not in IMAGE_TASKS or AUDIO_TASKS. "
                        "Add it to the appropriate set in vlm_server.py.")

            # ── Legacy custom loader path ─────────────────────────────────────
            else:
                _, processor, model, _ = cached

                if task == "ocr":
                    if req.image_base64:
                        image = decode_image(req.image_base64)
                    elif req.prompt:
                        image = render_text_as_image(req.prompt)
                    else:
                        raise HTTPException(400, "OCR requires 'image_base64' or 'prompt'.")
                    px  = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
                    ids = model.generate(px)
                    out = processor.batch_decode(ids, skip_special_tokens=True)[0]

                elif task == "captioning":
                    if not req.image_base64:
                        raise HTTPException(400, "Captioning requires 'image_base64'.")
                    image  = decode_image(req.image_base64)
                    inputs = processor(image, req.prompt or "", return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    ids    = model.generate(**inputs, max_new_tokens=64)
                    out    = processor.decode(ids[0], skip_special_tokens=True)

                elif task == "vqa":
                    if not req.image_base64 or not req.prompt:
                        raise HTTPException(400, "VQA requires 'image_base64' and 'prompt'.")
                    image  = decode_image(req.image_base64)
                    inputs = processor(image, req.prompt, return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    logits = model(**inputs).logits
                    out    = model.config.id2label[logits.argmax(-1).item()]

                elif task == "speech-to-text":
                    if not req.audio_base64:
                        raise HTTPException(400, "speech-to-text requires 'audio_base64'.")
                    audio_array, sample_rate = decode_audio(req.audio_base64)
                    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    ids    = model.generate(**inputs)
                    out    = processor.batch_decode(ids, skip_special_tokens=True)[0]

                elif task == "audio-classification":
                    if not req.audio_base64:
                        raise HTTPException(400, "audio-classification requires 'audio_base64'.")
                    audio_array, sample_rate = decode_audio(req.audio_base64)
                    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    logits = model(**inputs).logits
                    out    = model.config.id2label[logits.argmax(-1).item()]

                else:
                    raise HTTPException(400, f"No custom handler for task '{task}'.")

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Inference error [{req.model}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return InferenceResponse(model=req.model, task=task, output=out, device=str(DEVICE))
