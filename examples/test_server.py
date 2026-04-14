"""
examples/test_server.py
========================
Quick smoke-test script for the VLM server.
Run after starting the server with: uvicorn vlm_server:app --port 8000

Usage:
    python3 examples/test_server.py
    python3 examples/test_server.py --host http://localhost:8000 --model blip-base
"""

import argparse
import base64
import json
import urllib.request
from pathlib import Path


def get(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def make_test_image() -> str:
    """Create a white image with printed text. Returns base64 string."""
    from PIL import Image, ImageDraw, ImageFont
    img  = Image.new("RGB", (500, 80), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 20), "Security Test 12345", fill="black", font=font)

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="http://localhost:8000")
    parser.add_argument("--model", default="trocr-base-printed")
    args = parser.parse_args()

    host  = args.host.rstrip("/")
    model = args.model

    print(f"\n{'='*50}")
    print(f"  VLM Server Smoke Test")
    print(f"  Host : {host}")
    print(f"  Model: {model}")
    print(f"{'='*50}\n")

    # Health check
    print("1. Health check ...")
    resp = get(f"{host}/health")
    print(f"   status={resp['status']}  device={resp['device']}\n")

    # List models
    print("2. Registered models ...")
    models = get(f"{host}/models")
    for name, info in models.items():
        loaded = "loaded" if info["loaded"] else "not loaded"
        print(f"   [{loaded}] {name} — task={info['task']}  {info['description']}")
    print()

    # Inference via text prompt (OCR only)
    print("3. Inference — text prompt (OCR smoke test) ...")
    resp = post(f"{host}/inference", {"model": model, "prompt": "Hello VLM Server"})
    print(f"   output : {resp['output']}")
    print(f"   device : {resp['device']}\n")

    # Inference via real image
    print("4. Inference — image input ...")
    b64  = make_test_image()
    resp = post(f"{host}/inference", {"model": model, "image_base64": b64})
    print(f"   output : {resp['output']}")
    print(f"   device : {resp['device']}\n")

    print("All tests passed.")


if __name__ == "__main__":
    main()
