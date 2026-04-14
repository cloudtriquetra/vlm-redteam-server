"""
examples/send_image.py
=======================
Send any image file to the VLM server and print the result.

Usage:
    python3 examples/send_image.py /path/to/image.png
    python3 examples/send_image.py /path/to/image.png --model blip-base --prompt "What is in this image?"
    python3 examples/send_image.py /path/to/image.png --host http://192.168.1.10:8000
"""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Send an image to the VLM server")
    parser.add_argument("image",            help="Path to image file")
    parser.add_argument("--host",   default="http://localhost:8000")
    parser.add_argument("--model",  default="trocr-base-printed")
    parser.add_argument("--prompt", default=None, help="Text prompt / question for VQA or captioning")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {"model": args.model, "image_base64": b64}
    if args.prompt:
        payload["prompt"] = args.prompt

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{args.host.rstrip('/')}/inference",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as r:
            result = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"Server error {e.code}: {e.read().decode()}")
        sys.exit(1)

    print(f"model  : {result['model']}")
    print(f"task   : {result['task']}")
    print(f"device : {result['device']}")
    print(f"output : {result['output']}")


if __name__ == "__main__":
    main()
