from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image

from mofa.config import MofaConfig
from mofa.runner import MofaPipeline


def main():
    parser = argparse.ArgumentParser(description="MOFA neutral identity face reconstruction")
    parser.add_argument("--input", required=True)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--format", choices=["glb", "fbx"], default="glb")
    args = parser.parse_args()

    paths = [args.input] + [p for p in [args.left, args.right] if p]
    images = [Image.open(p) for p in paths]
    cfg = MofaConfig(texture_size=args.texture_size, device=args.device, output_format=args.format)
    pipeline = MofaPipeline(cfg)
    result = pipeline.run(images, output_dir=args.output)

    print("MOFA pipeline timing breakdown")
    for k, v in result.timings.items():
        print(f"  {k}: {v*1000:.2f} ms")
    print(f"Output written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
