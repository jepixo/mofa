# MOFA

Neutral identity face reconstruction scaffold that accepts 1-3 images and outputs a neutralized 3D head mesh with UV texture and export artifacts.

## Features
- Procedural base head mesh generation
- UV layout and preview
- Nine-stage pipeline orchestration
- GLB/FBX placeholder exporters plus identity JSON
- CLI for end-to-end execution

## Run tests
```bash
python -m pytest tests -v
```

## CLI
```bash
python cli.py --input front.jpg --output ./output
```
