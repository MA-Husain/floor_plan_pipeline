#!/usr/bin/env python
"""One-time download of all model weights into ./weights (the pipeline then runs offline).

  YOLO-World v2 x  (ultralytics, AGPL-3.0)          fixtures / room typing     ~146 MB
  Grounding DINO tiny (IDEA-Research, Apache-2.0)    damage proposals           ~690 MB
  SAM 2.1 hiera-small (Meta, Apache-2.0)             damage masks               ~184 MB
"""
import os
from pathlib import Path

W = Path(__file__).resolve().parent.parent / 'weights'
W.mkdir(exist_ok=True)
os.environ.setdefault('HF_HOME', str(W / 'hf'))


def main():
    from ultralytics import YOLOWorld
    cwd = os.getcwd(); os.chdir(W)
    YOLOWorld('yolov8x-worldv2.pt')           # downloads into ./weights
    os.chdir(cwd)
    from huggingface_hub import snapshot_download
    for rid in ('IDEA-Research/grounding-dino-tiny', 'facebook/sam2.1-hiera-small'):
        snapshot_download(rid, cache_dir=str(W / 'hf'), allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
    print('weights ready in', W)


if __name__ == '__main__':
    main()
