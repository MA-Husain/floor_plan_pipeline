#!/usr/bin/env python
"""One-time download of everything the pipeline needs into ./weights and ./third_party.
After this the pipeline runs fully offline (HF_HUB_OFFLINE=1 is set by run_capture.py).

  MapAnything (Meta, facebook/map-anything-apache, Apache-2.0)   video/photo tiers: camera poses          ~4.9 GB
    + its code (github facebookresearch/map-anything, Apache-2.0) and the DINOv2 hub code (Apache-2.0)
  MoGe-2 ViT-L (Microsoft, Ruicheng/moge-2-vitl-normal, MIT)     video/photo tiers: metric depth + focal  ~1.3 GB
    + its code (github microsoft/MoGe, MIT)
  CLIP ViT-L/14 (OpenAI, MIT)                                     damage classifier features               ~1.7 GB
  YOLO-World v2 x (Ultralytics, AGPL-3.0)                         fixtures for room typing                 ~146 MB
The damage probe itself (models/defect_probe.npz, 21 KB) is in the repo; to retrain it from the
BD3 dataset run scripts/train_defect_probe.py.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / 'weights'
W.mkdir(exist_ok=True)


def git_clone(url, dst):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '-q', '--depth', '1', url, str(dst)], check=True)


def main():
    # code
    git_clone('https://github.com/facebookresearch/map-anything.git', ROOT / 'third_party' / 'map-anything')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'uniception==0.1.7',
                    '-e', str(ROOT / 'third_party' / 'map-anything')], check=True)
    git_clone('https://github.com/facebookresearch/dinov2.git', W / 'torch' / 'hub' / 'facebookresearch_dinov2_main')
    git_clone('https://github.com/microsoft/MoGe.git', ROOT / 'third_party' / 'MoGe')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                    'git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37',
                    'git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4'], check=True)
    # weights
    from huggingface_hub import snapshot_download
    snapshot_download('facebook/map-anything-apache', cache_dir=str(W / 'hf'))
    snapshot_download('Ruicheng/moge-2-vitl-normal', cache_dir=str(W / 'hf'))
    snapshot_download('openai/clip-vit-large-patch14', cache_dir=str(W / 'hf'),
                      allow_patterns=['*.json', '*.safetensors', '*.txt'])
    from ultralytics import YOLOWorld
    cwd = os.getcwd(); os.chdir(W)
    YOLOWorld('yolov8x-worldv2.pt')
    os.chdir(cwd)
    print('weights ready in', W)


if __name__ == '__main__':
    main()
