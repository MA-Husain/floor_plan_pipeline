#!/usr/bin/env python
"""Train the damage classifier: logistic-regression probe on CLIP ViT-L/14 image features.

Data: BD3 building-defect dataset (CC-BY-4.0), HF mirror chandrabhuma/building_defect_vqa:
7 classes (algae, major_crack, minor_crack, peeling, plain, spalling, stain), official
train / test split. The test split is never used for fitting; its scores go to defect_probe.json.
usage: python scripts/train_defect_probe.py        (downloads ~0.8 GB into data/bd3 once)
"""
import glob
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data' / 'bd3'
REPO = 'https://huggingface.co/datasets/chandrabhuma/building_defect_vqa/resolve/main/data/'
FILES = ['test-00000-of-00001', 'train-00000-of-00002', 'train-00001-of-00002']


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        p = DATA / f'{f}.parquet'
        if not p.exists():
            subprocess.run(['curl', '-sL', '-o', str(p), REPO + f + '.parquet'], check=True)


def features(clf):
    cache = DATA / 'clip_l14.npz'
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d['Xtr'], d['ytr'], d['Xte'], d['yte']
    import pandas as pd
    from PIL import Image
    out = []
    for split in ('train', 'test'):
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / f'{split}-*.parquet')))])
        ims = [Image.open(io.BytesIO(d['bytes'])).convert('RGB') for d in df['image']]
        out += [clf.embed(ims), df['answer'].values]
    np.savez(cache, Xtr=out[0], ytr=out[1], Xte=out[2], yte=out[3])
    return out


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from brynx.damage import DefectClassifier, PROBE
    fetch()
    clf = DefectClassifier.__new__(DefectClassifier)       # embedder only (no probe yet)
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from brynx.damage import CLIP_ID, HF_CACHE
    clf.torch = torch
    clf.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    clf.model = CLIPModel.from_pretrained(CLIP_ID, cache_dir=HF_CACHE).to(clf.device).eval()
    clf.proc = CLIPProcessor.from_pretrained(CLIP_ID, cache_dir=HF_CACHE)
    Xtr, ytr, Xte, yte = features(clf)
    lr = LogisticRegression(C=10, max_iter=5000).fit(Xtr, ytr)
    p = lr.predict(Xte)
    classes = list(lr.classes_)
    yb, pb = yte != 'plain', p != 'plain'
    report = {'dataset': 'BD3 (chandrabhuma/building_defect_vqa), CC-BY-4.0', 'n_train': int(len(ytr)), 'n_test': int(len(yte)),
              'accuracy_7class': round(float((p == yte).mean()), 4),
              'defect_vs_plain_accuracy': round(float((yb == pb).mean()), 4),
              'plain_flagged_as_defect': round(float((pb & ~yb).sum() / (~yb).sum()), 4),
              'defect_missed': round(float((~pb & yb).sum() / yb.sum()), 4),
              'classes': classes, 'confusion_matrix(rows=true)': confusion_matrix(yte, p, labels=classes).tolist()}
    np.savez(PROBE, coef=lr.coef_.astype(np.float32), intercept=lr.intercept_.astype(np.float32), classes=np.array(classes))
    PROBE.with_suffix('.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if 'matrix' not in k}, indent=1))


if __name__ == '__main__':
    main()
