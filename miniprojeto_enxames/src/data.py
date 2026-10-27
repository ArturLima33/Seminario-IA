"""Satimage do PMLB; sem substituição silenciosa por bases artificiais."""
from __future__ import annotations
import hashlib
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

URL = ('https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/'
       '294_satellite_image/294_satellite_image.tsv.gz')


def load_data(path: str | None = None, *, smoke: bool = False):
    """Devolve (X,y,metadados); --smoke nunca se passa por Satimage."""
    if smoke:
        X, y = make_classification(n_samples=240, n_features=12, n_informative=9,
                                   n_redundant=1, n_classes=3, n_clusters_per_class=1,
                                   random_state=123)
        return X, y, {"source": "SYNTHETIC_SMOKE_NOT_FOR_REPORT", "sha256": None}
    file = Path(path) if path else Path('data/294_satellite_image.tsv.gz')
    if not file.exists():
        if path:
            raise FileNotFoundError(f'Base não encontrada em: {file}')
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix('.download')
        try:
            with urllib.request.urlopen(URL, timeout=40) as response, tmp.open('wb') as out:
                shutil.copyfileobj(response, out)
            tmp.replace(file)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError('Sem acesso à base real. Baixe o arquivo 294_satellite_image.tsv.gz '
                               'do PMLB/GitHub para data/ ou use --data-path. '
                               f'Causa: {exc}') from exc
    if file.suffix == '.gz' and file.open('rb').read(2) != b'\x1f\x8b':
        raise ValueError('Download não retornou gzip válido (possível ponteiro Git LFS). Baixe o arquivo real via link do PMLB e informe --data-path.')
    df = pd.read_csv(file, sep='\t')
    if 'target' not in df.columns:
        raise ValueError('Dataset incompatível: coluna target ausente')
    X = df.drop(columns=['target']).to_numpy(dtype=np.float64)
    y = df['target'].to_numpy()
    if X.shape != (6435, 36) or len(np.unique(y)) != 6:
        raise ValueError(f'Formato real inesperado {X.shape}; {len(np.unique(y))} classes')
    if not np.isfinite(X).all() or pd.isna(y).any():
        raise ValueError('Dataset contém dados nulos/infinitos')
    h = hashlib.sha256(file.read_bytes()).hexdigest()
    return X, y, {"source": str(file.resolve()), "sha256": h}


def split_data(X, y, test_fraction=0.2, seed=42):
    return train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=y)


def describe(X, y, meta):
    classes, counts = np.unique(y, return_counts=True)
    return {**meta, "samples": int(len(y)), "attributes": int(X.shape[1]),
            "n_classes": len(classes), "class_counts": {str(k): int(v) for k, v in zip(classes, counts)}}
