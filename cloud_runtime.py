"""Resources shared by the Streamlit pages; CPU only, no Colab or Drive access."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import zipfile

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir())/'rsna_yolo_config'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import streamlit as st
import torch
from ultralytics import YOLO


@st.cache_resource(show_spinner=False)
def inference_lock():
    return threading.RLock()


@st.cache_resource(show_spinner='Cargando detector entrenado…')
def load_detector(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Faltan los pesos entrenados: {path.name}')
    torch.set_num_threads(2)
    model = YOLO(str(path)).to('cpu')
    model.model.eval()
    return model


def predict_cpu(model, source, confidence):
    # Streamlit serves different sessions in different threads.
    with inference_lock(), torch.inference_mode():
        return model.predict(
            source, conf=float(confidence), device='cpu', imgsz=640,
            verbose=False, save=False, save_txt=False,
            save_conf=False, save_crop=False,
        )[0]


@st.cache_resource(show_spinner='Preparando las 1.000 imágenes de test…')
def _extract_test_assets(project_dir):
    folder = Path(project_dir)/'test_assets'
    checksums = json.loads((folder/'checksums.json').read_text())
    holder = tempfile.TemporaryDirectory(prefix='rsna_test_')
    root = Path(holder.name).resolve()
    for archive in sorted(folder.glob('test_*.zip')):
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                destination = (root/entry.filename).resolve()
                if not destination.is_relative_to(root) or entry.filename not in checksums:
                    raise ValueError('Archivo inesperado en los recursos de test.')
                content = z.read(entry)
                if hashlib.sha256(content).hexdigest() != checksums[entry.filename]:
                    raise ValueError('Recurso de test incompleto o dañado.')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    if len(list((root/'images/test').glob('*.jpg'))) != 1000:
        raise ValueError('El paquete debe contener las 1.000 imágenes de test.')
    if any(not (root/name).is_file() for name in checksums):
        raise ValueError('Faltan imágenes o etiquetas de test.')
    return holder


def prepare_test_assets(project_dir):
    return Path(_extract_test_assets(str(project_dir)).name)
