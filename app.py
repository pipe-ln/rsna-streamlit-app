import streamlit as st
import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_runtime import load_detector, predict_cpu, prepare_test_assets
st.set_page_config(page_title='Sistema de Soporte RSNA V7', layout='wide')
PROJECT_DIR = str(Path(__file__).resolve().parent)
CPU_DIR = f'{PROJECT_DIR}/inference_cpu'
TEST_METADATA_PATH = f'{PROJECT_DIR}/data/clinical_test.csv'
IMAGES_TEST_DIR = f'{PROJECT_DIR}/data/yolo_dataset/images/test'
LABELS_TEST_DIR = f'{PROJECT_DIR}/data/yolo_dataset/labels/test'
MANIF_TRAIN = f'{PROJECT_DIR}/data/patients_train.csv'
MANIF_VAL = f'{PROJECT_DIR}/data/patients_val.csv'
MANIF_TEST = f'{PROJECT_DIR}/data/patients_test.csv'

def _selected_yolo_run():
    marker = Path(PROJECT_DIR) / 'selected_yolo_run.txt'
    if not marker.is_file():
        raise FileNotFoundError('Falta selected_yolo_run.txt en el paquete publicado. Restaura este archivo desde el respaldo.')
    run = (Path(PROJECT_DIR) / marker.read_text(encoding='utf-8').strip()).resolve()
    if run.parent != (Path(PROJECT_DIR) / 'runs').resolve():
        raise ValueError(f'Ruta de corrida inválida: {run}')
    return run

class ClinicalMLP(nn.Module):

    def __init__(self, input_dim):
        super(ClinicalMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

@st.cache_resource
def load_multimodal_resources():
    with open(f'{CPU_DIR}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{CPU_DIR}/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    mlp_model = ClinicalMLP(len(feature_cols))
    mlp_model.load_state_dict(torch.load(f'{CPU_DIR}/clinical_mlp_weights.pt', map_location=torch.device('cpu'), weights_only=True))
    mlp_model.eval()
    yolo_weight_path = str(_selected_yolo_run() / 'weights/best.pt')
    if os.path.exists(yolo_weight_path):
        yolo_model = load_detector(yolo_weight_path)
    else:
        raise FileNotFoundError('Falta best.pt entrenado en RSNA; no se carga YOLO genérico para inferencia.')
    return (scaler, feature_cols, mlp_model, yolo_model)
try:
    scaler, feature_cols, mlp_model, yolo_model = load_multimodal_resources()
    resources_loaded = True
except Exception as e:
    resources_loaded = False
    st.error(f'Error cargando los modelos o recursos: {e}')
st.sidebar.title('PREDICCION DE OPACIDADES EN RX TORAX CON YOLOv8')
st.sidebar.caption('Demostración académica · Felipe Lizama Núñez · Inferencia en CPU')
st.sidebar.markdown('[Datos RSNA / NIH y atribución](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)')
st.markdown('<style>\n.block-container {max-width:1180px;padding:2rem 1.5rem;}\nh1 {font-size:2rem !important;} h2 {font-size:1.45rem !important;}\nh3 {font-size:1.15rem !important;}\n[data-testid="stMetricValue"] {font-size:1.7rem !important;}\n</style>', unsafe_allow_html=True)

def _preview_image(image):
    preview = image.copy() if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    preview.thumbnail((420, 420), Image.Resampling.LANCZOS)
    return preview

def _model_page():
    import threading
    import inspect

    @st.cache_resource
    def _rsna_plot_lock():
        return threading.RLock()

    def _rsna_show_fig(fig):
        try:
            if 'width' in inspect.signature(st.pyplot).parameters:
                st.pyplot(fig, width='stretch')
            else:
                st.pyplot(fig, use_container_width=True)
        finally:
            plt.close(fig)

    def _rsna_number(series):
        cleaned = series.astype('string').str.strip().str.replace(',', '.', regex=False)
        return pd.to_numeric(cleaned, errors='coerce').replace([np.inf, -np.inf], np.nan)
    st.title('Cómo funciona nuestro modelo')
    st.caption('Un recorrido por los datos, el aprendizaje y los resultados observados.')
    intro_tab, data_tab, model_tab, result_tab = st.tabs(['Empezar aquí', 'Datos', 'Modelo', 'Resultados'])
    with intro_tab:
        st.subheader('¿Qué intenta hacer?')
        st.write('Recibe una radiografía de tórax y propone recuadros sobre zonas que se parecen a las opacidades anotadas en sus ejemplos de entrenamiento. No determina por sí solo un diagnóstico de neumonía.')
        stage = st.radio('Explora el proceso', ['1. Ejemplos', '2. Aprendizaje', '3. Predicción', '4. Evaluación'], horizontal=True)
        explanations = {'1. Ejemplos': ('Aprender de imágenes anotadas', 'Las imágenes positivas tienen recuadros de referencia. Las negativas no tienen esas cajas; algunas pueden mostrar otras alteraciones. Train enseña al modelo; validación ayuda a elegirlo; test se reserva para evaluarlo.'), '2. Aprendizaje': ('Ajustar millones de parámetros', 'YOLO compara sus propuestas con las anotaciones y ajusta sus parámetros. Una época es una pasada por el conjunto de entrenamiento. Las pérdidas ayudan a seguir ese ajuste, pero no son un porcentaje de acierto.'), '3. Predicción': ('Proponer regiones, no diagnósticos', 'Cada caja tiene una puntuación del detector. Por ejemplo, 0,8 no significa automáticamente 80 % de probabilidad de neumonía. El umbral elegido determina qué propuestas se muestran.'), '4. Evaluación': ('Comparar contra una referencia reservada', 'Hay que evaluar tanto si detecta las regiones anotadas como si las localiza correctamente. Los errores y la separación de train, validación y test importan tanto como una cifra alta.')}
        heading, explanation = explanations[stage]
        with st.container(border=True):
            st.subheader(heading)
            st.write(explanation)
        metric = st.selectbox('Entender una métrica', ['Precisión', 'Recall', 'mAP', 'Pérdida', 'Matriz de confusión'])
        glossary = {'Precisión': 'De las cajas propuestas, ¿qué proporción coincide con una anotación según el criterio de solapamiento? Disminuye con falsas detecciones.', 'Recall': 'De las cajas de referencia, ¿qué proporción logra detectar? Disminuye cuando omite opacidades anotadas.', 'mAP': 'Resume la detección a distintos umbrales. mAP50 exige IoU de 0,50; mAP50-95 promedia varios requisitos de solapamiento, más exigentes. No es exactitud diagnóstica.', 'Pérdida': 'Cantidad que se intenta reducir al entrenar. Se compara a lo largo de las épocas y entre train/validación; no se interpreta como porcentaje de errores clínicos.', 'Matriz de confusión': 'Cuenta aciertos y errores. Una matriz por imagen y una matriz de detección por cajas no representan lo mismo; hay que indicar su origen y el conjunto evaluado.'}
        st.info(glossary[metric])
        st.write('Abre las pestañas Datos, Modelo y Resultados para ver la evidencia real. Usa Probar el modelo en el menú lateral para comparar una imagen con sus predicciones.')
    with data_tab:
        with st.expander('Fuentes y atribución de los datos'):
            st.markdown((Path(PROJECT_DIR) / 'DATA_CREDITS.md').read_text(encoding='utf-8'))
        st.subheader('1. Distribución de los datos')
        manifest_paths = {'Train': MANIF_TRAIN, 'Validación': MANIF_VAL, 'Test': MANIF_TEST}
        frames = {}
        metric_cols = st.columns(3)
        for column, (label, path) in zip(metric_cols, manifest_paths.items()):
            try:
                frame = pd.read_csv(path)
                frame.columns = frame.columns.str.strip()
                if 'patientId' not in frame:
                    raise ValueError('Falta patientId')
                if frame.patientId.isna().any():
                    raise ValueError('Existen IDs vacíos')
                if 'Target' in frame:
                    frame['Target'] = _rsna_number(frame.Target)
                    if not frame.Target.isin([0, 1]).all():
                        raise ValueError('Target debe ser 0 o 1')
                    if frame.groupby('patientId').Target.nunique().gt(1).any():
                        raise ValueError('Etiquetas contradictorias para un ID')
                frame = frame.drop_duplicates('patientId')
                frames[label] = frame
                column.metric(label, f'{len(frame):,}'.replace(',', '.'))
            except Exception as exc:
                column.metric(label, '—')
                st.warning(f'{label}: {exc}. Archivo: {path}')
        st.caption('Conteos de IDs de imagen únicos. No se presupone independencia de pacientes de origen.')
        valid = {name: df.Target.value_counts().reindex([0, 1], fill_value=0) for name, df in frames.items() if 'Target' in df}
        if valid:
            distribution = pd.DataFrame(valid).T
            distribution.columns = ['Sin opacidad anotada', 'Con opacidad anotada']
            with _rsna_plot_lock(), plt.style.context('default'):
                fig, ax = plt.subplots(figsize=(9, 3.3), layout='constrained')
                distribution.plot.bar(ax=ax, color=['#64748b', '#0891b2'], rot=0)
                ax.set_ylabel('Número de imágenes')
                ax.set_xlabel('')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(axis='y', alpha=0.2)
                _rsna_show_fig(fig)
            with st.expander('Ver conteos y proporciones'):
                st.dataframe(distribution)
                st.dataframe(distribution.div(distribution.sum(axis=1), axis=0).round(3))
        else:
            st.warning('No hay etiquetas Target válidas para graficar la distribución; no se infieren a partir de los tamaños.')
    with model_tab:
        st.subheader('2. Arquitectura de los modelos')
        weights = _selected_yolo_run() / 'weights/best.pt'

        @st.cache_resource
        def _rsna_load_dashboard_yolo(path, modified):
            return load_detector(path)
        if weights.is_file():
            try:
                detector = _rsna_load_dashboard_yolo(str(weights), weights.stat().st_mtime_ns)
                n_params = sum((p.numel() for p in detector.model.parameters()))
                a, b = st.columns(2)
                a.metric('Parámetros YOLO', f'{n_params:,}'.replace(',', '.'))
                b.metric('Clases del detector', len(detector.names))
                st.write('Backbone: extracción de características; neck: combinación a distintas escalas; head: cajas y puntuaciones por clase.')
                with st.expander('Ver arquitectura completa de YOLO', expanded=False):
                    st.code(str(detector.model), language='text')
                    st.write('Clases del archivo cargado:', detector.names)
                st.caption(f'Pesos: {weights}')
            except Exception as exc:
                st.error(f'No se pudo inspeccionar el detector: {exc}')
        else:
            st.warning(f'No se encontró el detector entrenado: {weights}. No se sustituye por pesos COCO.')
        if globals().get('resources_loaded', False):
            with st.expander('Ver arquitectura del MLP', expanded=False):
                st.code(str(mlp_model), language='text')
                st.write('Variables de entrada:', list(feature_cols))
            st.caption('La arquitectura mostrada corresponde al MLP cargado. El código actual combina puntuaciones con pesos fijos 0,6/0,4; esto no demuestra una fusión entrenada ni calibración clínica.')
        else:
            st.info('MLP no disponible. Su ausencia no impide visualizar las curvas del detector.')
    with result_tab:
        st.subheader('3. Entrenamiento y validación')
        run_dir = _selected_yolo_run()
        results_file = run_dir / 'results.csv'
        if results_file.is_file():
            try:
                raw = pd.read_csv(results_file)
                raw.columns = raw.columns.str.strip().str.lstrip('\ufeff')
                if len(raw.columns) == 1:
                    raise ValueError('CSV leído como una columna. Revisa su separador antes de graficar.')
                history = raw.apply(_rsna_number)
                history = history.dropna(how='all')
                if history.empty:
                    raise ValueError('No hay filas numéricas válidas en results.csv.')
                if 'epoch' in history and history.epoch.notna().any():
                    history = history.loc[history.epoch.notna()].sort_values('epoch')
                    x = history.epoch
                    xlabel = 'Época registrada'
                else:
                    x = pd.Series(np.arange(1, len(history) + 1), index=history.index)
                    xlabel = 'Fila del historial (no se encontró epoch)'
                st.caption(f'{len(history)} registros · {results_file}')
                plot_mask = pd.Series(True, index=history.index)
                if len(history) > 1 and int(x.min()) < int(x.max()):
                    lo, hi = st.slider('Explorar intervalo de épocas', min_value=int(x.min()), max_value=int(x.max()), value=(int(x.min()), int(x.max())))
                    plot_mask = x.between(lo, hi)
                if len(history) == 1:
                    st.info('Solo hay una época: se dibujarán puntos, no una curva de varias épocas.')
                groups = [('Pérdida de localización', ['train/box_loss', 'val/box_loss'], 'Pérdida'), ('Pérdidas de clasificación y DFL', ['train/cls_loss', 'val/cls_loss', 'train/dfl_loss', 'val/dfl_loss'], 'Pérdida'), ('Precisión y recall de validación', ['metrics/precision(B)', 'metrics/recall(B)'], 'Valor'), ('mAP de validación', ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], 'Valor')]
                for offset in (0, 2):
                    slots = st.columns(2)
                    for slot, (title, keys, ylabel) in zip(slots, groups[offset:offset + 2]):
                        with slot:
                            present = [key for key in keys if key in history and history[key].notna().any()]
                            if not present:
                                st.warning(f'{title}: faltan columnas o valores numéricos válidos.')
                                continue
                            with _rsna_plot_lock(), plt.style.context('default'):
                                fig, ax = plt.subplots(figsize=(5.3, 3.2), layout='constrained')
                                for key in present:
                                    ax.plot(x.loc[plot_mask], history.loc[plot_mask, key], marker='o', markersize=3, linewidth=1.5, label=key)
                                ax.set_title(title, fontsize=11)
                                ax.set_xlabel(xlabel)
                                ax.set_ylabel(ylabel)
                                ax.grid(alpha=0.2)
                                ax.legend(fontsize=7, loc='best')
                                if ylabel == 'Valor':
                                    ax.set_ylim(-0.03, 1.03)
                                _rsna_show_fig(fig)
                summary_keys = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
                last = history.iloc[-1]
                observed = {key: float(last[key]) for key in summary_keys if key in last and pd.notna(last[key])}
                if observed:
                    st.caption('Valores de validación de la última fila; no son métricas finales del test ni necesariamente de best.pt.')
                    slots = st.columns(len(observed))
                    for slot, (key, value) in zip(slots, observed.items()):
                        slot.metric(key.removeprefix('metrics/').replace('(B)', ''), f'{value:.3f}')
                with st.expander('Diagnóstico del CSV: valores originales, tipos y faltantes'):
                    st.dataframe(raw.tail(10))
                    st.dataframe(pd.DataFrame({'tipo original': raw.dtypes.astype(str), 'valores numéricos': history.notna().sum()}))
                    st.write('Columnas detectadas:', raw.columns.tolist())
                st.write('Las pérdidas resumen el ajuste durante el entrenamiento; su escala no equivale a exactitud. Precisión/recall y mAP describen rendimiento en validación. Revisa la tendencia completa; una sola época o un valor aislado no demuestra convergencia ni sobreajuste.')
            except Exception as exc:
                st.error(f'No se pudieron graficar datos válidos: {exc}')
        else:
            st.error(f'Falta {results_file}. Restaura el CSV de esta corrida desde Drive; no hace falta reentrenar si ya está respaldado.')
        with st.expander('Parámetros registrados en args.yaml'):
            args_path = run_dir / 'args.yaml'
            if args_path.is_file():
                st.code(args_path.read_text(), language='yaml')
            else:
                st.info('No se encuentra args.yaml en esta corrida.')
        st.subheader('4. Figuras guardadas de la corrida')
        st.caption('Figuras del directorio de entrenamiento/validación. No se etiquetan como test.')
        figures = [p for name in ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png', 'F1_curve.png', 'BoxPR_curve.png', 'BoxF1_curve.png'] if (p := (run_dir / name)).is_file()]
        if figures:
            for start in range(0, len(figures), 2):
                for slot, path in zip(st.columns(2), figures[start:start + 2]):
                    with slot:
                        st.image(str(path), caption=path.name, width=500)
        else:
            st.info('No se encontraron figuras guardadas en esta corrida. Las curvas anteriores se construyen directamente desde el CSV.')
        st.subheader('5. Evaluación independiente de test')
        st.info('Se retiró la matriz escrita a mano. Para mostrar métricas de test se necesita el archivo de evaluación real y su procedencia (split, pesos y umbral). results.csv del entrenamiento no sustituye esa evaluación.')
import io
import importlib.util

def _rsna_dicom_fallback(source):
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    ds = pydicom.dcmread(source)
    if int(ds.get('Rows', 0)) * int(ds.get('Columns', 0)) > 16_777_216:
        raise ValueError('La imagen supera el límite de 16,7 millones de píxeles de esta demo.')
    if int(ds.get('NumberOfFrames', 1)) != 1 or int(ds.get('SamplesPerPixel', 1)) != 1:
        raise ValueError('Solo se admiten radiografías monocromáticas de un cuadro; no volúmenes ni multiframe.')
    photo = str(ds.get('PhotometricInterpretation', '')).upper()
    if photo not in ('MONOCHROME1', 'MONOCHROME2'):
        raise ValueError('Interpretación fotométrica no compatible: ' + photo)
    try:
        raw = ds.pixel_array
    except Exception as exc:
        raise ValueError('No se pudieron decodificar los píxeles. Si el DICOM está comprimido, puede necesitar pylibjpeg-libjpeg o pylibjpeg-openjpeg. Detalle: ' + str(exc)) from exc
    if raw.ndim != 2:
        raise ValueError('Se requiere una imagen de dos dimensiones.')
    valid = np.isfinite(raw)
    if 'PixelPaddingValue' in ds:
        p1 = float(ds.PixelPaddingValue)
        p2 = float(ds.get('PixelPaddingRangeLimit', p1))
        valid &= ~((raw >= min(p1, p2)) & (raw <= max(p1, p2)))
    values = np.asarray(apply_voi_lut(apply_modality_lut(raw, ds), ds), dtype=np.float64)
    valid &= np.isfinite(values)
    if not valid.any():
        raise ValueError('El DICOM no contiene píxeles válidos.')
    lo, hi = np.percentile(values[valid], [0.5, 99.5])
    if hi <= lo:
        lo, hi = (values[valid].min(), values[valid].max())
    if hi <= lo:
        raise ValueError('La imagen tiene intensidad constante; no se puede normalizar.')
    scaled = np.zeros_like(values)
    scaled[valid] = np.clip((values[valid] - lo) / (hi - lo), 0, 1)
    if photo == 'MONOCHROME1':
        scaled[valid] = 1 - scaled[valid]
    gray = np.round(scaled * 255).astype(np.uint8)
    age = None
    try:
        text = str(ds.get('PatientAge', '')).strip().upper()
        unit = text[-1]
        number = float(text[:-1] if unit in 'DWMY' else text)
        years = number * {'D': 1 / 365.25, 'W': 7 / 365.25, 'M': 1 / 12, 'Y': 1}.get(unit, 1)
        if 0 <= years <= 120:
            age = years
    except (ValueError, IndexError):
        pass
    sex = str(ds.get('PatientSex', '')).upper()
    metadata = {'age': age, 'sex': sex if sex in ('M', 'F', 'O') else 'UNKNOWN', 'view': str(ds.get('ViewPosition', 'UNKNOWN'))}
    return (Image.fromarray(gray).convert('RGB'), metadata)

def _rsna_read_upload(upload):
    content = upload.getvalue()
    if not content:
        raise ValueError('El archivo está vacío.')
    if len(content) > 32 * 1024 * 1024:
        raise ValueError('El archivo supera el límite de 32 MB.')
    suffix = Path(upload.name).suffix.lower()
    if suffix in ('.dcm', '.dicom'):
        module_path = Path(PROJECT_DIR) / 'preprocess.py'
        shared = False
        if module_path.is_file():
            spec = importlib.util.spec_from_file_location('_rsna_shared_preprocess', module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if callable(getattr(module, 'dicom_image', None)):
                image, metadata = module.dicom_image(io.BytesIO(content))
                shared = True
            else:
                raise ValueError('preprocess.py existe pero no contiene dicom_image. Hay que conectar su función real de preparación; no se sustituye silenciosamente.')
        else:
            image, metadata = _rsna_dicom_fallback(io.BytesIO(content))
        rgb = np.asarray(image.convert('RGB'))
        description = 'Módulo preprocess.py del proyecto: dicom_image.' if shared else 'Conversión incluida: Modality LUT/Rescale → VOI → percentiles 0,5–99,5 sin padding → inversión MONOCHROME1. No se ha confirmado que coincida con la conversión usada en tu entrenamiento.'
        return (np.ascontiguousarray(rgb[:, :, ::-1]), metadata, description, not shared)
    if suffix not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('Formato no admitido.')
    with Image.open(io.BytesIO(content)) as header:
        if header.width * header.height > 16_777_216:
            raise ValueError('La imagen supera el límite de 16,7 millones de píxeles de esta demo.')
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('El archivo no es una imagen PNG/JPG válida o está dañado.')
    return (image, {'age': None, 'sex': 'UNKNOWN', 'view': 'UNKNOWN'}, 'PNG/JPG leído como en la app anterior.', False)

@st.cache_resource
def _rsna_upload_detector(weight_path, modified):
    return load_detector(weight_path)

def _test_page():
    test_root = prepare_test_assets(PROJECT_DIR)
    IMAGES_TEST_DIR = str(test_root / 'images/test')
    LABELS_TEST_DIR = str(test_root / 'labels/test')
    st.title('Probar el modelo')
    umbral_confianza = st.slider('Umbral de confianza YOLO', min_value=0.01, max_value=0.99, value=0.25, step=0.01, key='rsna_conf_threshold')
    st.caption('Umbral exploratorio: después de cambiarlo vuelve a pulsar Analizar. Las métricas guardadas no cambian.')
    st.caption('Vista compacta: original a la izquierda y predicción a la derecha. El tamaño de visualización no cambia la imagen usada para inferencia.')
    st.info('El score de combinación 0,6/0,4 del código original no es una probabilidad clínica calibrada ni demuestra que la fusión haya sido entrenada.')
    tab1, tab2 = st.tabs(['🎯 Imágenes de Test', '📤 Subir radiografía'])
    with tab1:
        st.subheader('Prueba sobre el Split de Test Interno')
        if os.path.exists(MANIF_TEST):
            df_test = pd.read_csv(MANIF_TEST)
            patient_ids = df_test['patientId'].unique()
            filter_class = st.selectbox('Filtrar por Target:', ['Todos', 'Con Opacidad (Target=1)', 'Sin Opacidad (Target=0)'])
            if filter_class == 'Con Opacidad (Target=1)':
                filtered_pids = df_test[df_test['Target'] == 1]['patientId'].values
            elif filter_class == 'Sin Opacidad (Target=0)':
                filtered_pids = df_test[df_test['Target'] == 0]['patientId'].values
            else:
                filtered_pids = patient_ids
            selected_pid = st.selectbox('Seleccione ID de paciente para test:', filtered_pids)
            if selected_pid:
                img_path = next((str(p) for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG') if (p := (Path(IMAGES_TEST_DIR) / f'{selected_pid}{ext}')).is_file()), str(Path(IMAGES_TEST_DIR) / f'{selected_pid}.jpg'))
                label_path = os.path.join(LABELS_TEST_DIR, f'{selected_pid}.txt')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    _original_col, _prediction_col = st.columns(2, gap='medium')
                    _original_col.image(_preview_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), caption=f'Paciente {selected_pid} (Original)', width='content')
                    if st.button('Analizar Caso de Test') and resources_loaded:
                        clinical_test = pd.read_csv(TEST_METADATA_PATH)
                        matched = clinical_test.loc[clinical_test['patientId'].astype(str) == str(selected_pid)]
                        if len(matched) != 1:
                            raise ValueError(f'Falta un registro clínico único para {selected_pid}')
                        pat_info = matched.iloc[0]
                        age = float(pat_info['Age'])
                        sex = str(pat_info['Sex'])
                        position = str(pat_info['Position'])
                        row_data = {col: 0.0 for col in feature_cols}
                        row_data['Age'] = age
                        for col in feature_cols:
                            if col.startswith('Sex_'):
                                row_data[col] = float(sex == col[len('Sex_'):])
                            elif col.startswith('Position_'):
                                row_data[col] = float(position == col[len('Position_'):])
                        features_scaled = scaler.transform(pd.DataFrame([row_data], columns=feature_cols).astype(np.float32))
                        st.caption(f'Metadatos del caso: edad {age:g}, sexo {sex}, posición {position}')
                        with torch.no_grad():
                            mlp_prob = mlp_model(torch.tensor(features_scaled)).item()
                        yolo_res = predict_cpu(yolo_model, img_path, umbral_confianza)
                        boxes = yolo_res.boxes.xyxy.cpu().numpy()
                        confs = yolo_res.boxes.conf.cpu().numpy() if len(yolo_res.boxes.conf) > 0 else np.array([0.0])
                        max_y_conf = np.max(confs)
                        score = 0.6 * max_y_conf + 0.4 * mlp_prob
                        out_img = img.copy()
                        for b, c in zip(boxes, confs):
                            x1, y1, x2, y2 = map(int, b)
                            cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                            cv2.putText(out_img, f'Conf: {c:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        _prediction_col.image(_preview_image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)), caption='Resultado de Inferencia', width='content')
                        st.metric('Score combinado experimental', f'{score:.3f}')
                else:
                    st.error(f'Imagen de prueba no encontrada en local en la ruta {img_path}')
        else:
            st.warning('No se ha encontrado el manifiesto patients_test.csv')
    with tab2:
        st.subheader('Subir radiografía')
        st.caption('PNG, JPG, JPEG, DCM y DICOM. DICOM: una imagen monocromática; no volúmenes ni archivos multiframe.')
        st.caption('Usa radiografías de prueba anonimizadas. La imagen se envía al servidor para procesarla; esta app no la guarda en disco.')
        uploaded_file = st.file_uploader('Selecciona la radiografía', type=['png', 'jpg', 'jpeg', 'dcm', 'dicom'], key='rsna_upload_dicom')
        if uploaded_file is not None:
            try:
                img_decoded, dicom_meta, conversion, needs_review = _rsna_read_upload(uploaded_file)
            except Exception as exc:
                st.error(f'No se pudo leer la imagen: {exc}')
                return
            if needs_review:
                st.warning('El DICOM se pudo convertir. Antes de interpretar su predicción, verifica que este preprocesamiento coincida con el que generó las imágenes de entrenamiento.')
            with st.expander('Cómo se preparó la imagen'):
                st.write(conversion)
                st.write({'ancho': int(img_decoded.shape[1]), 'alto': int(img_decoded.shape[0]), 'proyección': dicom_meta.get('view', 'UNKNOWN')})
                st.caption('Solo se reduce la vista previa; YOLO recibe la imagen convertida a resolución original.')
            original_col, prediction_col = st.columns(2, gap='medium')
            original_col.image(_preview_image(cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)), caption='Imagen recibida', width='content')
            if st.button('Ejecutar Inferencia Manual', key='rsna_predict_upload'):
                weights = _selected_yolo_run() / 'weights/best.pt'
                if not weights.is_file():
                    st.error(f'Falta el detector entrenado: {weights}')
                    return
                try:
                    with st.spinner('Analizando en CPU...'):
                        detector = _rsna_upload_detector(str(weights), weights.stat().st_mtime_ns)
                        prediction = predict_cpu(detector, img_decoded, umbral_confianza)
                        overlay = prediction.plot()
                    prediction_col.image(_preview_image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), caption=f'Predicción · umbral {umbral_confianza:.2f}', width='content')
                    boxes = prediction.boxes.xyxy.cpu().numpy()
                    scores = prediction.boxes.conf.cpu().numpy()
                    st.metric('Regiones detectadas', len(boxes))
                    if len(boxes) == 0:
                        st.info('No hay detecciones por encima del umbral seleccionado. Esto no permite descartar enfermedad.')
                    else:
                        table = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
                        table['confianza'] = scores
                        st.dataframe(table.round(3), hide_index=True)
                        st.download_button('Descargar cajas CSV', table.to_csv(index=False).encode('utf-8'), file_name='detecciones.csv', mime='text/csv')
                    png = io.BytesIO()
                    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(png, format='PNG')
                    st.download_button('Descargar predicción PNG', png.getvalue(), file_name='prediccion.png', mime='image/png')
                    st.caption('Resultado del detector YOLO. La puntuación de una caja no es una probabilidad clínica de neumonía.')
                except Exception as exc:
                    st.error(f'No se pudo ejecutar el detector: {exc}')
            if globals().get('resources_loaded', False):
                with st.expander('Modelo tabular adicional — opcional'):
                    st.caption('Usa solo datos confirmados. Este modelo no genera cajas y no se combina automáticamente con el score de YOLO.')
                    age_default = dicom_meta.get('age')
                    age = st.number_input('Edad en años', min_value=0.0, max_value=120.0, value=float(age_default) if age_default is not None else None, key='rsna_upload_age')
                    sex_options = ['Sin dato', 'M', 'F', 'O', 'UNKNOWN']
                    sex_default = dicom_meta.get('sex')
                    sex = st.selectbox('Sexo registrado', sex_options, index=sex_options.index(sex_default) if sex_default in sex_options else 0, key='rsna_upload_sex')
                    view_options = ['Sin dato', 'AP', 'PA', 'LL', 'RL', 'LATERAL', 'UNKNOWN']
                    view_default = dicom_meta.get('view')
                    view = st.selectbox('Proyección radiográfica registrada', view_options, index=view_options.index(view_default) if view_default in view_options else 0, key='rsna_upload_view')
                    if st.button('Evaluar modelo tabular', key='rsna_predict_tabular'):
                        needs_view = any((col.startswith('Position_') for col in feature_cols))
                        if age is None or sex not in ('M', 'F') or (needs_view and view not in ('AP', 'PA')):
                            st.warning('El MLP se entrenó con sexo M/F y proyección AP/PA. Completa esos datos y la edad; otras categorías no se interpretan automáticamente.')
                        else:
                            try:
                                row = {col: 0.0 for col in feature_cols}
                                if 'Age' in row:
                                    row['Age'] = float(age)
                                if 'Sex_' + sex in row:
                                    row['Sex_' + sex] = 1.0
                                if 'Position_' + view in row:
                                    row['Position_' + view] = 1.0
                                frame = pd.DataFrame([row], columns=feature_cols).astype(np.float32)
                                transformed = scaler.transform(frame)
                                with torch.no_grad():
                                    score = mlp_model(torch.tensor(transformed, dtype=torch.float32)).item()
                                st.metric('Salida del MLP tabular', f'{score:.3f}')
                                st.caption('Salida del modelo; no se ha demostrado calibración clínica.')
                            except Exception as exc:
                                st.error(f'No se pudo ejecutar el MLP: {exc}')
st.navigation([st.Page(_model_page, title='Modelo, datos y resultados', icon='📊', default=True), st.Page(_test_page, title='Probar el modelo', icon='🔎')]).run()
