# RSNA · YOLOv8 y MLP · Streamlit Cloud

Demostración académica de Felipe Lizama Núñez. Detecta regiones de opacidad anotada en radiografías de tórax mediante el modelo YOLOv8n entrenado en el proyecto, e incluye un MLP tabular con edad, sexo y proyección. No establece un diagnóstico médico.

## Qué incluye

- Página **Modelo, datos y resultados**, con manifiestos y curvas originales.
- Página **Probar el modelo**, con las 1.000 radiografías originales del test interno (225 positivas, 775 negativas), umbral ajustable y carga PNG/JPG/DCM/DICOM.
- Pesos `best.pt`, MLP, escalador y columnas originales, sin volver a entrenar.
- Cinco archivos ZIP de test menores de 25 MB; la app los abre automáticamente en almacenamiento temporal. No los descomprimas ni los elimines del repositorio.
- Ejecución en CPU y bloqueo de inferencias concurrentes para compartir el detector entre visitantes.

## Publicar sin usar Colab

1. Crea un repositorio GitHub para este proyecto; nombre sugerido: `rsna-streamlit`.
2. Sube **el contenido de esta carpeta**, manteniendo sus subcarpetas. `app.py` y `requirements.txt` deben estar en la raíz del repositorio. No subas el ZIP exterior como único archivo.
3. Abre <https://share.streamlit.io/> e inicia sesión. Completa los pasos de creación de cuenta o vinculación de GitHub si el servicio los solicita.
4. Elige **Create app** y usa el repositorio, la rama `main` y el archivo `app.py`.
5. En **Advanced settings**, selecciona **Python 3.12**. No se necesitan secretos de Kaggle ni credenciales de Drive.
6. Elige el subdominio disponible y pulsa **Deploy**.
7. Para permitir visitas con el enlace, abre **Settings → Sharing** y configura la app como pública. El repositorio puede ser privado si tu cuenta de Streamlit tiene acceso a él.
8. Verifica la portada, luego **Probar el modelo → Analizar Caso de Test**. Cambia el umbral y repite la predicción; comprueba también una imagen subida. Copia la URL real solo cuando el despliegue funcione.

La URL se mantiene, aunque Community Cloud puede suspender la app tras 12 horas sin tráfico. Un visitante puede reactivarla. El servicio tiene recursos limitados; las pruebas locales no garantizan latencia ni disponibilidad en el alojamiento gratuito.

Documentación oficial:
- <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>
- <https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app>
- <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>

## Ejecutar en un PC (opcional)

Con Python 3.12, desde esta carpeta:

```bash
python -m venv .venv
```

Activa el entorno según tu sistema. En Windows PowerShell: `.venv\Scripts\Activate.ps1`. En Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Validación y límites conservados

Se comprobaron localmente la carga del detector y MLP, inferencias de casos de test positivo/negativo, dos solicitudes concurrentes, umbrales 0,25 y 0,70, decodificación PNG/JPG, DICOM sintéticos MONOCHROME1/MONOCHROME2, rechazo multiframe y ejecución de ambas páginas mediante Streamlit AppTest. El reporte está en `VALIDACION.json`. No se ejecutó entrenamiento ni evaluación completa nueva.

**DICOM:** el respaldo no contiene `preprocess.py`. Se mantiene la conversión que traía la app y su aviso de compatibilidad pendiente. El notebook preparó sus JPEG con `pixel_array`, redimensionado a 640×640 y guardado mediante OpenCV. La conversión DICOM de la app aplica LUT y percentiles; no debe asumirse equivalencia con el entrenamiento. La prueba sintética verifica el flujo de archivos, no valida radiografías clínicas externas ni todos los formatos de compresión.

**Métricas:** `results.csv` corresponde a entrenamiento/validación. El respaldo no aporta un archivo de evaluación independiente de test que la app pueda mostrar; se conserva ese aviso y no se inventan métricas.

**Subidas:** usa datos de prueba anonimizados. Se procesan en memoria del servidor sin que este código guarde las imágenes subidas en disco. Límite: 32 MB por archivo, hasta 16.777.216 píxeles y un solo cuadro monocromático en DICOM.

**MLP:** solo se aceptan las categorías M/F y AP/PA que corresponden a las columnas entrenadas; no se codifican categorías desconocidas como si fueran conocidas. La combinación 0,6/0,4 que aparece en test es experimental y no está calibrada clínicamente.

## Archivos

| Ruta | Contenido |
|---|---|
| `app.py` | Dos páginas Streamlit adaptadas desde el respaldo |
| `cloud_runtime.py` | Carga en CPU, bloqueo y recuperación del test |
| `requirements.txt` | Versiones de las dependencias usadas en las pruebas |
| `.streamlit/config.toml` | Configuración del servidor y límite de subida |
| `selected_yolo_run.txt` | Corrida seleccionada original |
| `runs/yolo_rsna/` | Mejor detector, resultados y figuras originales |
| `inference_cpu/` | MLP y preprocesamiento tabular originales |
| `data/` | Manifiestos y metadatos de test |
| `test_assets/` | Las 1.000 imágenes/etiquetas empaquetadas y sus hashes |
| `DATA_CREDITS.md` | Fuente y atribución de los datos |

Los checkpoints de cada época, `last.pt` y las imágenes de entrenamiento no son necesarios para esta app y permanecen en el respaldo original.

## Dependencias y atribución

Los datos y anotaciones conservan las condiciones de sus proveedores; consulta `DATA_CREDITS.md`. El detector usa Ultralytics, cuyo proyecto identifica AGPL-3.0 como licencia de su distribución de código abierto: <https://github.com/ultralytics/ultralytics>. Este archivo no cambia la licencia ni la titularidad de los materiales originales del usuario.
