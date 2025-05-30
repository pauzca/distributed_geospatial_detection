# 🛰️ Proyecto de Detección Geoespacial de Frailejones  
**Inferencia distribuida con PySpark + YOLOv5 + Spark YARN**

Este proyecto permite detectar frailejones en imágenes satelitales (mosaicos `.tif`), divididas en tiles, usando modelos de visión por computador (YOLOv5 o RTDETR), de forma distribuida usando PySpark, tanto localmente como en un clúster Hadoop.

---

## 📁 Estructura del Proyecto

ADGE_GH_FINAL/
├── datasets/
│ └── 19_dic_2024_bajito_mask_test/
│ ├── dataset/images/test/ # Tiles PNG (512x512)
│ └── tile_metadata.json # Metadatos geoespaciales
├── spark_env/ # Entorno virtual con dependencias instaladas
├── spark_env.tar.gz # Entorno comprimido para usar con YARN
├── distributed_geospatial_detection/ 
│   └── original_scripts # Código base original
│   └── spark_scripts/
│       ├── data/
│       │ ├── input_images.txt # Lista de tiles a procesar
│       │ └── generate_input_images_txt.py # Script que genera input_images.txt
│       ├── models/
│       │ └── yolo_best_fine_tune_800.pt # Pesos del modelo YOLOv5
│       ├── spark_job/
│       │ ├── distributed_inference.py # Script principal de inferencia
│       │ └── utils.py # Funciones auxiliares
│       ├── run_local.sh # Ejecución local
│       └── run_yarn.sh # Ejecución en Hadoop YARN
│    └── requirements.txt # Requisitos Python (excepto GDAL)
|    └── README.md


---

## ⚙️ Instalación del entorno virtual


# 1. Crear y activar entorno virtual
```bash
python3 -m venv spark_env
source spark_env/bin/activate
```

# 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

# 3. Comprimir el entorno para usar con Spark YARN
```bash
deactivate
tar -czf spark_env.tar.gz spark_env
```

🧾 Generar lista de imágenes a procesar
```bash
python spark_scripts/data/generate_input_images_txt.py
```

🚀 Ejecutar el proyecto
🔹 Opción 1: Local (pruebas en PC)
```bash
bash spark_scripts/run_local.sh
```

Esto ejecuta el proceso en paralelo en tu máquina local usando todos los núcleos disponibles.

🔹 Opción 2: En clúster Hadoop (YARN)
```bash
bash spark_scripts/run_yarn.sh
```

Esto ejecuta el proceso en modo distribuido entre los nodos del clúster usando YARN y el entorno empaquetado.

📦 Salida esperada
Se generará una carpeta tipo:

```bash
spark_scripts/data/predictions.csv/
├── part-00000...
```