# 🛰️ Proyecto de Detección Geoespacial de Frailejones  
**Inferencia distribuida con PySpark + YOLOv11 + Spark YARN**

Este proyecto permite detectar frailejones en imágenes satelitales (mosaicos `.tif`), divididas en tiles, usando modelos de visión por computador de forma distribuida usando PySpark, tanto localmente como en un clúster Hadoop.

Aquí es posible ver más información sobre los datos: https://colab.research.google.com/drive/1Tf5RDTGdcj-VcZLKzN4YK7w2fZLf65x_?usp=sharing

---

## 📁 Estructura del Proyecto
```
ADGE_GH_FINAL/
├── distributed_geospatial_detection/ 
│   └── original_scripts # Código base original
│   └── datasets/ # tiles del mosaico
│      └── 19_dic_2024_bajito_mask_test/
│      ├── dataset/images/test/ # Tiles PNG (512x512)
│      └── tile_metadata.json # Metadatos geoespaciales
│   └── spark_scripts/
│       ├── data/
│       │ ├── input_images.txt # Lista que corre dentro del hadoop de la u
│       │ ├── input_images copy.txt # Lista de tiles a procesar *fuera* del hadoop de la u
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
├── mosaicos
├── env/ # Entorno virtual con dependencias instaladas
├── env.tar.gz # Entorno comprimido para usar con YARN
```

---

## ⚙️ Instalación del entorno virtual


# 1. Crear y activar entorno virtual

⚠ Este proyecto funciona con python 3.10 ⚠

```bash
python3.10 -m venv env
source env/bin/activate
```

# 2. Instalar dependencias
```bash
pip install -r requirements2.txt
```

# 3. Comprimir el entorno para usar con Spark YARN
```bash
deactivate
tar -czf spark_env.tar.gz spark_env
```

🧾 Generar lista de los tiles a procesar
```bash
python spark_scripts/data/generate_input_images_txt.py
```

🚀 Ejecutar el proyecto

🔹 Opción 1: Local (pruebas en PC) desde la ruta `ADGE_GH_FINAL\distributed_geospatial_detection`

```bash
bash spark_scripts/run_local.sh
```

Esto ejecuta el proceso en paralelo en tu máquina local usando todos los núcleos disponibles.


🔹 Opción 2: En clúster Hadoop (YARN)

```bash
bash spark_scripts/run_yarn.sh
```

Esto ejecuta el proceso en modo distribuido entre los nodos del clúster usando YARN y el entorno empaquetado.

Las predicciones deberían ser visibles en el hdfs
```bash
hdfs dfs -ls /user/hadoop/spark_scripts/data
```