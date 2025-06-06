# 🛰️ Proyecto de Detección Geoespacial de Frailejones  
**Inferencia distribuida con PySpark + YOLOv11 + Spark**

Este proyecto permite detectar frailejones en imágenes satelitales (mosaicos `.tif`), divididas en tiles, usando modelos de visión por computador de forma distribuida usando PySpark, tanto localmente como en un clúster Hadoop.

---

## 📁 Estructura del Proyecto
```
ADGE_GH_FINAL/
├── distributed_geospatial_detection/ 
│   └── original_scripts # Código base original y estudio del dataset
│   └── datasets/ # tiles del mosaico
│      └── 19_dic_2024_bajito_mask_test/
│      ├── dataset/images/test/ # Tiles PNG (512x512)
│      └── tile_metadata.json # Metadatos geoespaciales
│   └── spark_scripts/
│       ├── data/
│       │ ├── input_images.txt # Lista que corre dentro del hadoop de la u
│       │ ├── input_images copy.txt # Lista de tiles a procesar *fuera* del hadoop de la u
│       │ └── generate_input_images_txt.py # Script que genera input_images.txt
│       │ └── tile_metadata.json #metadatos
│       ├── models/
│       │ └── yolo_best_fine_tune_800.pt # Pesos del modelo YOLOv11
│       ├── spark_job/
│       │ ├── distributed_inference.py # Script principal de inferencia
│       │ └── utils.py # Funciones auxiliares
│       │ └── Box.py # clase para las cajas
│       │ └── DetectionGeospatial.py # Calculo de métricas y NMS
│       ├── run_local.sh # Ejecución local
│       └── run_yarn.sh # Ejecución en Hadoop YARN
│    └── requirements.txt # Requisitos Python (excepto GDAL)
|    └── README.md
├── mosaicos # Dataset, se descarga aparte
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
