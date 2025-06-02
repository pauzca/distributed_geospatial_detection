#!/bin/bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --archives env.tar.gz#env \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/bin/python \
  --conf spark.executorEnv.PYSPARK_PYTHON=./env/bin/python \
  --py-files spark_scripts/spark_job.zip \
  --files spark_scripts/data/input_images.txt,spark_scripts/data/tile_metadata.json,spark_scripts/models/yolo_best_fine_tune_800.pt \
  spark_scripts/spark_job/distributed_inference.py
