#!/bin/bash
spark-submit \
  --master local[*] \
  --py-files spark_scripts/spark_job.zip \
  --files spark_scripts/models/yolo_best_fine_tune_800.pt \
  spark_scripts/spark_job/distributed_inference.py
