from pyspark.sql import SparkSession
from pyspark import SparkFiles
from ultralytics import YOLO
from spark_job.utils import load_image, load_tile_metadata
from spark_job.Box import Box
import os

os.environ["YOLO_CONFIG_DIR"] = "/tmp"

tile_metadata = load_tile_metadata()

def predict_partition(image_paths):
    model_path = SparkFiles.get("yolo_best_fine_tune_800.pt")
    model = YOLO(model_path)
    results = []

    for path in image_paths:
        image_name = os.path.basename(path)
        tile_info = next((tile for tile in tile_metadata["tiles"] if tile["tile"] == image_name), None)
        if tile_info is None:
            continue

        preds = model.predict(source=path, imgsz=512, verbose=False)

        if preds[0].boxes.data.nelement() == 0:
            continue

        for _, b in preds[0].to_df().iterrows():
            box = Box(
                b["class"],
                tile_info,
                tile_metadata["mosaic"],
                b["box"]["x1"], b["box"]["y1"], b["box"]["x2"], b["box"]["y2"],
                b["confidence"]
            )
            resized = box.resize_to_original().convert_geospatial()
            results.append((
                image_name,
                resized.label,
                resized.xmin,
                resized.ymin,
                resized.xmax,
                resized.ymax,
                resized.confidence
            ))


    return iter(results)

if __name__ == "__main__":
    spark = SparkSession.builder.appName("DistributedDetection").getOrCreate()
    sc = spark.sparkContext

    rdd = sc.textFile("spark_scripts/data/input_images.txt")
    detections = rdd.mapPartitions(predict_partition)
    df = detections.toDF(["tile", "label", "xmin", "ymin", "xmax", "ymax", "confidence"])
    df.coalesce(1).write.csv("spark_scripts/data/predictions.csv", header=True, mode="overwrite")