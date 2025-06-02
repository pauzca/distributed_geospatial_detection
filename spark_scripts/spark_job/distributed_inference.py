from pyspark.sql import SparkSession
from pyspark import SparkFiles
from ultralytics import YOLO
from spark_job.utils import load_image, load_tile_metadata
from spark_job.Box import Box
from spark_job.DetectionGeospatial import DetectionGeospatial
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

os.environ["YOLO_CONFIG_DIR"] = "/tmp"

tile_metadata = load_tile_metadata()
detectionGeo = DetectionGeospatial(tile_metadata)


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



def compute_labels_per_tile(tile_data):
    gt = tile_data["gt"]
    preds = tile_data["preds"]
    
    gt_labels, pred_labels = detectionGeo.compute_labels_fast(gt, preds, iou_threshold=0.5, conf_threshold=0.5)
    return (gt_labels, pred_labels)

def print_confusion_matrix(gt_labels, pred_labels, output_file):
    cm = confusion_matrix(gt_labels, pred_labels, labels=[1, 0])

    # Optional: print raw matrix values
    print("Confusion Matrix:")
    print(cm)

    # Display and save the confusion matrix as an image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Frailejon", "Background"])
    disp.plot(cmap=plt.cm.Blues)

    # Save the plot
    plt.title("Detection Confusion Matrix")
    cf_file = output_file.replace(".csv", "confusion_matrix.png")
    plt.savefig(cf_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("DistributedDetection").getOrCreate()
    sc = spark.sparkContext

    ########## mapper, inferencia de cajas delimitadoras ############

    rdd = sc.textFile("spark_scripts/data/input_images copy.txt")
    print("Doing inference...")
    s = time.time()
    detections = rdd.mapPartitions(predict_partition)
    e = time.time()
    print("Inference time elapsed: ", e-s)
    
    # reduce guardar todo en un archivo csv
    output_folder = "spark_scripts/data/predictions"
    df = detections.toDF(["tile", "id", "xmin", "ymin", "xmax", "ymax", "confidence"])
    df.coalesce(1).write.csv(output_folder, header=True, mode="overwrite")

    ########### Reduce #1 computar NMS #########################
    print("Computing NMS...")

    s = time.time()
    bounding_boxes_t = detectionGeo.tiled_nms(df.toPandas())
    e = time.time()
    print("NMS time elapsed: ", e-s)
    save_path = output_folder + "/bounding_boxes_truncated.csv"
    bounding_boxes_t.to_csv(save_path, index=False)


    ############## map #2 para computar metricas parciales #######################
    print("Computing metrics...")
    s = time.time()

    bounding_boxes_t = pd.read_csv(save_path)
    # vamos a repartir las predicciones en los workers que tenemos disponibles
    ground_truth_file = "spark_scripts/data/ground_truth.csv"
    df_gt = pd.read_csv(ground_truth_file)

    print(sc.getConf().get("spark.executor.instances"))
    num_workers = int(sc.getConf().get("spark.executor.instances", "1"))
    print(num_workers)

    #num_workers = 3

    tiles_in_area = detectionGeo.assing_tiles(df_gt, bounding_boxes_t, num_workers, overlap_percentage=0)

    rdd2 = sc.parallelize(tiles_in_area, num_workers)
    results = rdd2.map(compute_labels_per_tile).collect()

    ############# reduce, compute metrics #########################
    final_GT_LABELS = []
    final_PRED_LABELS = []

    for gt_labels, pred_labels in results:
        final_GT_LABELS.extend(gt_labels)
        final_PRED_LABELS.extend(pred_labels)

    # print confusion matrix
    output_file = output_folder + "/confusion_matrix.png"
    print_confusion_matrix(final_GT_LABELS, final_PRED_LABELS, output_file)

    precision, recall, f1_score  = detectionGeo.compute_metrics(final_GT_LABELS, final_PRED_LABELS)
    
    # save metrics
    metrics_file = output_folder + "/detection_metrics.json"
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    e = time.time()
    print("Metrics computation time elapsed: ", e-s)





    