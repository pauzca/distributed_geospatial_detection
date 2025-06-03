from ultralytics import YOLO, RTDETR
import time
import pandas as pd
import csv
from DetectionGeospatial import DetectionGeospatial
from Box import Box
import json
import argparse
import yaml
import os
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


"""
Detección de frailejones con modelos Yolov11 o DETR
Las configuraciones deberian estar en un archivo .yaml que contenga:

weights: ruta a pesos del modelo
modelo: nombre del modelo "YOLO" o "RTDETR"
folder: ruta donde se encuentra el dataset de detección
output: ruta donde se guardaran los archivos de salida
"""

def main(config):
    # this should be inputs to the file
    print(f"Model weights: {config['weights']}")
    print(f"Model name: {config['model']}")
    print(f"Files directory: {config['folder']}")
    print(f"Output folder: {config['output_folder']}")

    model_weights = config['weights']
    test_directory = config['folder'] + "/dataset/images"
    metadata_file = config['folder'] + "/tile_metadata.json"
    output_file = config['output_folder'] + "/yolov11_19_dic_1cm_pedazo_all_predictions.csv"
    model_name = config['model']

    ground_truth_file = config['folder'] + "/ground_truth.csv"

    detectionGeo = DetectionGeospatial(metadata_file)

    times = []  # Lista para almacenar los tiempos


    # load the model 
    if model_name == "YOLO":
        model = YOLO(model_weights)
    elif model_name == "RTDETR":
        model = RTDETR(model_weights)
    else:
        raise ValueError("Invalid model name. Choose 'YOLO' or 'RTDETR'.")

    # Check if the output file already exists
    if os.path.exists(output_file):
        print("taking predictions from file ")
    else:
        # predicting
        print("predicting")
        s = time.time()
        results = model.predict(source=test_directory, imgsz=512)

        #### resultados reduced y metrics reduced
        # convert predictions in pixels to geospatial bboxes
        bboxes_geo = []
        for r in results:
            bbox_file = r.path.split("images/")[1]
            geo_tile = detectionGeo.tiles_geo[detectionGeo.tiles_geo["tile"] == bbox_file].iloc[0]
            # check of it contains bbox
            if r.boxes.data.nelement() == 0:
                continue

            boxes = r.to_df().apply(lambda b: Box(b["class"],
                                                geo_tile, detectionGeo.geo_mosaic,
                                                b["box"]["x1"], b["box"]["y1"], b["box"]["x2"], b["box"]["y2"],
                                                b["confidence"]), axis =1)      
            #resize and convert to geospatial
            #map box 
            bgeo = [b.resize_to_original() for b in boxes]
            bgeo = [b.convert_geospatial() for b in bgeo]

            # append to bboxes_geo  
            bboxes_geo.extend(bgeo)

        # save the boxes in a file
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['id', 'xmin', 'ymin', 'xmax', 'ymax','confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for bbox in bboxes_geo:
                writer.writerow({
                    'id': bbox.label,
                    'xmin': bbox.xmin,
                    'ymin': bbox.ymin,
                    'xmax': bbox.xmax,
                    'ymax': bbox.ymax,
                    'confidence': bbox.confidence
                })

        e = time.time()
        print(f"Prediction time {e-s} seconds")
        inference_time = e - s
        times.append(f"Inference time: {inference_time:.2f} seconds")

    bounding_boxes = pd.read_csv(output_file)
    # run non maximum suppresion
    s = time.time()
    bounding_boxes_t = detectionGeo.tiled_nms(bounding_boxes)
    print("NMS time elapsed: ", e-s)

    # save boxes with non maximum suppresion
    save_path = output_file.replace(".csv", "_truncated.csv")
    bounding_boxes_t.to_csv(save_path, index=False)
    e = time.time()
    nms_time = e - s
    times.append(f"NMS time: {nms_time:.2f} seconds")


    bounding_boxes_t = pd.read_csv(save_path)
    bounding_boxes_gt = pd.read_csv(ground_truth_file)
    
    # computar metricas
    s = time.time()
    gt_labels, pred_labels = detectionGeo.tiled_compute_labels(bounding_boxes_gt, bounding_boxes_t)

    # save a confusion matrix
    # Compute the confusion matrix
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

    presicion, recall, f1_score = detectionGeo.compute_metrics(gt_labels, pred_labels)
    print(f"Precision: {presicion}, Recall: {recall}, F1 Score: {f1_score}")

    # save metrics
    metrics_file = output_file.replace(".csv", "_metrics.json")
    metrics = {
        "precision": presicion,
        "recall": recall,
        "f1_score": f1_score
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    e = time.time()
    metrics_time = e - s
    times.append(f"Metrics computation time: {metrics_time:.2f} seconds")
    return times

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/config_19_dic_test_inferencia_geo.yaml', help='config.yaml')
    args = parser.parse_args()

    with open(args.config,  'r') as f:
        config = yaml.safe_load(f)

    s = time.time()
    times = main(config)
    e = time.time()

    times.append(f"Total time: {e-s:.2f} seconds")

    # Guardar los tiempos en times.txt
    with open(config['output_folder'] + "/times.txt", "w") as f:
        for line in times:
            f.write(line + "\n")


