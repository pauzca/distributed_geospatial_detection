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


"""
Detección de frailejones con modelos Yolov11 o DETR
Las configuraciones deberian estar en un archivo .yaml que contenga:

weights: ruta a pesos del modelo
modelo: nombre del modelo "YOLO" o "RTDETR"
folder: ruta donde se encuentra el dataset de detección
output: ruta donde se guardara un archivo csv con las predicciones georrefrenciadas al mosaico
"""

def main(config):
    # this should be inputs to the file
    print(f"Model weights: {config['weights']}")
    print(f"Model name: {config['model']}")
    print(f"Files directory: {config['folder']}")
    print(f"Output file: {config['output']}")

    model_weights = config['weights']
    test_directory = config['folder'] + "/dataset/images"
    metadata_file = config['folder'] + "/tile_metadata.json"
    output_file = config['output']
    model_name = config['model']

    detectionGeo = DetectionGeospatial(metadata_file)

    # add area 

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
        e = time.time()
        print(f"Prediction time {e-s} seconds")
        # sacar metricas 

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


    bounding_boxes = pd.read_csv(output_file)
    # run non maximum suppresion
    s = time.time()
    bounding_boxes_t = detectionGeo.tiled_nms(bounding_boxes)
    e = time.time()
    print("NMS time elapsed: ", e-s)

    # save boxes with non maximum suppresion
    save_path = output_file.replace(".csv", "_truncated.csv")
    bounding_boxes_t.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/config_19_dic_test_inferencia_geo.yaml', help='config.yaml')
    args = parser.parse_args()

    with open(args.config,  'r') as f:
        config = yaml.safe_load(f)

    s = time.time()
    main(config)
    e = time.time()

    print( f"{e-s}/60 Minutes in inference and nms")

