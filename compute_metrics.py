import argparse
import numpy as np
from osgeo import gdal
from Mosaic import Mosaic
from ultralytics import YOLO


def main(args):
    model = YOLO(args.weights)
    yaml = args.yaml
    # Validate with a custom dataset
    metrics = model.val(data=yaml, save_json=True)

    # this is what needs to be saved at the end
    print(metrics.results_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default='pesos/yolo_best_fine_tune_800.pt', help='path of weights')
    parser.add_argument('--yaml', type=str, default='config/19_dic_2024_test_metrics.yaml', help='path for ultralytics yaml file for metrics')

    args = parser.parse_args()

    main(args)

