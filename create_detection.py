import argparse
import yaml
import json
import numpy as np
from osgeo import gdal
from Mosaic import Mosaic

"""
Codigo para crear dataset de segmentación a partir de un mosaico georreferenciado. 

Recibe el parametro 
--config (un archivo json): debe contener los parametros obligatorios:
[input_tiff]: ruta absoluta del archivo tiff del mosaico (renderizado)
[output_dir]: ruta donde se creará el dataset y se guardaran sus metadatos
[bboxes_path]: ruta donde se encuentran cajas de ground truth si las hay
[output_tile_width]: ancho de imagen de salida
[output_tile_height]: alto de imagen de salida

Parametros adicionales: 

[accepted_labels]: id de las cajas que deseo guardar. 
[blank_threshold]: porcentaje de la imagen que debe estar vacia o en blanco para descartarla
[overlap_percentage]: porcentaje de sobrelape entre teselas (la idea es que sea mayor a 2m que es el tamaño maximo de estas rosetas)
"""

REQUIRED_CONFIG = [
    'input_tiff',
    'output_dir',
    'bboxes_path',
    'output_tile_width',
    'output_tile_height'
]

# TODO add default values
OVERLAP_PERCENTAGE = 0.2
ACCEPTED_LABELS = "id"
ACCEPTED_LABELS_CLASS = "frailejones"
TILE_WIDTH_CM = 800
TILE_HEIGHT_CM = 800
LABEL_COLUMN = 1
BLANK_THRESHOLD = 0.85


def main(config_file):

    with open(config_file, 'r') as config_file:
        config = json.load(config_file)

        missing_keys = [key for key in REQUIRED_CONFIG if key not in config]
        if missing_keys:
            raise KeyError(f"Missing required config keys: {', '.join(missing_keys)}")

        input_tiff = config['input_tiff']
        output_dir = config['output_dir']
        bboxes_path = config['bboxes_path']
        output_tile_width = config['output_tile_width']
        output_tile_height = config['output_tile_height']

        accepted_labels = config.get('accepted_labels', ACCEPTED_LABELS)
        tile_width_cm = config.get('tile_width_cm', TILE_WIDTH_CM)
        tile_height_cm = config.get('tile_height_cm', TILE_HEIGHT_CM)
        overlap_percentage = config.get('overlap_percentage', OVERLAP_PERCENTAGE)
        blank_threshold = config.get('blank_threshold', BLANK_THRESHOLD)


    mosaic = Mosaic(input_tiff)

    # divide mosaic in tiles and save in output dir
    mosaic.generate_tiles(tile_width_cm, overlap_percentage, blank_threshold, accepted_labels, bboxes_path, output_dir, output_tile_width)


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/config_19_dic_2024_test_dataset.json', help='config.json')
    args = parser.parse_args()

    main(args.config)

