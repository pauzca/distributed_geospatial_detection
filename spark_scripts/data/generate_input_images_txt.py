# spark_scripts/data/generate_input_images_txt.py
import os

image_dir = "datasets/19_dic_2024_bajito_mask_test/dataset/images/test"
output_txt = "distributed_geospatial_detection/spark_scripts/data/input_images.txt"

with open(output_txt, "w") as f:
    for img in sorted(os.listdir(image_dir)):
        if img.endswith(".png"):
            f.write(os.path.join(image_dir, img) + "\n")

print(f"Archivo generado: {output_txt}")
