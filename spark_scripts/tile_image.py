from Mosaic import Mosaic
import json

if __name__ == "__main__":
    config = {
        "input_tiff": "mosaicos/19_dic_2024_bajito_mask_test.tif",
        "output_dir": "spark_scripts/data",
        "bboxes_path": None,
        "output_tile_width": 512,
        "output_tile_height": 512,
        "tile_width_cm": 800,
        "tile_height_cm": 800,
        "overlap_percentage": 0.2,
        "blank_threshold": 0.85
    }

    mosaic = Mosaic(config["input_tiff"])
    mosaic.generate_tiles(
        config["tile_width_cm"],
        config["overlap_percentage"],
        config["blank_threshold"],
        accepted_labels=[1],
        bboxes_path=config["bboxes_path"],
        output_dir=config["output_dir"],
        output_tile_width=config["output_tile_width"]
    )

    # Crear input_images.txt
    with open("spark_scripts/data/tile_metadata.json") as f:
        metadata = json.load(f)

    with open("spark_scripts/data/input_images.txt", "w") as out:
        for tile in metadata["tiles"]:
            out.write(f"spark_scripts/data/dataset/images/test/{tile['tile']}\n")