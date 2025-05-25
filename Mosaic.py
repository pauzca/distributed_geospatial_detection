import pandas as pd
import numpy as np
from osgeo import gdal
import os
from PIL import Image
import json
from BoxGeo import BoxGeo

class Mosaic:


    def __init__(self, input_tiff = None):

        if input_tiff:
            self.dataset = gdal.Open(input_tiff)

            if self.dataset is None:
                print(f"Error opening file {input_tiff}")
                exit(1)

            self.image_width = self.dataset.RasterXSize
            self.image_height = self.dataset.RasterYSize

            self.geotransform = self.dataset.GetGeoTransform()
            self.gds_x, self.gds_y = self.get_pixel_cm()

    

    def get_pixel_cm(self):
        # Approximate conversion factor: 1 degree = 111,320 meters at the equator
        # Convert pixel size from degrees to centimeters
        pixel_size_x_cm = abs(self.geotransform[1]) * 111320 * 100
        pixel_size_y_cm = abs(self.geotransform[5]) * 111320 * 100

        return pixel_size_x_cm, pixel_size_y_cm

    def calculate_tile_size_px(self, tile_width_cm):
        tile_width = int(tile_width_cm / self.gds_x)
        tile_height = int(tile_width_cm / self.gds_y)
        return tile_width, tile_height


    def generate_tiles(self, tile_width_cm, overlap_percentage, blank_threshold, accepted_labels, bboxes_path, output_dir, output_tile_width):

        tile_width, tile_height = self.calculate_tile_size_px(tile_width_cm)
        print( tile_width, tile_height)

        overlap_pixels = int(tile_width * overlap_percentage) # overlap in pixels
        print(overlap_pixels)

        if bboxes_path:
            bounding_boxes = pd.read_csv(bboxes_path)
            bounding_boxes = bounding_boxes.apply(lambda b: BoxGeo(0,
                b["xmin"], b["ymax"], b["xmax"],   b["ymin"]), axis =1)  
        else:
            labels_per_tile = [0]

        # assign id to bounding boxes


        # Calculate the step size (stride) for the loop
        x_stride = tile_width - overlap_pixels
        y_stride = tile_height - overlap_pixels

        almost_empty_counter = 0
        tile_with_no_labels_counter = 0
        labels_per_tile = []
        tile_count = 0
        # Calculate the size of the dataset in hectares
        pixel_area_cm2 = self.gds_x * self.gds_y  # Area of one pixel in cm²
        pixel_area_m2 = pixel_area_cm2 / 10000  # Convert cm² to m²
        pixel_area_ha = pixel_area_m2 / 10000  # Convert m² to hectares

        total_area_ha = self.image_width * self.image_height * pixel_area_ha


        os.makedirs(output_dir + "/dataset/images/test", exist_ok=True)
        os.makedirs(output_dir + "/dataset/labels/test", exist_ok=True)
        all_metadata = {"tiles": [],
                        "mosaic": {
                            "width": self.image_width,
                            "height": self.image_height,
                            "geotransform": self.geotransform,
                            "gds_x": self.gds_x,
                            "gds_y": self.gds_y,
                            "pixel_width": self.geotransform[1],
                            "pixel_height": self.geotransform[5],
                            "output_tile_width": output_tile_width,
                            "output_tile_height": output_tile_width,
                            "overlap_percentage": overlap_percentage,
                            "overlap_pixels": overlap_pixels,
                            "tile_size_cm": tile_width_cm
                        } }
        
        bbox_counter = 0

        # Loop over the image, generating tiles
        for x_offset in range(0, self.image_width, x_stride):
            for y_offset in range(0, self.image_height, y_stride):
                # Calculate width and height for the current tile (handle edge tiles)
                w = min(tile_width, self.image_width - x_offset)
                h = min(tile_height, self.image_height - y_offset)

                # Read the tile data using GDAL
                tile = self.dataset.ReadAsArray(x_offset, y_offset, w, h)
                tile = np.transpose(tile, (1, 2, 0))  # Rearrange to (height, width, bands)
                #print(tile.shape)

                # Check if the tile is completely white (255, 255, 255 for RGB)
                if np.all(tile == 255) or np.all(tile == 0):
                    #print(f"Skipping tile at ({x_offset}, {y_offset}) - completely empty")
                    continue  # Skip this tile
                
                # Check if the tile is mostly blank (90% or more)
                blank_pixels = np.sum(tile == 255) + np.sum(tile == 0)
                total_pixels = tile.size
                if blank_pixels / total_pixels >= blank_threshold:
                    tile_image = Image.fromarray(tile)
                    almost_empty_counter += 1
                    continue  # Skip this tile

                # Convert the tile to a PNG format using PIL
                tile_image = Image.fromarray(tile)

                if tile_image.mode != "RGB":
                    tile_image = tile_image.convert("RGB")
                
                # resize tile image to fit the output pixel size for the model
                tile_resized = tile_image.resize((output_tile_width, output_tile_width))

                # Define the output paths
                output_tile = os.path.join(output_dir, f"dataset/images/test/tile_{x_offset}_{y_offset}.png")
                
                # Save the tile as a PNG image
                tile_resized.save(output_tile)
                print(f"Saved tile: {output_tile} ({w}x{h} pixels)")


                # Calculate the geospatial coordinates of the top-left corner of the tile
                # rembember the output tile width and height is just a resize, the coordinates are still the same
                x_min_tile = self.geotransform[0] + x_offset * self.geotransform[1]
                y_min_tile = self.geotransform[3] + y_offset * self.geotransform[5]
                x_max_tile = x_min_tile + w * self.geotransform[1]
                y_max_tile = y_min_tile + h * self.geotransform[5]

                # Store the geospatial metadata for this tile
                # This will help convert the bboxes for the tile back to geo format to display them un qgis
                metadata = {
                    "tile": f"tile_{x_offset}_{y_offset}.png",  # Name of the tile
                    "xmin": x_min_tile,
                    "ymin": y_min_tile,
                    "xmax": x_max_tile,
                    "ymax": y_max_tile,
                    "tile_width": w,
                    "tile_height": h,
                    "x_offset": x_offset,
                    "y_offset": y_offset
                }

                # Append this tile's metadata to the list
                all_metadata["tiles"].append(metadata)

                # Searching for bounding boxes inside the tile
                
                tile_annotations = []
                if bboxes_path:
                    for box in bounding_boxes:                        
                        tile_coords = BoxGeo(None, x_min_tile, y_max_tile, x_max_tile, y_min_tile, None)
                        # Check if the bbox is inside the tile and if it is the label we want
                        if box.is_bbox_in_tile(tile_coords, min_area_ratio=1):
                            b_pixel = box.geospatial_to_pixel_coords(self.geotransform, tile_width, tile_height, x_offset, y_offset)
                            # Resize bounding boxes to fit the tile with the output size
                            bbox_resized  = b_pixel.resize_bounding_box(w, h, output_tile_width, output_tile_width)
                            # Append to tile_annotations
                            tile_annotations.append(bbox_resized)
                            bbox_counter = bbox_counter + 1


                    if len(tile_annotations) == 0:
                        tile_with_no_labels_counter += 1
                    else:
                        labels_per_tile.append(len(tile_annotations))

                    #"""
                    # Save tile annotations as txt file if any bounding boxes are present
                    if tile_annotations:
                        annotations_file = os.path.join(output_dir, f"dataset/labels/test/tile_{x_offset}_{y_offset}.txt")
                        with open(annotations_file, 'w') as f:
                            for annotation in tile_annotations:
                                label, x_center, y_center, bbox_width, bbox_height = annotation.bbox_to_yolo(output_tile_width, output_tile_width)
                                f.write(f"{label} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                        print(f"Saved {len(tile_annotations)} annotations in YOLO format: {annotations_file}")
                else:
                    tile_with_no_labels_counter += 1 


        # Save all the metadata to a single JSON file
        metadata_file = os.path.join(output_dir, "tile_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(all_metadata, f, indent=4)
            print(f"Metadata saved to {metadata_file}")


        total_tiles = len(labels_per_tile) + tile_with_no_labels_counter

        print(f"Area en hectareas: {total_area_ha}")
        print(f"Tiles discarded because they were almost all blank: {almost_empty_counter}")

        if labels_per_tile:
            print(f"Average boxes per tile in tiles that have at least one box: {np.mean(labels_per_tile)}")
            print(f"Maximum boxes in a tile: {np.max(labels_per_tile)}")
            print(f"Minimum boxes in a tile: {np.min(labels_per_tile)}")
        else:
            print("No tiles with boxes found.")

        print(f"Tiles saved that have boxes in them: {len(labels_per_tile)}  ({(len(labels_per_tile)/total_tiles)*100:.2f}%)")
        print(f"Saved tiles that have no labels on them: {tile_with_no_labels_counter}  ({(tile_with_no_labels_counter/total_tiles)*100:.2f}%)")
        print(f"Total of saved tiles: {total_tiles}")
        print(f"Total of saved bboxes: {bbox_counter}")

        # Save the statistics to a text file
        stats_output_path = os.path.join(output_dir, "tile_statistics.txt")
        with open(stats_output_path, 'w') as stats_file:
            stats_file.write(f"Area en hectareas: {total_area_ha}\n")
            stats_file.write(f"Tiles discarded because they were almost all blank: {almost_empty_counter}\n")
            
            if labels_per_tile:
                stats_file.write(f"Average boxes per tile in tiles that have at least one box: {np.mean(labels_per_tile)}\n")
                stats_file.write(f"Maximum boxes in a tile: {np.max(labels_per_tile)}\n")
                stats_file.write(f"Minimum boxes in a tile: {np.min(labels_per_tile)}\n")
            else:
                stats_file.write("No tiles with boxes found.\n")

            stats_file.write(f"Tiles saved that have boxes in them: {len(labels_per_tile)} ({(len(labels_per_tile)/total_tiles)*100:.2f}%)\n")
            stats_file.write(f"Saved tiles that have no labels on them: {tile_with_no_labels_counter} ({(tile_with_no_labels_counter/total_tiles)*100:.2f}%)\n")
            stats_file.write(f"Total of saved tiles: {total_tiles}\n")
            stats_file.write(f"Total of saved bboxes: {bbox_counter}\n")

        print(f"Statistics saved to {stats_output_path}")



    # create segmentation dataset
    def generate_tiles_segmentation(self, metadata_file, bboxes_path, output_dir):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # iterate each image 
        # convert predictions in pixels to geospatial bboxes
        boxes = pd.read_csv(bboxes_path).apply(lambda b: BoxGeo(b["id"],b["xmin"], b["ymax"], b["xmax"], b["ymin"], None), axis = 1)
        count = 0
        for b in boxes:
            count += 1
            b.setId(count)
            

        tiles_geo = pd.DataFrame(metadata["tiles"])
        mosaic = pd.DataFrame(metadata["mosaic"])
        print(mosaic.geotransform)

        saved_boxes_ids = []
        bbox_counter = 0
        for _, tile in tiles_geo.iterrows():
            tile_annotations = []
            tile_coords = BoxGeo(None, tile["xmin"], tile["ymax"], tile["xmax"], tile["ymin"], None)

            for b in boxes:
                # Check if the bbox is inside the tile and if it is the label we want
                if b.is_bbox_in_tile(tile_coords, min_area_ratio=1) and (b.id not in saved_boxes_ids):
                    b_pixel = b.geospatial_to_pixel_coords(mosaic["geotransform"], tile["tile_width"], tile["tile_height"], tile["x_offset"], tile["y_offset"])
                    b_resized = b_pixel.resize_bounding_box_segmentation(tile["tile_width"], tile["tile_height"], mosaic.iloc[0]["output_tile_width"], mosaic.iloc[0]["output_tile_height"])
                    # Append to tile_annotations
                    tile_annotations.append(b_resized)
                    bbox_counter = bbox_counter + 1
                    saved_boxes_ids.append(b.id)

            if tile_annotations:
                print(tile_annotations)
                annotations_file = os.path.join(output_dir, tile.tile.replace(".png", ".txt"))
                with open(annotations_file, 'w') as f:
                    for annotation in tile_annotations:
                        label, x_center, y_center, bbox_width, bbox_height = annotation.bbox_to_yolo(mosaic.iloc[0]["output_tile_width"], mosaic.iloc[0]["output_tile_width"])
                        f.write(f"{annotation.id} {label} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                print(f"Saved {len(tile_annotations)} annotations in YOLO format: {annotations_file}")

        print(f"saved {bbox_counter} boxes")
