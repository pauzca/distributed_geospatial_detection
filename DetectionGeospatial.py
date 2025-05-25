import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import math

class DetectionGeospatial:

    # creo que es mas legible el codigo si no hay ningun parametro para inicializacion
    # solo quiero usar las funciones at the end of the day
    # y es mas legible si le paso los parametros a las funciones ya en el codigo, asi se que es lo que necesitan
    # y exactamente que es lo que estan haciendo
    def __init__(self):
        pass

    def __init__(self, metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self.tiles_geo = pd.DataFrame(metadata["tiles"])

        self.geo_mosaic = metadata["mosaic"]
        self.gds_x = self.geo_mosaic['gds_x']
        self.gds_y = self.geo_mosaic['gds_y']

        self.image_width = self.geo_mosaic["width"]
        self.image_height = self.geo_mosaic["height"]

        self.geotransform = self.geo_mosaic['geotransform']


    def compute_area_box_pixels(self, box):
        pixel_width = float(self.geo_mosaic["pixel_width"])
        pixel_height = float(self.geo_mosaic["pixel_height"])

        width_pixels = abs((box["xmax"] - box["xmin"]) / pixel_width)
        height_pixels = abs((box["ymax"] - box["ymin"]) / pixel_height)

        area_pixels = width_pixels * height_pixels
        return area_pixels
        


    # Check if bbox is inside of tile
    def is_bbox_in_tile(self, tile_cords, bbox_cords, min_area_ratio=0.3):
        # Calculate intersection rectangle

        x_min_tile, y_min_tile, x_max_tile, y_max_tile = tile_cords
        x_min_bbox, y_min_bbox, x_max_bbox, y_max_bbox = bbox_cords

        inter_x_min = max(x_min_tile, x_min_bbox)
        inter_y_min = max(y_min_tile, y_min_bbox)
        inter_x_max = min(x_max_tile, x_max_bbox)
        inter_y_max = min(y_max_tile, y_max_bbox)

        # Check if intersection is valid
        inter_width = inter_x_max - inter_x_min
        inter_height = inter_y_max - inter_y_min
        if inter_width > 0 or inter_height > 0: # if there is an intersection
            # Calculate intersection area
            inter_area = inter_width * inter_height
            bbox_area = (x_max_bbox - x_min_bbox) * (y_max_bbox - y_min_bbox)
            if inter_area / bbox_area >= min_area_ratio:
                return True
        return False

    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box1, box2: dictionaries with keys ['xmin', 'ymin', 'xmax', 'ymax']
    """
    def calculate_iou(self, box1, box2):
        # Calculate intersection coordinates
        x1 = max(box1['xmin'], box2['xmin'])
        y1 = min(box1['ymin'], box2['ymin']) # since we are above the equator latitudes are positive
        x2 = min(box1['xmax'], box2['xmax'])
        y2 = max(box1['ymax'], box2['ymax'])

        # Compute intersection area
        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y1 - y2) # because y2 is the southernmost, is less than y1

        
        intersection_area = intersection_width * intersection_height

        # Compute areas of both boxes
        area_box1 = abs((box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin']))
        area_box2 = abs((box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin']))

        # Compute union area
        union_area = area_box1 + area_box2 - intersection_area

        # Return IoU
        if union_area == 0:
            return 0
        return intersection_area / union_area


    # Check if box width and area ratio is not a square
    def box_is_normal(self, bbox, area_threshold ):
        width = abs(bbox['xmax'] - bbox['xmin'])
        height = abs(bbox['ymin'] - bbox['ymax'])

        if (width / height ) < area_threshold or (height / width) < area_threshold:
            return False
        else:
            return True
        
    # Non maximum suppresion!
    def custom_non_maximum_suppresion(self, bboxes_todo, indexes_in_area, iou_threshold=0.2, area_threshold = 0.55):
        try:
            bboxes = bboxes_todo.loc[indexes_in_area]
            if bboxes_todo.empty:
                    print("No bounding boxes to process.")
                    return bboxes_todo
        except Exception as e:
            # If there's an error (e.g., invalid indexes), return the original bboxes
            print(f"Error copying bounding boxes: {e}")
            return bboxes_todo
        
        bboxes.loc[:, 'area'] = abs((bboxes['xmax'] - bboxes['xmin']) * (bboxes['ymax'] - bboxes['ymin']))
        keep = []

        for i, box1 in bboxes.iterrows():
            #print(i)
            redundant = False
            for j in keep:
                box2 = bboxes.loc[j]

                # Calculate IoU
                iou = self.calculate_iou(box1, box2)

                #print(f"bbox1 {box1['id']} bbox2 {box2['id']} -> iou = {iou}")

                if iou > iou_threshold:
                    redundant = True

                    #print(f"bbox1 [{box1['id']}] area = {box1['area']} \nbbox2 [{box2['id']}] area = {box2['area']} \n cambiar {box1['area'] > box2['area']}")

                    if box1['area'] > box2['area']:
                        keep.remove(j)
                        keep.append(i)

            if not redundant and self.box_is_normal(box1, area_threshold):
                #print(f"Adding {box1['id']} not redundant")
                keep.append(i)

        updated_bboxes = bboxes.loc[keep].drop(columns=['area'])
        bboxes_todo = pd.concat([bboxes_todo.drop(indexes_in_area), updated_bboxes]).reset_index(drop=True)
        return bboxes_todo


    # do non maximum suppresion by parts
    def tiled_nms(self, bounding_boxes, tile_width_cm = 1000, overlap_percentage = 0.25):

        tile_width, tile_height = self.calculate_tile_size_px(tile_width_cm, tile_width_cm)
        print( tile_width, tile_height)

        overlap_pixels = int(tile_width * overlap_percentage) # overlap in pixels
        print(overlap_pixels)

        # Calculate the step size (stride) for the loop
        x_stride = tile_width - overlap_pixels
        y_stride = tile_height - overlap_pixels

        tile_count = 0

        tiles_x = math.ceil((self.image_width - tile_width) / x_stride) + 1
        tiles_y = math.ceil((self.image_height - tile_height) / y_stride) + 1

        total_tiles = tiles_x * tiles_y

        print(f"Expected number of tiles: {total_tiles} ({tiles_x} x {tiles_y})")

        # Loop over the image, generating tiles
        for x_offset in range(0, self.image_width, x_stride):
            for y_offset in range(0, self.image_height, y_stride):
                tile_count += 1
                print(f"{tile_count}/{total_tiles}")
                # Calculate width and height for the current tile (handle edge tiles)
                w = min(tile_width, self.image_width - x_offset)
                h = min(tile_height, self.image_height - y_offset)

                # Calculate the geospatial coordinates of the top-left corner of the tile
                # rembember the output tile width and height is just a resize, the coordinates are still the same
                x_min_tile = self.geotransform[0] + x_offset * self.geotransform[1]
                y_min_tile = self.geotransform[3] + y_offset * self.geotransform[5]
                x_max_tile = x_min_tile + w * self.geotransform[1]
                y_max_tile = y_min_tile + h * self.geotransform[5]

                
                # Searching for bounding boxes inside the tile
                index_bboxes_in_area = []
                for index, row in bounding_boxes.iterrows():
                    label  = int(row["id"])
                    x_min_bbox = row["xmin"]
                    x_max_bbox = row["xmax"]
                    y_min_bbox = row["ymax"]
                    y_max_bbox = row["ymin"]

                    tile_coords = (x_min_tile, y_max_tile, x_max_tile, y_min_tile)
                    bbox_coords = (x_min_bbox, y_min_bbox, x_max_bbox, y_max_bbox)

                    # Check if the bbox is inside the tile and if it is the label we want
                    if self.is_bbox_in_tile(tile_coords, bbox_coords, min_area_ratio=0.5):
                        index_bboxes_in_area.append(index)
                
                # now maximum supression on the bboxes in this tile and update bboxes
                bounding_boxes = self.custom_non_maximum_suppresion(bounding_boxes, index_bboxes_in_area)

        return bounding_boxes



    def compute_labels(self, gt_df, pred_df, iou_threshold=0.5, conf_threshold=0.0):
        filtered_preds = pred_df[pred_df['confidence'] >= conf_threshold]
        predictions = filtered_preds.sort_values(by='confidence', ascending=False)
    

        matched_gt = set()
        pred_labels = []
        gt_labels = []

        for _, pred_box in predictions.iterrows():
            match_found = False
            for gt_idx, gt_box in gt_df.iterrows():
                if gt_idx in matched_gt:
                    continue
                iou = self.calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    matched_gt.add(gt_idx)
                    pred_labels.append(1)  # Prediction is a TP
                    gt_labels.append(1)    # Corresponds to real object
                    match_found = True
                    break
            if not match_found:
                pred_labels.append(1)  # Prediction exists
                gt_labels.append(0)    # But no object matched = FP

        # Now add FN: GTs that were never matched
        num_fn = len(gt_df) - len(matched_gt)
        pred_labels.extend([0] * num_fn)  # These "phantom predictions" never happened
        gt_labels.extend([1] * num_fn)    # They are real GTs that were missed

        return gt_labels, pred_labels



    def calculate_tile_size_px(self, tile_width_cm,tile_height_cm):
        tile_width_px = int(tile_width_cm / self.gds_x)
        tile_height_px = int(tile_height_cm / self.gds_y)
        return tile_width_px, tile_height_px
    

    def get_pixel_cm(self):
        # Approximate conversion factor: 1 degree = 111,320 meters at the equator
        # Convert pixel size from degrees to centimeters
        pixel_size_x_cm = abs(self.geotransform[1]) * 111320 * 100
        pixel_size_y_cm = abs(self.geotransform[5]) * 111320 * 100

        return pixel_size_x_cm, pixel_size_y_cm

