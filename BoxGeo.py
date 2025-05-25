class BoxGeo():
        
    def __init__(self,label, xmin, ymin, xmax, ymax, confidence=None, id = None):
        self.id = None
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence
        self.id = id

    def setId(self, id):
        self.id = id

    # Function to resize and clip bounding boxes to match resized image
    def resize_bounding_box(self, orig_width, orig_height, new_width, new_height):
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        # Resize the bounding box
        xmin = self.xmin * scale_x
        ymin = self.ymin * scale_y
        xmax = self.xmax * scale_x
        ymax = self.ymax * scale_y

        # Clip the bounding box to ensure it stays within the image limits
        xmin = max(0, min(xmin, new_width))
        ymin = max(0, min(ymin, new_height))
        xmax = max(0, min(xmax, new_width))
        ymax = max(0, min(ymax, new_height))

        # Write to output
        b_resized = BoxGeo(self.label, xmin, ymin, xmax, ymax, self.confidence, self.id)
        b_resized.setId(self.id)
        return b_resized

    def resize_bounding_box_segmentation(self, orig_width, orig_height, new_width, new_height):
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        # Resize the bounding box
        xmin = self.xmin * scale_x
        ymin = self.ymin * scale_y
        xmax = self.xmax * scale_x
        ymax = self.ymax * scale_y

        # Write to output
        b_resized = BoxGeo(self.label, xmin, ymin, xmax, ymax, self.confidence, self.id)
        b_resized.setId(self.id)
        return b_resized


    # convert xmin,ymin,xmax,ymax format to yolo format
    def bbox_to_yolo(self, image_width, image_height):
        bbox_width = int(self.xmax - self.xmin)
        bbox_height = int(self.ymax - self.ymin)
        x_center = int(self.xmin + bbox_width / 2)
        y_center = int(self.ymin + bbox_height / 2)

        # Normalize coordinates
        x_center /= image_width
        y_center /= image_height
        bbox_width /= image_width
        bbox_height /= image_height
        # Add to results in YOLO format
        return self.label, x_center, y_center, abs(bbox_width), abs(bbox_height)
    

    # Check if bbox is inside of tile
    def is_bbox_in_tile(self, tile, min_area_ratio=0.3):
        # Calculate intersection rectangle

        inter_x_min = max(tile.xmin, self.xmin)
        inter_y_min = max(tile.ymin, self.ymin)
        inter_x_max = min(tile.xmax, self.xmax)
        inter_y_max = min(tile.ymax, self.ymax)

        # Check if intersection is valid
        inter_width = inter_x_max - inter_x_min
        inter_height = inter_y_max - inter_y_min
        if inter_width > 0 and inter_height > 0: # if there is an intersection
            # Calculate intersection area
            inter_area = inter_width * inter_height
            bbox_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
            if inter_area / bbox_area >= min_area_ratio:
                return True

        return False
    
    def geospatial_to_pixel_coords(self, geotransform, tile_width, tile_height,x_offset,y_offset):
        # Convert geospatial to pixel
        # Convert geo coordinates to pixel coordinates in the full orthomosaic
        xmin_full_pixel = int((self.xmin - geotransform[0]) / geotransform[1])
        xmax_full_pixel = int((self.xmax - geotransform[0]) / geotransform[1])
        ymin_full_pixel = int((self.ymin - geotransform[3]) / geotransform[5])
        ymax_full_pixel = int((self.ymax - geotransform[3]) / geotransform[5])

        # Adjust these pixel coordinates relative to the current tile (from 0 to 640)
        xmin_tile_pixel = max(0, xmin_full_pixel - x_offset)
        xmax_tile_pixel = min(tile_width, xmax_full_pixel - x_offset)
        ymin_tile_pixel = max(0, ymin_full_pixel - y_offset)
        ymax_tile_pixel = min(tile_height, ymax_full_pixel - y_offset)
        boxpixel =  BoxGeo(self.label, xmin_tile_pixel, ymin_tile_pixel, xmax_tile_pixel, ymax_tile_pixel, self.confidence, self.id)
        boxpixel.setId(self.id)
        return boxpixel
    
    # convert xmin,ymin,xmax,ymax format to yolo format
    def bbox_to_yolo(self, image_width, image_height):
        bbox_width = int(self.xmax - self.xmin)
        bbox_height = int(self.ymax - self.ymin)
        x_center = int(self.xmin + bbox_width / 2)
        y_center = int(self.ymin + bbox_height / 2)

        # Normalize coordinates
        x_center /= image_width
        y_center /= image_height
        bbox_width /= image_width
        bbox_height /= image_height
        # Add to results in YOLO format
        return self.label, x_center, y_center, abs(bbox_width), abs(bbox_height)
    
    def bbox_to_coco(self):
        bbox_width = self.xmax - self.xmin
        bbox_height = self.ymax - self.ymin
        x = self.xmin
        y = self.ymin

        # COCO format: [x_top_left, y_top_left, width, height]
        # Category ID is assumed to be an int already
        return {
            "category_id": int(self.label),
            "bbox": [float(x), float(y), float(abs(bbox_width)), float(abs(bbox_height))],
            "area": abs(bbox_width * bbox_height)
        }