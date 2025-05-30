# create a class Box
class Box:
    def __init__(self, geotile, geomosaic, label, xmin, ymin, xmax, ymax):
        self.label = label
        self.geotile = geotile
        self.geomosaic = geomosaic
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = None
        self.area_pixels = None

    #other constructor with confidence
    def __init__(self, label, geotile, geomosaic, xmin, ymin, xmax, ymax, confidence):
        self.label = label
        self.geotile = geotile
        self.geomosaic = geomosaic
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence

    def __repr__(self):
        return f"Box(label={self.label}, xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"


    def resize_to_original(self):
        scale_x =  self.geotile["tile_width"] / self.geomosaic["output_tile_width"]
        scale_y =  self.geotile["tile_height"] / self.geomosaic["output_tile_height"]
        return Box(self.label, self.geotile, self.geomosaic,
            self.xmin * scale_x, self.ymin * scale_y,
            self.xmax * scale_x, self.ymax * scale_y, self.confidence
        )

    def convert_geospatial(self):

        pixel_width = float(self.geomosaic["pixel_width"])
        pixel_height = float(self.geomosaic["pixel_height"])

        # Get the tile's geospatial info
        top_left_x = float(self.geotile['xmin'])
        top_left_y = float(self.geotile['ymin'])

        # Convert to geospatial coordinates
        geo_x_min = top_left_x + self.xmin * pixel_width
        geo_y_min = top_left_y + self.ymin * pixel_height
        geo_x_max = top_left_x + self.xmax * pixel_width
        geo_y_max = top_left_y + self.ymax * pixel_height

        # Write to output
        return Box(self.label, self.geotile, self.geomosaic, geo_x_min, geo_y_min, geo_x_max, geo_y_max, self.confidence)
    



