# coding: utf-8

# --------------------------
#        Imports
# --------------------------

import logging
import os
import glob
import gzip
from timeit import default_timer as timer
import cv2
import numpy as np
from osgeo import osr, ogr, gdal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Linear interpolation for color maps

from managementMetadataFromRS.centerpivot import Utils
from managementMetadataFromRS.Utils import norm_minmax, salvar_banda, handle_ndvi_band, handle_bands
from managementMetadataFromRS.pymasker.pymasker import LandsatMasker, LandsatConfidence
import resource
import sys


def export_circles_detected(circles, filename_out, dataset, type='shp', targetEPSG=4326):
    """
    :param targetEPSG:
    :param type:
    :param circles:
    :param filename_out:
    :type dataset: sensor object attribute with an instance of gdal dataset class
    """
    if circles is not None:
        if circles.dtype == 'float32':
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

        geotransform = dataset.GetGeoTransform()

        circles_features = []
        # loop over the (x, y) coordinates and radius of the circles
        for x, y, r in circles:
            lrx = geotransform[0] + (x * geotransform[1])
            lry = geotransform[3] + (y * geotransform[5])

            circles_features.append([lrx,lry,r * geotransform[1]])

        geoCollection = Utils.create_geoCollection(circles_features)

        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromWkt(dataset.GetProjectionRef())
        target = spatial_reference

        """
        The coordinate reference system for all GeoJSON coordinates is a geographic coordinate reference system,
        using the World Geodetic System 1984 (WGS 84) [WGS84] datum, with longitude and latitude units of decimal
        degrees.
        ...
        Note: the use of alternative coordinate reference systems was specified in [GJ2008], but it has been
        removed from this version of the specification because the use of different coordinate reference systems
        - - especially in the manner specified in [GJ2008] - - has proven to have interoperability issues.In general,
        GeoJSON processing software is not expected to have access to coordinate reference system databases or to have
        network access to coordinate reference system transformation parameters.However, where all involved parties 
        have a prior arrangement, alternative coordinate reference systems can be used without risk of data being 
        misinterpreted. Source: RFC 7946 Butler, et al. (2016) https://tools.ietf.org/html/rfc7946#page-3"""

        #spatial reference Authority for WGS84 is 4326:
        if type == 'geojson' and spatial_reference.GetAttrValue("AUTHORITY", 1) != 4326:
            target = osr.SpatialReference()
            target.ImportFromEPSG(4326)
            if float('.'.join(gdal.__version__.split('.')[0:2])) >= 3.0:
              target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #About changes in GDAL since 3.0 https://gdal.org/tutorials/osr_api_tut.html - CRS and axis order

            transform = osr.CoordinateTransformation(spatial_reference, target)
            geoCollection.Transform(transform)
        elif spatial_reference.GetAttrValue("AUTHORITY", 1) != targetEPSG:
            print('Origem:\n', spatial_reference)

            target = osr.SpatialReference()
            target.ImportFromEPSG(targetEPSG)
            if float('.'.join(gdal.__version__.split('.')[0:2])) >= 3.0:
              target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #About changes in GDAL since 3.0 https://gdal.org/tutorials/osr_api_tut.html - CRS and axis order

            print('Destino:\n', target)

            transform = osr.CoordinateTransformation(spatial_reference, target)
            geoCollection.Transform(transform)


        if type == 'shp':
            Utils.write_shapefile(geoCollection,
                                  target,
                                  ogr.wkbPolygon,
                                  filename_out+'_polygon.shp')
        elif type == 'geojson':
            Utils.write_geojson(geoCollection,
                                target,
                                ogr.wkbPolygon,
                                filename_out+'.geojson')

    else:
        print('None circles detected!')


# Adapted from: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.ma.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    start = timer()
    edged = cv2.Canny(image, lower, upper)
    end = timer()
    print("Edged in: ", end - start," seconds.")
    print('median:', v, 'lower:', lower, ' upper:', upper)

    # return the edged image
    return edged


def finding_circles(auto, dp_value=1, min_dist_px=40,
                    param1=50,param2=15,
                    min_radii_px=15, max_radii_px=40):
    # ----------------------------------
    # Hough Transform to detect Circles:
    # ----------------------------------
    """Center pivots are typically less than 1600 feet (500 meters) in length (circle radius)
    with the most common size being the standard 1/4 mile (400 m) machine. A typical 1/4 mile
    radius crop circle covers about 125 acres of land"""

    print('Parameters for HoughCircles: dp:', dp_value,'minDist:', min_dist_px,
          'param1:',param1,'param2:',param2,'minRadius:', min_radii_px,
          'maxRadius:',max_radii_px)

    # detect circles in the image
    circles = cv2.HoughCircles(auto, cv2.HOUGH_GRADIENT,
                               dp=dp_value, minDist=min_dist_px,
                               param1=param1, param2=param2,
                               minRadius=int(min_radii_px), maxRadius=int(max_radii_px))


    # Ensure at least some circles were found
    if circles is not None:
        # Remove circles with radii missing value:
        circles_tmp = np.delete(circles, np.where(circles[:, :, 2] == 0.), axis=1)

        if circles_tmp.size == 0:
            circles = None
        else:
            circles = circles_tmp
            print('Nr. Circles Detected:', circles.shape[1])

    return circles


def calc_mask(index, _array, radius: int, *, methods=[np.ma.mean, np.ma.std]):
    """
    :param index: tuple with (x,y) center of circle
    :param radius: radii of circle to mask
    :param _array: numpy array image data
    :param methods: functions to apply on array circle mask
    :return: depends on methods parameter (mean, standard deviation or count pixels) of circle mask
    """
    a, b = index
    ny, nx = _array.shape
    x, y = np.ogrid[-b:ny - b, -a:nx - a]
    mask = x*x + y*y <= radius*radius

    result = []
    for method in methods:
        result.append(method(_array[mask]))

    return result

def define_geo_parameters(dataset):
    geotransform = dataset.GetGeoTransform()

    lrx = geotransform[0] + (dataset.RasterXSize * geotransform[1])
    lry = geotransform[3] + (dataset.RasterYSize * geotransform[5])

    corners = [geotransform[0], lry, lrx, geotransform[3]]
    # print("The extent should be inside: " + str(corners))

    center_pivot = 500.  # tipycally radii meters
    global min_dist_px
    global min_radii_px
    global max_radii_px

    # Optimal tunned:
    min_dist_px = (center_pivot + 300.) / geotransform[1]
    min_radii_px = (center_pivot - 200.) / geotransform[1]
    max_radii_px = (center_pivot + 300.) / geotransform[1]

    global res_x
    global res_y
    res_x = geotransform[1]
    res_y = -geotransform[5]


def FindCenterPivots(sensor: object, **kwargs):
    logging.captureWarnings(True)

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Starting detect pivots is: {0} KB".format(mem))

    """
    :type sensor: object defined in core package with info about bands of satellite
    :type kwargs: dict extra args
    
    """
    # Bands needed to calc NDVI:
    if hasattr(sensor, 'Red_array'):
        array_red = sensor.Red_array
    else:
        array_red = handle_bands(sensor, 'Red', **kwargs)
    sensor.Red_array = array_red

    print('Band red:',sys.getsizeof(array_red))
    print('Object sensor:',sys.getsizeof(sensor))
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at band red created is: {0} KB".format(mem))

    if hasattr(sensor, 'NIR_array'):
        array_nir = sensor.NIR_array
    else:
        array_nir = handle_bands(sensor, 'NIR', **kwargs)
    sensor.NIR_array = array_nir
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at band nir created is: {0} KB".format(mem))

    ## Allow division by zero
    # np.seterr(divide='ignore', invalid='ignore')
    
    ########################################################
    # Mask Pixels of Cloud Using Quality Accessment band:
    if hasattr(sensor, 'BQA') and sensor.BQA is not None:
        masker = LandsatMasker(sensor.BQA, collection=1)
        conf = LandsatConfidence.high
        cloud_mask = masker.get_cloud_mask(conf)
    
        array_red = np.ma.masked_where(cloud_mask == 1, array_red)
        sensor.Red_array = array_red
        array_nir = np.ma.masked_where(cloud_mask == 1, array_nir)
        sensor.NIR_array = array_nir
    
    """Satellite maps of vegetation show the density of plant growth over the entire 
    globe. The most common measurement is called the Normalized Difference Vegetation
     Index (NDVI). Very low values of NDVI (0.1 and below) correspond to barren areas 
     of rock, sand, or snow. Moderate values represent shrub and grassland (0.2 to 0.3), 
     while high values indicate temperate and tropical rainforests (0.6 to 0.8).
     https://earthobservatory.nasa.gov/Features/MeasuringVegetation"""

    #####################################################################
    # Normalizaded Difference Vegetation Index (Rouse et al.,1973):
    if hasattr(sensor, 'NDVI_array'):
        array_ndvi = sensor.NDVI_array
        print("NDVI limits:", array_ndvi.min(), array_ndvi.max())
    else:
        try:
            array_ndvi = handle_ndvi_band(sensor, **kwargs)
        except Exception as err:
            print("Unexpected error:", err)
            logging.error(err)
            raise
            return 1
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at band ndvi created is: {0} KB".format(mem))

    ####################################################################
    # Soil Adjusted Vegetation Index (SAVI) (Huete et al.,1988):
    """An L value of 0.5 in reflectance space was found to minimize soil
     brightness variations and eliminate the need for additional calibration
     for different soils"""

    L = 0.5 #Valor constante de ajuste de solo

    array_savi = (1.0 + L) * (array_nir - array_red) / (array_nir + array_red + L)
    print("SAVI limits:", array_savi.min(),array_savi.max())

    array_savi = np.ma.array(array_savi, mask=np.isnan(array_savi))

    array = np.uint8(norm_minmax(array_ndvi, 0, 255))
    # array_mask = np.ma.getmask(array_ndvi)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at band savi created is: {0} KB".format(mem))

    # Morphological Transformations
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    #
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(array,kernel,iterations = 1)
    array = erosion

    edges = auto_canny(array, sigma=0.33)
    # masked_data = np.ma.masked_where(edges == 0, edges)

    # Limiares para solo exposto, com base no trabalho:
    # IMAGENS DO LANDSAT- 8 NO MAPEAMENTO DE SUPERFÍCIES EM ÁREA IRRIGADA
    edges[(array_ndvi <= 0.44) & (array_savi <= 0.26)] = 0

    
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at before hough is: {0} KB".format(mem))

    """The function 'define_geo_parameters' get parameters needed for method Circle Hough Transform, 
    with aim to map center pivot irrigation on remote sensing image, these parameters depend on the 
    spatial resolution of the image"""
    define_geo_parameters(sensor.dataset)
    dp_value=1; par1=50; par2=15

    circles = finding_circles(edges,dp_value=dp_value,min_dist_px=min_dist_px,
                              param1=par1,param2=par2,
                              min_radii_px=min_radii_px,max_radii_px=max_radii_px)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at after hough is: {0} KB".format(mem))

    if circles is not None:
        if 'epsg_target' in kwargs:  # Transformation if necessary
            export_circles_detected(circles, os.path.join(kwargs['odir'], sensor.sat_name), sensor.dataset,
                                    type=kwargs['export_type'], targetEPSG=int(kwargs['epsg_target']))
        else:  # Save using original spatial reference of image
            spatial_reference = osr.SpatialReference()
            spatial_reference.ImportFromWkt(sensor.dataset.GetProjectionRef())

            EPSG_value = spatial_reference.GetAttrValue("AUTHORITY", 1)
            export_circles_detected(circles, os.path.join(kwargs['odir'], sensor.sat_name), sensor.dataset,
                                    type=kwargs['export_type'], targetEPSG=EPSG_value)

    else:
        print("None circle detected from this scene:", sensor.sat_name)
