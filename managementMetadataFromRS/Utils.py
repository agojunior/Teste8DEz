# coding: utf-8

# --------------------------
#        Imports
# --------------------------
from managementMetadataFromRS.centerpivot import Utils
import os
import gzip
import errno
from osgeo import gdal, gdal_array, osr, ogr
from gdalconst import *             # importar constantes
import numpy as np
import glob
import logging
import matplotlib.pyplot as plt
import sys
import resource

gdal.UseExceptions()                # informar o uso de exceções

# Notes - numpy.seterr
#
# The floating-point exceptions are defined in the IEEE 754 standard [1]:
#
#     Division by zero: infinite result obtained from finite numbers.
#     Overflow: result too large to be expressed.
#     Underflow: result so close to zero that some precision was lost.
#     Invalid operation: result is not an expressible number, typically indicates that a NaN was produced.
#
# [1]	http://en.wikipedia.org/wiki/IEEE_754
#
np.seterr(under='ignore')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def salvar_banda(matriz_de_pixels, nome_do_arquivo, dataset_de_referencia, NoData=None):
    # obter metadados
    linhas = dataset_de_referencia.RasterYSize
    colunas = dataset_de_referencia.RasterXSize
    bandas = 1
    # definir driver
    driver = gdal.GetDriverByName('GTiff')
    # copiar tipo de dados da banda já existente
    #data_type = dataset_de_referencia.GetRasterBand(1).DataType
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(matriz_de_pixels.dtype)
    # criar novo dataset
    dataset_output = driver.Create(nome_do_arquivo, colunas, linhas, bandas, data_type)
    # copiar informações espaciais da banda já existente
    dataset_output.SetGeoTransform(dataset_de_referencia.GetGeoTransform())
    # copiar informações de projeção
    dataset_output.SetProjection(dataset_de_referencia.GetProjectionRef())
    # escrever dados da matriz NumPy na banda
    dataset_output.GetRasterBand(1).WriteArray(matriz_de_pixels)
    # define no data value if required
    if NoData != None:
        dataset_output.GetRasterBand(1).SetNoDataValue(NoData)
    # salvar valores
    dataset_output.FlushCache()
    # fechar dataset
    dataset_output = None

    return


def get_raster_extent(dataset, targetEPSG=4326):

    geotransform = dataset.GetGeoTransform()

    ulx = geotransform[0]; uly = geotransform[3]  # upper left cell in raster

    # lower right cell in raster
    lrx = geotransform[0] + (dataset.RasterXSize * geotransform[1])
    lry = geotransform[3] + (dataset.RasterYSize * geotransform[5])

    # Setup the source projection
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(dataset.GetProjectionRef())

    # The target projection, convert if necessary:
    if int(spatial_reference.GetAttrValue("AUTHORITY", 1)) != targetEPSG:
        print('Origem:\n', spatial_reference)

        target = osr.SpatialReference()
        target.ImportFromEPSG(targetEPSG)
        if float('.'.join(gdal.__version__.split('.')[0:2])) >= 3.0:
              target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #About changes in GDAL since 3.0 https://gdal.org/tutorials/osr_api_tut.html - CRS and axis order

        print('Destino:\n', target)

        # Create the transform - this can be used repeatedly
        transform = osr.CoordinateTransformation(spatial_reference, target)

        # Transform the point. You can also create an ogr geometry and use the more generic `point.Transform()`
        ulx,uly = transform.TransformPoint(ulx, uly)[0:2]
        lrx, lry = transform.TransformPoint(lrx, lry)[0:2]


    xs = [lrx, ulx]
    ys = [lry, uly]
    bounds = [min(xs), min(ys), max(xs), max(ys)]
    print('Bounds:',bounds)

    return bounds


def export_raster_feature_extent(filename_out, dataset, array, targetEPSG=4326):

    geotransform = dataset.GetGeoTransform()
    raster_feature = []

    # Minimum box of valid values or extent of raster
    if isinstance(array,np.ndarray) and isinstance(array,np.ma.MaskedArray):

        points = np.zeros((4, 2), dtype=int)

        edge0 = np.ma.notmasked_edges(array, axis=0)
        #print('bottow left(y,x):', edge0[0][0][0], edge0[0][1][0])
        points[3,:] = [edge0[0][0][0], edge0[0][1][0]]

        #print('upper right(y,x):',edge0[0][0][-1], edge0[0][1][-1])
        points[1, :] = [edge0[0][0][-1], edge0[0][1][-1]]

        edge1 = np.ma.notmasked_edges(array, axis=1)
        #print('upper left(y,x):', edge1[0][0][0], edge1[0][1][0])
        points[0, :] = [edge1[0][0][0], edge1[0][1][0]]

        #print('bottow right(y,x):',edge1[1][0][-1], edge1[1][1][-1])
        points[2, :] = [edge1[1][0][-1], edge1[1][1][-1]]

        for y,x in points:
            lrx = geotransform[0] + (x * geotransform[1])
            lry = geotransform[3] + (y * geotransform[5])
            raster_feature.append([lrx, lry])

    else:
        raster_feature.append([geotransform[0],geotransform[3]]) #upper left cell in raster

        lrx = geotransform[0] + (dataset.RasterXSize * geotransform[1])
        lry = geotransform[3] + (0 * geotransform[5])
        raster_feature.append([lrx, lry])  # upper right cell in raster

        lrx = geotransform[0] + (dataset.RasterXSize * geotransform[1])
        lry = geotransform[3] + (dataset.RasterYSize * geotransform[5])
        raster_feature.append([lrx, lry])  # lower right cell in raster

        lrx = geotransform[0] + (0 * geotransform[1])
        lry = geotransform[3] + (dataset.RasterYSize * geotransform[5])
        raster_feature.append([lrx, lry])  # lower left cell in raster


    geoPolygon = Utils.create_geoPolygon(raster_feature)

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(dataset.GetProjectionRef())
    target = spatial_reference

    if int(spatial_reference.GetAttrValue("AUTHORITY", 1)) != targetEPSG:
        print('Origem:\n', spatial_reference)

        target = osr.SpatialReference()
        target.ImportFromEPSG(targetEPSG)
        if float('.'.join(gdal.__version__.split('.')[0:2])) >= 3.0:
              target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #About changes in GDAL since 3.0 https://gdal.org/tutorials/osr_api_tut.html - CRS and axis order

        print('Destino:\n', target)

        transform = osr.CoordinateTransformation(spatial_reference, target)
        geoPolygon.Transform(transform)


    destino = './shp'
    mkdir_p(destino)

    Utils.write_shapefile(geoPolygon,
                          target,
                          ogr.wkbPolygon,
                          os.path.join(destino,filename_out+'_polygon.shp'))


"""Feature scaling is used to bring all values into the range [0,1]. This is also called unity-based
normalization. This can be generalized to restrict the range of values in the dataset between any arbitrary
points a and b using:

X ′ = a + ( X − Xmin ) ( b − a ) / (Xmax − Xmin)"""


def norm_minmax(array, min, max):
    return (array - array.min()) / (array.max() - array.min()) * (max - min) + min


def handle_ndvi_band(sensor: object, **kwargs):

    """Satellite maps of vegetation show the density of plant growth over the entire
        globe. The most common measurement is called the Normalized Difference Vegetation
         Index (NDVI). Very low values of NDVI (0.1 and below) correspond to barren areas
         of rock, sand, or snow. Moderate values represent shrub and grassland (0.2 to 0.3),
         while high values indicate temperate and tropical rainforests (0.6 to 0.8).
         https://earthobservatory.nasa.gov/Features/MeasuringVegetation"""
    # Ignore underflow Floating error:
    np.seterr(under='ignore')

    # Bands to calc NDVI:
    if hasattr(sensor, 'Red_array'):
        array_red = sensor.Red_array
    else:
        array_red = handle_bands(sensor, 'Red', **kwargs)
    sensor.Red_array = array_red

    if hasattr(sensor, 'NIR_array'):
        array_nir = sensor.NIR_array
    else:
        array_nir = handle_bands(sensor, 'NIR', **kwargs)
    sensor.NIR_array = array_nir

    #####################################################################
    # Normalizaded Difference Vegetation Index (Rouse et al.,1973):
    filename_ndvi = (glob.glob(os.path.join(kwargs['odir'], sensor.sat_name + '_NDVI.tif[xml]*'))[:1] or [None])[0] #Excluding auxiliary files (.xml)
    array_ndvi = np.array([])

    if filename_ndvi is not None:
        array_ndvi = sensor.get_band_array(filename_ndvi).astype('float')

    if filename_ndvi is None or array_ndvi.shape != array_red.shape:
        if filename_ndvi is not None:
            os.unlink(filename_ndvi)  # delete previous NDVI band different size

        array_ndvi = (array_nir - array_red) / (array_nir + array_red)
        array_ndvi = np.ma.array(array_ndvi, mask=np.isnan(array_ndvi))  # Use a mask to mark the NaNs

        filename_ndvi = os.path.join(kwargs['odir'], sensor.sat_name + '_NDVI.tif')
        try:
            salvar_banda(array_ndvi, filename_ndvi, sensor.dataset, NoData=0)
        except Exception as err:
            print(err)

        with open(filename_ndvi, 'rb') as f_in, gzip.open(filename_ndvi + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)

        # If you want to delete the original file after the gzip is done:
        os.unlink(filename_ndvi)

    sensor.NDVI_array = array_ndvi
    print("NDVI limits:", array_ndvi.min(), array_ndvi.max())

    return array_ndvi


def handle_bands(sensor: object, name: str, **kwargs):
    """
    :type sensor: object defined in core package with info about bands of satellite
    :type name: str name of band to handle
    :type kwargs: dict extra args
    """
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at handle band before create band is: {0} KB".format(mem))

    if name == 'Red':
        if hasattr(sensor, 'RedSR') and sensor.RedSR is not None:
            array = sensor.get_band_array(sensor.RedSR).astype('float')
        else:
            array = sensor.get_band_array(sensor.Red).astype('float')
        sensor.Red_array = array
    elif name == 'NIR':
        if hasattr(sensor, 'NIRSR') and sensor.NIRSR is not None:
            array = sensor.get_band_array(sensor.NIRSR).astype('float')
        else:
            array = sensor.get_band_array(sensor.NIR).astype('float')
        sensor.NIR_array = array
    elif name == 'SWIR1':
        if hasattr(sensor, 'SWIR1SR') and sensor.SWIR1SR is not None:
            array = sensor.get_band_array(sensor.SWIR1SR).astype('float')
        else:
            array = sensor.get_band_array(sensor.SWIR1).astype('float')
        sensor.SWIR1_array = array
    elif name == 'Blue':
        if hasattr(sensor, 'BlueSR') and sensor.BlueSR is not None:
            array = sensor.get_band_array(sensor.BlueSR).astype('float')
        else:
            array = sensor.get_band_array(sensor.Blue).astype('float')
        sensor.Blue_array = array
    elif name == 'Green':
        if hasattr(sensor, 'GreenSR') and sensor.GreenSR is not None:
            array = sensor.get_band_array(sensor.GreenSR).astype('float')
        else:
            array = sensor.get_band_array(sensor.Green).astype('float')
        sensor.Green_array = array
    elif name == 'SWIR2':
        if hasattr(sensor, 'SWIR2SR') and sensor.SWIR2SR is not None:
            array = sensor.get_band_array(sensor.SWIR2SR).astype('float')
        else:
            array = sensor.get_band_array(sensor.SWIR2).astype('float')
        sensor.SWIR2_array = array
    else:
        print("ERROR:The function handle_bands not prepared to handle band:", name)
        logging.error("The function handle_bands not prepared to handle band: %s", name)
        raise RuntimeError

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at handle band after create band is: {0} KB".format(mem))
    print('Band in handle band:',sys.getsizeof(array))
    return array
