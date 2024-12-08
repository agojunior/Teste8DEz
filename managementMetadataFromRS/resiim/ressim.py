# coding: utf-8

# --------------------------
#        Imports
# --------------------------
import logging
import os
import sys
import shutil
import numpy as np
from osgeo import gdal
import json
from fmask.cmdline.usgsLandsatStacked import mainRoutine
from managementMetadataFromRS.Utils import handle_ndvi_band, handle_bands, get_raster_extent


def runFMASK(sensor: object, **kwargs):
    """
    For Cloud Mask

    The input image must contain multispectral bands within the following wavelength ranges:
    Blue: 0.45 - 0.523 µm
    Green: 0.52 - 0.605 µm
    Red: 0.63 - 0.69 µm
    NIR: 0.75 - 0.9 µm
    SWIR1: 1.55 - 1.75 µm
    SWIR2: 2.05 - 2.380 µm

    If the image has thermal and cirrus bands, these will improve the accuracy of the cloud mask result.
    Landsat 8 OLI and Sentinel-2A sensors have cirrus bands. The thermal band must range from 10.4 to 12.5 µm.
    The cirrus band must range from 1.36 to 1.385 µm.
    Source: https://www.harrisgeospatial.com/docs/CalculateCloudMaskUsingFmask.html


    Python Fmask
    ------------

    A set of command line utilities and Python modules that implement the ‘fmask’ algorithm as published in:

    Zhu, Z. and Woodcock, C.E. (2012). Object-based cloud and cloud shadow detection in Landsat imagery Remote
    Sensing of Environment 118 (2012) 83-94.

    and

    Zhu, Z., Wang, S. and Woodcock, C.E. (2015). Improvement and expansion of the Fmask algorithm: cloud, cloud shadow,
     and snow detection for Landsats 4-7, 8, and Sentinel 2 images Remote Sensing of Environment 159 (2015) 269-277.

    Also includes optional extension for Sentinel-2 from Frantz, D., Hass, E., Uhl, A., Stoffels, J., & Hill, J. (2018).
    Improvement of the Fmask algorithm for Sentinel-2 images: Separating clouds from bright surfaces based on parallax
    effects. Remote Sensing of Environment 215, 471-481.

    The output from the core algorithm module is a single thematic raster, with integer codes representing null, clear,
    cloud, shadow, snow, water respectively.

    :return: None
    """
    fullname = getattr(sensor, 'Blue')
    dirname = os.path.split(fullname)[0]
    source_dir = os.path.join(dirname, sensor.sat_name)
    output_fmask = os.path.join(kwargs['odir'], sensor.sat_name + '_fmask.img')


    sys.argv = [sys.argv[0]]

    fmask_args = '-o ' + output_fmask + ' --scenedir ' + source_dir + ' -e ' + source_dir
    for arg in fmask_args.split():
        sys.argv.append(arg)

    print('Running Fmask...')
    logging.info('Running Fmask...')
    try:
        mainRoutine()  # Main routine that calls fmask, get args from command line arguments (sys.argv)
        shutil.rmtree(source_dir)
    except:
        shutil.rmtree(source_dir)
        raise


    return

#https://gis.stackexchange.com/questions/73768/converting-geojson-to-python-objects
def save_ressim_targets_information(filename_out, metadata_dict, bbox):
    if os.path.exists(filename_out+'.geojson'):
        # read in json  and append properties:
        geo_objects = json.load(open(filename_out+'.geojson'))
        geo_objects['properties'] = metadata_dict
        geo_objects['bbox'] = bbox
    else:
        geo_objects = {}
        geo_objects['properties'] = metadata_dict
        geo_objects['bbox'] = bbox

    with open(filename_out + '.geojson', 'w') as outfile:
        json.dump(geo_objects, outfile)



def FindReSSIMtargets(sensor: object, **kwargs):
    logging.captureWarnings(True)

    """
    :type sensor: object defined in core package with info about bands of satellite
    :type kwargs: dict extra args

    """
    output_fmask = os.path.join(kwargs['odir'], sensor.sat_name + '_fmask.img')

    if not os.path.exists(output_fmask):
        try:
            runFMASK(sensor, **kwargs)
        except Exception as err:
            print("Unexpected error:", err)
            logging.error(err)
            raise
            return 1

    # Bands needed to calc Indexes:
    if hasattr(sensor, 'Red_array'):
        array_red = sensor.Red_array
    else:
        array_red = handle_bands(sensor, 'Red', **kwargs)

    if hasattr(sensor, 'NIR_array'):
        array_nir = sensor.NIR_array
    else:
        array_nir = handle_bands(sensor, 'NIR', **kwargs)

    if hasattr(sensor, 'Blue_array'):
        array_blue = sensor.Blue_array
    else:
        array_blue = handle_bands(sensor, 'Blue', **kwargs)

    if hasattr(sensor, 'Green_array'):
        array_green = sensor.Green_array
    else:
        array_green = handle_bands(sensor, 'Green', **kwargs)

    if hasattr(sensor, 'SWIR1_array'):
        array_swir1 = sensor.SWIR1_array
    else:
        array_swir1 = handle_bands(sensor, 'SWIR1', **kwargs)

    if hasattr(sensor, 'SWIR2_array'):
        array_swir2 = sensor.SWIR2_array
    else:
        array_swir2 = handle_bands(sensor, 'SWIR2', **kwargs)


    #####################################################################
    # Normalizaded Difference Vegetation Index (Rouse et al.,1973):
    if hasattr(sensor, 'NDVI_array'):
        array_ndvi = sensor.NDVI_array
        print("NDVI limits:", array_ndvi.min(), array_ndvi.max())
    else:
        try:
            array_ndvi = handle_ndvi_band(sensor,**kwargs)
        except Exception as err:
            print("Unexpected error:", err)
            logging.error(err)
            raise
            return 1

    #####################################################################
    # Bare Soil Index (Rikimaru et al.,2002):
    # Rikimaru, P.S. Roy and S. Miyatake, 2002. Tropical forest cover density mapping.
    # Tropical Ecology Vol. 43, №1, pp 39–47.
    # Link useful: https://medium.com/regen-network/remote-sensing-indices-389153e3d947
    array_bsi = ((array_red + array_blue) - array_green) / ((array_red + array_blue) + array_green)
    print("BSI limits:", array_bsi.min(), array_bsi.max())

    #####################################################################
    # Normalized Difference Built-up Index (ZHA; GAO; NI, 2003):
    # ZHA, Y.; GAO, J.; NI, S. Use of normalized difference built-up index in automatically mapping
    # urban areas from TM imagery. International Journal of Remote Sensing, v. 24, n. 3, 2003. pp. 583–594.
    # Links useful: https://www.linkedin.com/pulse/ndvi-ndbi-ndwi-calculation-using-landsat-7-8-tek-bahadur-kshetri/
    # https://gis.stackexchange.com/questions/277993/ndbi-formula-for-landsat-8
    array_ndbi = (array_swir1 - array_nir) / (array_swir1 + array_nir)
    print("NDBI limits:", array_ndbi.min(), array_ndbi.max())

    #####################################################################
    # Burned Area Index (Chuvieco et al.,2002):
    # Chuvieco, E., M. Pilar Martin, and A. Palacios. “Assessment of Different Spectral Indices in the Red-Near-Infrared
    # Spectral Domain for Burned Land Discrimination.” Remote Sensing of Environment 112 (2002): 2381-2396.
    # https://www.researchgate.net/profile/Emilio_Chuvieco/publication/228788017_Assessment_of_different_spectral_indices_in_the_red-near-infrared_spectral_domain_for_burned_land_discrimination/links/54aea1bf0cf29661a3d39e94.pdf
    #
    # Martín, M. Cartografía e inventario de incendios forestales en la Península Iberica a partir de imágenes NOAA
    # AVHRR. Doctoral thesis, Universidad de Alcalá, Alcalá de Henares (1998).

    # This index highlights burned land in the red to near-infrared spectrum, by emphasizing the charcoal signal in
    # post-fire images. The index is computed from the spectral distance from each pixel to a reference spectral point,
    # where recently burned areas converge. Brighter pixels indicate burned areas.
    # Source: https://www.harrisgeospatial.com/docs/BackgroundBurnIndices.html
    # Note:
    # The data must be calibrated to reflectance before applying the index.

    # Values of 0.1 (Red) and 0.06 (NIR) are reference reflectances, respectively, based on literature and analysis of
    # several sets of satellite sensor images (Martín 1998). These values tend to emphasize the charcoal signal of
    # burned areas.
    array_bai = 1.0/((0.1 - array_red)**2.0 + (0.06 - array_nir)**2.0)
    print("BAI limits:", array_bai.min(),array_bai.max())

    #####################################################################
    # Normalized Burn Ratio (NBR) (Key and Benson, 2005):
    # Lopez Garcia, M., and V. Caselles. "Mapping Burns and Natural Reforestation using Thematic Mapper Data.
    # Geocarto International 6 (1991): 31-37.
    #
    # Key, C. and N. Benson, N. "Landscape Assessment: Remote Sensing of Severity, the Normalized Burn Ratio; and
    # Ground Measure of Severity, the Composite Burn Index." In FIREMON: Fire Effects Monitoring and Inventory System,
    # RMRS-GTR, Ogden, UT: USDA Forest Service, Rocky Mountain Research Station (2005).
    #
    # This index highlights burned areas in large fire zones greater than 500 acres. NBR was originally developed for
    # use with Landsat TM and ETM+ bands 4 and 7, but it will work with any multispectral sensor with a NIR band
    # between 0.76-0.9 µm and a SWIR band between 2.08-2.35 µm.
    #
    # You can create pre-fire and post-fire NBR images, then subtract the post-fire image from the pre-fire image to
    # create a differenced (or delta) NBR image that indicates burn severity.
    #
    # Darker pixels indicate burned areas.
    # Source: https://www.harrisgeospatial.com/docs/BackgroundBurnIndices.html
    array_nbr = (array_nir - array_swir2) / (array_nir + array_swir2)
    print("NBR limits:", array_nbr.min(), array_nbr.max())

    # Read category raster output from Fmask:
    fmask = gdal.Open(output_fmask, gdal.GA_ReadOnly)
    fmask_arr = fmask.ReadAsArray()

    clear_land = (fmask_arr == 1)
    land = np.sum(clear_land)
    cloud = np.sum(fmask_arr == 2)
    shadow = np.sum(fmask_arr == 3)
    snow = np.sum(fmask_arr == 4)
    water = np.sum(fmask_arr == 5)
    total = land + cloud + shadow + water

    per_land = (land * 100.) / total
    per_cloud = (cloud * 100.) / total
    per_shadow = (shadow * 100.) / total
    per_water = (water * 100.) / total

    print('Percent Clean Land:',per_land)
    print('Percent Cloud:', per_cloud)
    print('Percent Shadow:', per_shadow)
    print('Percent Water:', per_water)


    print('NDVI total nonzero:', np.count_nonzero(array_ndvi))
    print('NDVI total count masked array:', np.ma.MaskedArray.count(array_ndvi))
    print('NDVI type:', type(array_ndvi))

    print('BSI total nonzero:', np.count_nonzero(array_bsi))
    print('BSI total count masked array:', np.ma.MaskedArray.count(array_bsi))
    print('BSI type:', type(array_bsi))

    print('NDBI total nonzero:', np.count_nonzero(array_ndbi))
    print('NDBI total count masked array:', np.ma.MaskedArray.count(array_ndbi))
    print('NDBI type:', type(array_ndbi))

    print('BAI total nonzero:', np.count_nonzero(array_bai))
    print('BAI total count masked array:', np.ma.MaskedArray.count(array_bai))
    print('BAI type:', type(array_bai))

    print('NBR total nonzero:', np.count_nonzero(array_nbr))
    print('NBR total count masked array:', np.ma.MaskedArray.count(array_nbr))
    print('NBR type:', type(array_nbr))


    # Thresholds for especifics targets defined for (Plesth et al., 2018)
    limiar_ndvi_min = 0.0
    limiar_ndvi_max = 0.2
    limiar_bsi = 0.0
    limiar_ndvi_forest = 0.8
    limiar_ndbi = 0.0
    limiar_bai = 250.
    limiar_nbr = 0.1

    mask_forest = (array_ndvi >= limiar_ndvi_forest)
    count_forest = np.sum(mask_forest[clear_land])
    mask_baresoil = (array_ndvi > limiar_ndvi_min) & (array_ndvi < limiar_ndvi_max) & (array_bsi > limiar_bsi)
    count_baresoil = np.sum(mask_baresoil[clear_land])
    mask_built_up = (array_ndbi >= limiar_ndbi) & (array_bsi <= limiar_bsi)
    count_built_up = np.sum(mask_built_up[clear_land])
    mask_burnedarea = (array_bai >= limiar_bai) & (array_nbr > limiar_nbr)
    count_burnedarea = np.sum(mask_burnedarea[clear_land])
    count_total = np.ma.MaskedArray.count(array_ndvi)

    per_forest = (count_forest * 100.) / count_total
    per_baresoil = (count_baresoil * 100.) / count_total
    per_built_up = (count_built_up * 100.) / count_total
    per_burnedarea = (count_burnedarea * 100.) / count_total

    print('Percent Forest:', per_forest)
    print('Percent Bare Soil:', per_baresoil)
    print('Percent Built-Up:', per_built_up)
    print('Percent Burned Areas:', per_burnedarea)

    # RFC 7946 - GeoJSON - August 2016 - https://tools.ietf.org/html/rfc7946#section-5
    # A GeoJSON object MAY have a member named "bbox" to include information on the coordinate
    # range for its Geometries, Features, or FeatureCollections.
    #
    # They usually follow the standard format of:
    #
    # bbox = left,bottom,right,top
    # bbox = min Longitude , min Latitude , max Longitude , max Latitude
    # For example, Greater London is enclosed by:
    #
    # {{bbox|-0.489|51.28|0.236|51.686}}
    # Source: https://wiki.openstreetmap.org/wiki/Bounding_Box
    bbox = get_raster_extent(sensor.dataset, targetEPSG=4326)
    print(bbox)

    metadatas = {'Cloud':per_cloud, 'Shadow':per_shadow, 'Water':per_water, 'Clear Land':per_land, 'Forest':per_forest,\
                 'Bare Soil':per_baresoil, 'Built-Up':per_built_up,'Burned Area':per_burnedarea}

    save_ressim_targets_information(os.path.join(kwargs['odir'], sensor.sat_name), metadatas, bbox)


    return