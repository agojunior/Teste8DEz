# coding: utf-8

# --------------------------
#        Imports
# --------------------------
import logging
import argparse
import sys, glob, os
import re
import numpy as np
import binascii
from osgeo import gdal
from gdalconst import *  # importar constantes
from zipfile import ZipFile, is_zipfile
from timeit import default_timer as timer
import shutil
import gzip
#include parent directory in sys.path:
sys.path.append(os.path.dirname(os.getcwd()))
from managementMetadataFromRS.centerpivot import DetectingCenterPivot
from managementMetadataFromRS.Utils import mkdir_p
from managementMetadataFromRS.resiim import ressim
import resource 

gdal.UseExceptions()  # Enable exceptions

# Test Magic Number for GZIP files
# https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed


def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'


class Sensor:
    def __init__(self, listfiles: list):
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory usage at Starting object creation is: {0} KB".format(mem))
        """

        :type listfiles: object
        """
        self.logger = logging.getLogger(__name__)

        if 'CPL_ZIP_ENCODING' not in os.environ: # to fix https://github.com/conda-forge/gdal-feedstock/issues/83
            os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'     # Set a new env. variable

        self.sensorID = (os.path.split(listfiles[0])[1]).split('_')[0]
        #self.Red = self.NIR = self.NDVI_array = None
        self.Red = self.NIR = None

        if self.sensorID == 'CBERS':
            if (os.path.split(listfiles[0])[1])[0:11] == 'CBERS_4_MUX':
                self.Blue = next((s for s in listfiles
                if re.search(r'_BAND5(.zip)|BAND5([.][Tt][Ii][Ff])',s)), None)
                self.Green = next((s for s in listfiles
                if re.search(r'_BAND6(.zip)|BAND6([.][Tt][Ii][Ff])',s)), None)
                self.Red = next((s for s in listfiles
                if re.search(r'_BAND7(.zip)|BAND7([.][Tt][Ii][Ff])',s)), None)
                self.NIR = next((s for s in listfiles
                if re.search(r'_BAND8(.zip)|BAND8([.][Tt][Ii][Ff])',s)), None)
                # Surf Reflectance
                self.BlueSR = next((s for s in listfiles
                if re.search(r'_BAND5_GRID_SURFACE(.zip)|BAND5_GRID_SURFACE([.][Tt][Ii][Ff])',s)), None)
                self.GreenSR = next((s for s in listfiles
                if re.search(r'_BAND6_GRID_SURFACE(.zip)|BAND6_GRID_SURFACE([.][Tt][Ii][Ff])', s)), None)
                self.RedSR = next((s for s in listfiles
                if re.search(r'_BAND7_GRID_SURFACE(.zip)|BAND7_GRID_SURFACE([.][Tt][Ii][Ff])', s)), None)
                self.NIRSR = next((s for s in listfiles
                if re.search(r'_BAND8_GRID_SURFACE(.zip)|BAND8_GRID_SURFACE([.][Tt][Ii][Ff])', s)), None)

                self.sat_name = os.path.split(listfiles[0])[1][0:28]
                self.dataset = None

        # Fonte - https://landsat.usgs.gov/landsat-collections#C1%20Tiers
        # LXSS_LLLL_PPPRRRYYYYMMDD_yyyymmdd_CC.TX
        # L = Landsat (constant)
        # X = Sensor (“C” = OLI/TIRS Combined, “O” = OLI-only, “E” = ETM+, “T” = TM, “M”= MSS)
        # SS = Satellite (e.g. “07” = Landsat 7, “08” = Landsat 8)
        # LLLL = Processing correction level (L1TP/L1GT/L1GS)
        # PPP = WRS path
        # RRR = WRS row
        # YYYYMMDD = Acquisition year(YYYY)/Month(MM)/Day(DD)
        # yyyymmdd = Processing year(yyyy)/Month(mm)/Day(dd)
        # CC = Collection number (“01”,”02”,...)
        # TX = Collection category (“RT” = Real-Time, “T1” = Tier 1, “T2” = Tier 2)

        elif self.sensorID == 'LE07':
            self.Blue = next((s for s in listfiles if re.search(r'_[Bb]1[.]', s)), None)
            self.Green = next((s for s in listfiles if re.search(r'_[Bb]2[.]', s)), None)
            self.Red = next((s for s in listfiles if re.search(r'_[Bb]3[.]', s)), None)
            self.NIR = next((s for s in listfiles if re.search(r'_[Bb]4[.]', s)), None)
            self.SWIR1 = next((s for s in listfiles if re.search(r'_[Bb]5[.]', s)), None)
            self.TIRS = next((s for s in listfiles if re.search(r'_[Bb]6[.]', s)), None)
            self.SWIR2 = next((s for s in listfiles if re.search(r'_[Bb]7[.]', s)), None)
            self.Pan = next((s for s in listfiles if re.search(r'_[Bb]8[.]', s)), None)
            self.BlueSR = next((s for s in listfiles if re.search(r'_sr_band1[.]', s)), None) # Surf Reflectance
            self.GreenSR = next((s for s in listfiles if re.search(r'_sr_band2[.]', s)), None)
            self.RedSR = next((s for s in listfiles if re.search(r'_sr_band3[.]', s)), None)
            self.NIRSR = next((s for s in listfiles if re.search(r'_sr_band4[.]', s)), None)
            self.SWIR1SR = next((s for s in listfiles if re.search(r'_sr_band5[.]', s)), None)
            self.SWIR2SR = next((s for s in listfiles if re.search(r'_sr_band7[.]', s)), None)
            self.MTL = next((s for s in listfiles if re.search(r'_MTL[.]', s)), None)
            self.BQA = next((s for s in listfiles if re.search(r'_BQA[.]', s)), None)

            self.sat_name = os.path.split(listfiles[0])[1][0:25]
            self.dataset = None

        elif self.sensorID == 'LC08':
            self.Coastal = next((s for s in listfiles if re.search(r'_[Bb]1[.]', s)), None)
            self.Blue = next((s for s in listfiles if re.search(r'_[Bb]2[.]', s)), None)
            self.Green = next((s for s in listfiles if re.search(r'_[Bb]3[.]', s)), None)
            self.Red = next((s for s in listfiles if re.search(r'_[Bb]4[.]', s)), None)
            self.NIR = next((s for s in listfiles if re.search(r'_[Bb]5[.]', s)), None)
            self.SWIR1 = next((s for s in listfiles if re.search(r'_[Bb]6[.]', s)), None)
            self.SWIR2 = next((s for s in listfiles if re.search(r'_[Bb]7[.]', s)), None)
            self.Pan = next((s for s in listfiles if re.search(r'_[Bb]8[.]', s)), None)
            self.Cirrus = next((s for s in listfiles if re.search(r'_[Bb]9[.]', s)), None)
            self.TIRS1 = next((s for s in listfiles if re.search(r'_[Bb]10[.]', s)), None)
            self.TIRS2 = next((s for s in listfiles if re.search(r'_[Bb]11[.]', s)), None)
            self.CoastalSR = next((s for s in listfiles if re.search(r'_sr_band1[.]', s)), None)  # Surf Reflectance
            self.BlueSR = next((s for s in listfiles if re.search(r'_sr_band2[.]', s)), None)
            self.GreenSR = next((s for s in listfiles if re.search(r'_sr_band3[.]', s)), None)
            self.RedSR = next((s for s in listfiles if re.search(r'_sr_band4[.]', s)), None)
            self.NIRSR = next((s for s in listfiles if re.search(r'_sr_band5[.]', s)), None)
            self.SWIR1SR = next((s for s in listfiles if re.search(r'_sr_band6[.]', s)), None)
            self.SWIR2SR = next((s for s in listfiles if re.search(r'_sr_band7[.]', s)), None)
            self.MTL = next((s for s in listfiles if re.search(r'_MTL[.]', s)), None)
            self.BQA = next((s for s in listfiles if re.search(r'_BQA[.]', s)), None)

            self.sat_name = os.path.split(listfiles[0])[1][0:25]
            self.dataset = None

        else:
            self.sat_name = os.path.split(listfiles[0])[1][0:25]
            print("Unknown Satellite Image:", os.path.split(listfiles[0])[1])
            logging.warning("Unknown Satellite Image: %s", os.path.split(listfiles[0])[1])
        
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory usage at Finish object creation is: {0} KB".format(mem))

    def get_band_array(self, filename, band_num=1):
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory usage at get_band_array before open tif is: {0} KB".format(mem))

        # https://gdal.org/gdal_virtual_file_systems.html
        global dataset
        if is_gz_file(filename):
            print("Tentar abrir gzip " + filename)

            try:
                start = timer()
                dataset = gdal.Open('/vsigzip/%s' % (filename))
                end = timer()
                print("File gunziped in: ", end - start, " seconds.")

            except:
                if 'dataset' in locals():
                    del dataset
                print("Erro na abertura do arquivo!")
                logging.error(gdal.GetLastErrorMsg())
                raise

        elif is_zipfile(filename):
            with ZipFile(filename) as theZip:
                fileNames = theZip.namelist()
                for fileName in fileNames:
                    if fileName.endswith('.tif'):
                        print("Tentar abrir zip " + fileName)

                        try:
                            start = timer()
                            dataset = gdal.Open('/vsizip/%s/%s' % (filename, fileName))
                            end = timer()
                            print("File unziped in: ", end - start, " seconds.")

                        except:
                            if 'dataset' in locals():
                                del dataset
                            print("Erro na abertura do arquivo!")
                            logging.error(gdal.GetLastErrorMsg())
                            raise
        else:
            print("Tentar abrir " + filename)

            try:
                dataset = gdal.Open(filename)
            except:
                if 'dataset' in locals():
                    del dataset
                print("Erro na abertura do arquivo!")
                logging.error(gdal.GetLastErrorMsg())
                raise

        if self.dataset is None:
            self.dataset = dataset

        # no caso da imagem RapidEye, as bandas 5
        # e 3 correspondem às bandas NIR e RED
        banda = dataset.GetRasterBand(band_num)

        # obtencao dos arrays numpy das bandas
        array = banda.ReadAsArray()
        print(array.min(), array.max())

        # Exclude the pixels with no data value and normalize data
        NoData = banda.GetNoDataValue()
        print('NoData:',NoData)
        if NoData == None:
            NoData = array.min()
        
        """...Any pixel with a scaled radiance value of 255 (TM and ETM+) 
           or 65,535 (OLI) - Implications of Pixel Quality Flags on the
           Observation Density of a Continental Landsat Archive
           https://www.mdpi.com/2072-4292/10/10/1570/pdf"""
        #
        print('Data type:',gdal.GetDataTypeName(banda.DataType))
        if gdal.GetDataTypeName(banda.DataType) == 'Byte':
            scale = 255.
        elif gdal.GetDataTypeName(banda.DataType) == 'Int16':
            scale = 10000.
        elif gdal.GetDataTypeName(banda.DataType) == 'UInt16':
            scale = 65535.  # valid_range max
        elif gdal.GetDataTypeName(banda.DataType) == 'Float64':
            scale = 1.  # Nothing to do
        else:
            raise TypeError("band type:", gdal.GetDataTypeName(banda.DataType), "not allowed!")

        
        marray = np.ma.masked_where(array == NoData, array / scale) #When condition tests floating point values for equality, consider using masked_values instead.
        marray = np.ma.masked_invalid(marray) #Mask NAN and Inf

        print(marray.min(), marray.max())
        if gdal.GetDataTypeName(banda.DataType) != 'Float64':
            #np.ma.clip(marray, 0.0, 1.0, out=marray)
            marray = np.ma.clip(marray, 0.0, 1.0)
            print(marray.min(), marray.max())
        
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory usage at get_band_array after open tif is: {0} KB".format(mem))
        print('Object marray:',sys.getsizeof(marray))
        print('Object banda:',sys.getsizeof(banda))
        print('Object dataset:',sys.getsizeof(dataset))

        return marray

        # fechar o dataset e liberar memória
        dataset = None



    def checkBandsForReSSIM(self, **kwargs):

        if self.sensorID == 'LC08':
            listBands = ['Coastal', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Cirrus', 'TIRS1',\
                         'TIRS2', 'MTL']
        elif self.sensorID == 'LE07':
            listBands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS', 'MTL']
        else:
            return False

        for args in listBands:
            if not hasattr(self, args) or getattr(self, args) is None:
                return False
            else:
                output_fmask = os.path.join(kwargs['odir'], self.sat_name + '_fmask.img')

                # Check existence of thematic raster output from Fmask before create temporary directory
                # needed to run Fmask algorithm
                if not os.path.exists(output_fmask):
                    fullname = getattr(self, args)
                    dirname, filename = os.path.split(fullname)

                    mkdir_p(os.path.join(dirname, self.sat_name))

                    if '.TIF' in filename.upper():
                        result = re.search(r'.TIF', filename.upper())
                        prefix = filename.upper()[:result.end() - 4]
                        suffix = '.TIF'
                    else:
                        result = re.search(r'MTL.TXT', filename.upper())
                        prefix = filename.upper()[:result.end() - 4]
                        suffix = '.txt'

                    if is_gz_file(fullname):
                        with gzip.open(fullname, 'rb') as f_in:
                            with open(os.path.join(dirname,self.sat_name,prefix + suffix), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        dest = os.path.join(dirname,self.sat_name,prefix + suffix)

                        if os.path.lexists(dest):
                            os.remove(dest)
                        os.symlink(fullname,dest)

        return True


def FileSensorParser(inputfile: str, **kwargs):
    dirname, filename = os.path.split(inputfile)
    print(filename)

    # ------------------------
    # Process files of CBERS:
    # ------------------------
    if re.search(r'CBERS.*MUX.*_BAND\d.*([.]zip|[.][Tt][Ii][Ff])$',filename):
        logging.info('Processing CBERS MUX file name convention:\n %s' % filename)
        result = re.search(r'_BAND\d+', filename)

        prefix = filename[:result.end() - 1]

        listfiles = glob.glob(os.path.join(dirname, prefix + '[0-9]*'))


    # --------------------------
    # Process files of Landsat:
    # --------------------------
    if re.search(r'_([Bb]\d+|sr_band\d+)', filename):
        logging.info('Processing Landsat file name convention:\n %s' % filename)

        result = re.search(r'_([Bb]\d+|sr_band\d+)', filename)
        len_num = len(result.group()) - \
                  len(''.join(filter(lambda x: not x.isdigit(), result.group())))
        len_word = len(result.group()) - len_num

        prefix = filename[:result.start() + len_word]

        listfiles = glob.glob(os.path.join(dirname, prefix + '[0-9]*'))

        prefix = "_".join(token for token in filename.split("_")[0:7])

        #Add surface reflectance files
        listfiles.extend(glob.glob(os.path.join(dirname, prefix + '_sr_band[0-9]*')))

        # Add metadata files
        listfiles.extend(glob.glob(os.path.join(dirname, prefix + '_MTL.txt*')))

        # Add Quality Accessment band
        listfiles.extend(glob.glob(os.path.join(dirname, prefix + '_BQA*')))


    if 'listfiles' in locals() and len(listfiles) > 1:
        if 'odir' not in kwargs: # define default output directory if needed
            kwargs['odir'] = os.path.join(os.path.split(listfiles[0])[0],'processed_output')

        logging.info(listfiles)
        sensor = Sensor(listfiles)

        try:
            mkdir_p(os.path.join(os.path.split(listfiles[0])[0],'processed_output'))
        except RuntimeError:
            return 1

        attrs = vars(sensor)
        # now dump this in some way or another
        print('\n'.join("%s: %s" % item for item in attrs.items()))

        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory usage before detecting pivots is: {0} KB".format(mem))

        if (sensor.NIR is not None and sensor.Red is not None) or (sensor.NIRSR is not None and sensor.RedSR is not None) :
            try:
                DetectingCenterPivot.FindCenterPivots(sensor, **kwargs)
            except RuntimeError:
                return 1
        else:
            print("WARNING: Error while processing scene:\n" + sensor.sat_name + "\nNot exist bands requested for\
             detecting center pivots")
            logging.warning("Error while processing scene:\n" + sensor.sat_name + "\nNot exist bands requested for\
             detecting center pivots")
            return 1

        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory usage before RESSIM is: {0} KB".format(mem))
        if sensor.checkBandsForReSSIM(**kwargs):  # Check bands and create scene directory needed for Fmask
            try:
                ressim.FindReSSIMtargets(sensor, **kwargs)
            except RuntimeError:
                return 1
        else:
            print("WARNING: Error while processing scene:\n" + sensor.sat_name +
                  "\nNot exist bands requested for ReSSIM module")
            logging.warning("Error while processing scene:\n" + sensor.sat_name +
                            "\nNot exist bands requested for ReSSIM module")



def FolderSensorParser(inputdir: str, **kwargs):
    # -----------------------------------------
    # List files with convention name of CBERS:
    # -----------------------------------------
    ext = "_band"
    prefix = []
    for inputfiles in glob.glob(os.path.join(inputdir, '*_[Bb][Aa][Nn][Dd][0-9]*.zip')):
        namefile = os.path.split(inputfiles)[1]

        if namefile[:namefile.lower().find(ext) + len(ext)] not in prefix:
            FileSensorParser(inputfiles, **kwargs)

            prefix.append(namefile[:namefile.lower().find(ext) + len(ext)])

    # -------------------------------------------
    # List files with convention name of Landsat:
    # -------------------------------------------        
    ext = "_b"
    prefix = []
    for inputfiles in glob.glob(os.path.join(inputdir, '*_[Bb][0-9]*.[Tt][Ii][Ff]*')):
        namefile = os.path.split(inputfiles)[1]
        print(inputfiles[:inputfiles.lower().find(ext) + len(ext)])
        print('Prefix:',prefix)

        if inputfiles[:inputfiles.lower().find(ext) + len(ext)] not in prefix:
            FileSensorParser(inputfiles, **kwargs)

            prefix.append(inputfiles[:inputfiles.lower().find(ext) + len(ext)])
            ext = "_sr_band"
            prefix.append(inputfiles[:inputfiles.lower().find(ext) + len(ext)])

    ext = "_sr_band"
    for inputfiles in glob.glob(os.path.join(inputdir, '*_sr_band[0-9]*.[Tt][Ii][Ff]*')):
        namefile = os.path.split(inputfiles)[1]

        if inputfiles[:inputfiles.lower().find(ext) + len(ext)] not in prefix:
            FileSensorParser(inputfiles, **kwargs)

            prefix.append(inputfiles[:inputfiles.lower().find(ext) + len(ext)])


class Action(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        allowed_keywords = ['epsg_target', 'export_type']
        keyword_dict = {}

        for arg in values:  #values => The args found for keyword_args
            pieces = arg.split('=')

            if len(pieces) == 2 and pieces[0] in allowed_keywords:
                keyword_dict[pieces[0]] = pieces[1]
            else: #raise an error
                #Create error message:
                msg_inserts = ['{}='] * len(allowed_keywords)
                msg_template = 'Example usage: epsg_target=4326 export_type=geojson or shp. Only {} allowed.'.format(', '.join(msg_inserts))
                msg = msg_template.format(*allowed_keywords)

                raise argparse.ArgumentTypeError(msg)

        setattr(namespace, self.dest, keyword_dict) #The dest key specified in the
                                                    #parser gets assigned the keyword_dict--in
                                                    #this case it defaults to 'keyword_args'


def main(argv):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Starting program is: {0} KB".format(mem))

    logging.basicConfig(filename='managementMetadataFromRS.log',\
                        format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',\
                        level=logging.INFO, \
                        filemode = 'w')
    logging.info('Started')
    log = logging.getLogger()

    args_dict = vars(argv)
    keyword_args = args_dict['keyword_args']

    if args_dict['odir'] is not None:
        keyword_args['odir'] = args_dict['odir']  # add output directory to dict Keyword_args

    if 'export_type' not in keyword_args:
        keyword_args['export_type'] = 'geojson' # add default type to export metadata

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage before File parser is: {0} KB".format(mem))

    if argv.i is not None:
        try:
            FileSensorParser(argv.i, **keyword_args)
        except Exception as err:
            log.exception("Error while running file sensor parser:\n %s", err)
            return 1
    else:
        try:
            FolderSensorParser(argv.idir, **keyword_args)
        except Exception as err:
            log.exception("Error while running folder sensor parser\n %s", err)
            return 1
       
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory usage at Finishing program is: {0} KB".format(mem))

    logging.info('Finished')
    return 0


if __name__ == "__main__":

    # Prompt user for (optional) command line arguments, when run from IDLE:
    if 'idlelib' in sys.modules: sys.argv.extend(input("Args: ").split())

    # Process the arguments
    import argparse
    import arghelper

    parser = argparse.ArgumentParser(
        description='Receive and process scenes from various sensors aim to improve metadata.',
        epilog="Please choose only one input options!")

    # Add mutually exclusive group - https://stackoverflow.com/a/44469700
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i',
                       help='Input file of scene, for example /home/images/CBERS_4_MUX_20150723_159_110_L4_BAND5.zip.',
                        metavar='input_file', type=lambda x: arghelper.is_valid_file(parser, x))
    group.add_argument('-idir',
                        help='Directory containing the input files.',
                        metavar='input_dir', type=lambda x: arghelper.is_valid_directory(parser, x))

    parser.add_argument('-odir', nargs='?',
                        help='Directory to store output metadata files. Default is "processed_output" in local input files',
                        metavar='output_dir', type=lambda x: arghelper.is_valid_directory(parser, x))

    # Using argparse with function that takes kwargs argument - https://stackoverflow.com/a/33712815
    parser.add_argument("keyword_args", help="extra args", nargs='*', action=Action)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    sys.exit(main(parser.parse_args()))
