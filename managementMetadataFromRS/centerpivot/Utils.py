# coding: utf-8

# --------------------------
#        Imports
# --------------------------
from osgeo import ogr
from managementMetadataFromRS.Utils import mkdir_p
import os
import json

# https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html#count-geometries-in-a-geometry
# another example (Buffer Creation): https://gist.github.com/jgomezdans/808194


def create_geoCollection(coords):
    # Create a geometry collection
    geomcol = ogr.Geometry(ogr.wkbGeometryCollection)
    
    for cont,coord in enumerate(coords):
        wkt = "POINT ("+str(coord[0])+" "+str(coord[1])+")"
        pt = ogr.CreateGeometryFromWkt(wkt)
        # Add a point
        geomcol.AddGeometry(pt)

        bufferDistance = coord[2]
        poly = pt.Buffer(bufferDistance)
        #print("%s buffered by %d is %s" % (pt.ExportToWkt(), bufferDistance, poly.ExportToWkt()))
        # Add a polygon
        geomcol.AddGeometry(poly)

    # print(geomcol.ExportToWkt())
    print("Geometry Collection has %i geometries" % (geomcol.GetGeometryCount()))

    return geomcol


def create_geoPolygon(coords):
    # Add the points to the ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in coords:
        lon = point[0]
        lat = point[1]
        ring.AddPoint(lon, lat)

    # Add first point again to ring to close polygon
    ring.AddPoint(coords[0][0], coords[0][1])

    # Add the ring to the polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    print(poly.ExportToWkt())

    return poly


"""
The ESRI Shapefile driver supports one geometry type per layer, and this cannot be a geometry collection type.

So it would look something like this:

# this will create a directory with 1 or more shapefiles
dst = driver.CreateDataSource("common-border")
# create common-border/points.shp
pointLayer = dst.CreateLayer("points", spatialReference, ogr.wkbPoint)
# create common-border/polygons.shp
polygonLayer = dst.CreateLayer("polygons", spatialReference, ogr.wkbPolygon)
# ... as needed
dst.GetLayerCount()  # 2 layers / shapefiles

And you would need to add each geometry type to the appropriate layer. There are 
probably some really smart things you can do with dictionaries to do the mapping, 
and if a key does not exist for the geometry type, it could create the required 
layer, etc.
"""


def write_shapefile(geoCollection, spatialRef, geomType, out_shp):
    """
    Adapted from:
    https://gis.stackexchange.com/a/52708/8104
    """
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('ESRI Shapefile')

    if os.path.exists(out_shp):
        driver.DeleteDataSource(out_shp)
    else:
        mkdir_p(os.path.split(out_shp)[0])

    ds = driver.CreateDataSource(out_shp)

    layer = ds.CreateLayer('',
                           geom_type=geomType,
                           srs=spatialRef)

    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # If there are multiple geometries, put the "for" loop here
    if geoCollection.GetGeometryCount() > 1:
        for id, geom in enumerate(geoCollection):
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            feat.SetField('id', id)

            # Set new geometry
            feat.SetGeometry(geom)

            # Add new feature to output Layer
            layer.CreateFeature(feat)

            feat = geom = None  # destroy these
    else:
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', 1)

        # Set new geometry
        feat.SetGeometry(geoCollection)

        # Add new feature to output Layer
        layer.CreateFeature(feat)

        feat = None  # destroy

    # Save and close everything
    ds = layer = feat = geom = None


# https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
def write_geojson(geoCollection, spatialRef, geomType, out_json):

    # Now convert it to a GeoJSON with OGR
    driver = ogr.GetDriverByName('GeoJSON')

    add_metadata = False
    if os.path.exists(out_json):
        geo_objects = json.load(open(out_json))
        if 'properties' in geo_objects and 'bbox' in geo_objects:
            add_metadata = True
        driver.DeleteDataSource(out_json)
    else:
        mkdir_p(os.path.split(out_json)[0])

    ds = driver.CreateDataSource(out_json)

    layer = ds.CreateLayer('',
                           geom_type=geomType,
                           srs=spatialRef)
    #print('Create Layer - geom_type:',geomType,' SRS:',spatialRef)

    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # If there are multiple geometries, put the "for" loop here
    for id, geom in enumerate(geoCollection):
        if geom.GetGeometryType() == ogr.wkbPolygon:
            # Create a new feature (attribute and geometry)
            feat = ogr.Feature(defn)
            feat.SetField('id', id)
            feat.SetGeometry(geom)
            
            layer.CreateFeature(feat)
            feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None

    #Append metadata info:
    if add_metadata:
        geo_objects_tmp = json.load(open(out_json))
        geo_objects_tmp['properties'] = geo_objects['properties']
        geo_objects_tmp['bbox'] = geo_objects['bbox']
        with open(out_json, 'w') as outfile:
            json.dump(geo_objects_tmp, outfile)

    return
