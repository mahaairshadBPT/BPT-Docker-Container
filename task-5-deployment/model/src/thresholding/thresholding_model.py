
#!pip install rasterio fiona geopandas

# Geospatial raster packages
import fiona
import rasterio as rio
from rasterio import mask as riomask
from rasterio import plot as rioplot
from rasterio import sample as riosample
from rasterio import transform as riotransform
from rasterio.features import rasterize
from rasterio.plot import show
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import os

# Arrays and DataFrames
import numpy as np
from PIL import Image

import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, LineString, box

import sklearn
from sklearn import preprocessing

from osgeo import gdal, ogr
import sys
import shutil
import re
import glob

from scipy import stats
#from google.colab import files

import subprocess

import warnings

Hole_F = dict({
    'F1' : [2, 3, 4, 5, 6, 7, 8],
    'F2' : [1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
})

"""# Rescaling raster"""

from sklearn.preprocessing import MinMaxScaler

def test_resample_rescale_pixel_vals(raster, localname):
    import matplotlib.pyplot as plt
    from osgeo import osr
    fn = rio.open(localname)
    data_array = fn.read(1)

    data_array = data_array[::-1]
    data_array_scaled = np.interp(data_array, (data_array.min(), data_array.max()), (0, 255))
    data_array_scaled[data_array_scaled == 0] = 128

    r = data_array_scaled

    show(r)

    RES    = 1
    WIDTH  = data_array_scaled.shape[1]
    HEIGHT = data_array_scaled.shape[0]

    output_file = "./output/thresholding/rgbOut.tif"

    # Create GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_file, WIDTH, HEIGHT, 1, gdal.GDT_Byte)

    # Upper Left x, Eeast-West px resolution, rotation, Upper Left y, rotation, North-South px resolution
    dst_ds.SetGeoTransform( [ -180, 1, 0, 90, 0, -1 ] )

    # Set CRS
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    dst_ds.SetProjection( srs.ExportToWkt() )

    # Write the band
    dst_ds.GetRasterBand(1).SetNoDataValue(128) #optional if no-data transparent
    dst_ds.GetRasterBand(1).WriteArray(r)


    band = dst_ds.GetRasterBand(1)
    colors = gdal.ColorTable()

    colors.SetColorEntry(128, (255, 255, 255)) #WHITE
    colors.CreateColorRamp(0, (255, 0, 0), 63, (255, 165, 0)) #RED ORANGE
    colors.CreateColorRamp(63, (255, 165, 0), 127, (255, 255, 0)) #ORANGE YELLOW
    colors.CreateColorRamp(129, (255, 255, 0), 190, (144, 238, 144)) #YELLOW LYTGREEN
    colors.CreateColorRamp(190, (144, 238, 144), 255, (0, 128, 0)) #LYTGREEN GREEN

    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    #del band, dst_ds

def resample_rescale_pixel_vals(raster, localname, downsample=True, rescale=None):
    t = raster.transform
    scale = 1/4

    # rescale the metadata
    #transform = Affine(t.a, t.b, t.c, t.d, t.e, t.f)
    #profile = raster.profile.copy()


    #print('Raster shape:', raster.shape)

    if downsample:
      print(f"\nDownsampling for {localname}...")
      transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
      height = int(raster.height * scale)
      width = int(raster.width * scale)
      data = raster.read(1,
            out_shape=(raster.count, height, width),
            resampling=Resampling.bilinear
              )

    else:
      transform = Affine(t.a, t.b, t.c, t.d, t.e, t.f)
      height = int(raster.height)
      width = int(raster.width)
      data = raster.read(1)

    profile = raster.profile.copy()
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    #data = raster.read(1)
    if rescale!=None:
      #print('Rescaling to ',rescale)
      scaled_data = sklearn.preprocessing.minmax_scale (data.reshape(len(data[:,0]) * len(data[0,:])), feature_range = (rescale[0], rescale[1]))
      scaled_data = scaled_data.reshape((len(data[:,0]),len(data[0,:])))

    else:
      scaled_data = data.copy()

    #print('Resampled shape:', scaled_data.shape)

    with rio.open(localname, 'w', **profile) as dst:
      dst.write_band(1, scaled_data)
    return np.min(data), np.max(data)

def invScale(X_scld, datamin, datamax, layer):
  if layer == 'ndvi':
    sclmin, sclmax = 0, 30
  else: #therm
    sclmin, sclmax = 0, 800
  X_orig = (((X_scld-sclmin)/(sclmax-sclmin))*(datamax-datamin)) + datamin
  return X_orig

"""# File and dir"""



def makeDir(newdir):
  try:
    os.mkdir(newdir)
  except FileExistsError:
    print('Directory {} already exists'.format(newdir))
  else:
    print('Directory {} created'.format(newdir))

def deleteExistingFile(file_path):
  try:
    os.remove(file_path)
  except OSError:
    pass

"""# Masking"""

def mask_raster(raster_path, shape_path, mask_hole, out_path):
  print('\n'+ ('-'*100) +'\n'+(" "*46)+'Hole '+str(mask_hole)+'\n'+ ('-'*100) +'\n') #Section display
  print(f"\nMasking {mask_hole} fairway(s)...\n")
  in_shape = gpd.read_file(shape_path)
  in_shape = in_shape.rename({'Hole':'hole'}, axis=1)

  shapefile = in_shape

  try:
    with rio.open(raster_path) as src:
        if mask_hole!='all':
          out_image, out_transform = rio.mask.mask(src, shapefile[shapefile['hole']==mask_hole].geometry, crop=True, filled=True, nodata=np.nan)
          #out_image, out_transform = rio.mask.mask(src, in_shape[in_shape['hole']==mask_hole].geometry, crop=True, filled=True, nodata=np.nan)
        else:
          out_image, out_transform = rio.mask.mask(src, shapefile.geometry, crop=True, filled=True, nodata=np.nan)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    deleteExistingFile(out_path)
    with rio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)
  except:
      print(f'Fairway(s) {mask_hole} does not overlap with raster for the date.')
  return out_image

"""# Polygonization"""

def polygonize(in_path, out_path):
  print("\nPolygonizing...\n")
  deleteExistingFile(out_path)
  # Polygonization
  script = './utils/gdal_polygonize.py'
  subprocess.call(["python", script, in_path, out_path, '-b','1', '-f','GPKG','OUTPUT', 'DN'])
  shapefile = gpd.read_file(out_path, layer='OUTPUT')
  return shapefile

"""# Threshold selection"""

def findThreshold(pixelVals, layer, holename, filename, minval, maxval):
  print(f"Finding Threshold for {layer}...")
  arr = np.delete(pixelVals, np.where((pixelVals<=0)|(pixelVals>100))) # Dropping out of bound values
  if layer=='ndvi':
    arr = np.delete(arr, np.where(arr==15)) # ndvi is 15 where no image exists within fairway

  arr = np.around(arr)

  mode, med, q2, q1, q3 = stats.mode(arr)[0][0], np.median(arr), np.quantile(arr, .50), np.quantile(arr, .25), np.quantile(arr, .75)
  out = q1 - (1.5 * (q3 - q1))
  t1 = q3 # threshold unhealthy (ndvi)
  t2 = q3 # threshold waterstress (therm)
  t3 = np.mean(arr) # threshold waterlogged (therm)

  # Calculate the histogram data
  hist_array, bin_array = np.histogram(arr, bins='auto', density=False)

  # Finding coolest valid temperature in the Hole
  for pix in hist_array:
    if pix>=20:
      h_min = pix
      break

  l = np.argwhere(hist_array == h_min)
  loc = l[0,0]
  min_val = np.around(bin_array[loc])

  print(f"\nAvg {layer} (orig scale): ", np.around(invScale(np.mean(arr), minval, maxval, layer), 2))
  print(f'Minimum {layer} (orig scale): ', np.around(invScale(min_val, minval, maxval, layer), 2))

  if(layer=='ndvi'):
    print(f"\nThresh: Less than {invScale(t1, minval, maxval, layer)} is Unhealthy\n")
    return t1

  if(layer=='therm'):
    print(f"\nThresh: Greater than {np.around(invScale(t2, minval, maxval, layer), 2)}  deg C is Very Hot (waterstressed)")
    print(f"Thresh: Less than equals to {np.around(invScale(t3, minval, maxval, layer), 2)}  deg C is Very Cool (waterlogged)\n")
    return t2, t3

"""# Shapefile generation"""

def shp_to_tif(shape_path, out_path):
  df = gpd.read_file(shape_path)

  # resolution
  shape = 1000, 1000

  transform = rio.transform.from_bounds(*df['geometry'].total_bounds, *shape)
  rasterize_rivernet = rasterize(
      [(shape, 1) for shape in df['geometry']],
      out_shape=shape,
      transform=transform,
      fill=np.nan,
      all_touched=True,
      dtype='float32')

  with rio.open(
      out_path, 'w',
      driver='GTiff',
      dtype='float32',
      count=1,
      width=shape[0],
      height=shape[1],
      transform=transform,
      nodata=np.nan,
  ) as dst:
      dst.write(rasterize_rivernet, indexes=1)

def get_shpfile_ndvi(minndvi =-1, maxndvi=1, mask_hole = 'all'):
  #mask_hole = 7 #'all or hole number
  layer = 'ndvi' #'ndvi', 'ndre' or 'therm'

  # Mask fairway(s) of interest
  in_raster = f"./tmp/scaled/scaled_{layer}.tif"
  in_shape = './input/holes_outlines/holes_outlines.shp'
  out_mask_path = f'./tmp/masked/masked_hole_{mask_hole}_{layer}.tif'

  # Drop NaN pixel values (out of mask)
  out_mask = mask_raster(in_raster, in_shape, mask_hole, out_mask_path)
  ndvi = np.delete(out_mask,np.where(np.isnan(out_mask.reshape(out_mask.shape[1]*out_mask.shape[2]))))
  #print(ndvi.shape)

  # Get and visualize threshold
  t1 = findThreshold(ndvi, 'ndvi', mask_hole, out_mask_path, minndvi, maxndvi)

  # Polygonize (Raster to vector)
  out_vect_path = f'./tmp/polygonized/polygonized_{mask_hole}_{layer}.gpkg'
  gdf = polygonize(out_mask_path, out_vect_path)

  # Make selection of regions within threshold range
  gdf = gdf[(gdf['DN']>=0) & (gdf['DN']<t1) & (gdf['DN']!=15)]

  print("\nExporting ",len(gdf) ," geometries within threshold range...")
  out_path = f"./output/thresholding/Unhealthy_ndvi"
  deleteExistingFile(out_path+".gpkg")
  gdf.to_file(out_path+".gpkg", driver="GPKG")
  shp_to_tif(out_path+".gpkg", out_path+'.tif')

def shpsIntersectionWL(wl_therm, mask_hole):
  shapefile = gpd.read_file('./input/holes_outlines/holes_outlines.shp')
  shapefile = shapefile.rename({'Hole':'hole'}, axis=1)
  hole_shp = shapefile[shapefile['hole']==mask_hole]
  unhlt_gdf = gpd.read_file(f'./output/thresholding/Unhealthy_ndvi.gpkg', mask=hole_shp)
  DEM_shp = gpd.read_file('./thresholding/DEM-waterlogging prone.gpkg', mask=hole_shp)

  intersection1_gdf = gpd.GeoDataFrame(columns=['DN','geometry'])
  intersection2_gdf = gpd.GeoDataFrame(columns=['DN','geometry'])

  # dissolve the boundaries by region for smooth intersection operation
  unhlt_gdf = (unhlt_gdf.dissolve()).reset_index()
  wl_gdf = (wl_therm.dissolve()).reset_index()
  DEM_shp = (DEM_shp.dissolve()).reset_index()

  for index, orig in unhlt_gdf.iterrows():
    for index2, ref in wl_gdf.iterrows():
      if ref['geometry'].intersects(orig['geometry']):
        df1 = {'DN': orig['DN'], 'geometry' : ref['geometry'].intersection(orig['geometry'])}
        intersection1_gdf = intersection1_gdf.append(df1, ignore_index = True)

  for index, orig in intersection1_gdf.iterrows():
    for index2, ref in DEM_shp.iterrows():
        df2 = {'DN': orig['DN'], 'geometry' : ref['geometry'].intersection(orig['geometry'])}
        intersection2_gdf = intersection2_gdf.append(df2, ignore_index = True)
  return intersection2_gdf

def get_shpfile_therm(mask_hole, mintemp=0, maxtemp=382.35, display = True):
  #mask_hole = 7 #'all or hole number
  layer = 'therm' #'ndvi', 'ndre' or 'therm'

  # Mask fairway(s) of interest
  in_raster = f"./tmp/scaled/scaled_{layer}.tif"
  in_shape = './input/holes_outlines/holes_outlines.shp'
  out_mask_path = f'./tmp/masked/masked_hole_{mask_hole}_{layer}.tif'

  # Drop NaN pixel values (out of mask)
  out_mask = mask_raster(in_raster, in_shape, mask_hole, out_mask_path)
  therm = np.delete(out_mask,np.where(np.isnan(out_mask.reshape(out_mask.shape[1]*out_mask.shape[2]))))
  #print(therm.shape)

  # Get and visualize threshold
  t2, t3 = findThreshold(therm, 'therm', mask_hole, out_mask_path, mintemp, maxtemp)

  # Polygonize (Raster to vector)
  out_vect_path = f'./tmp/polygonized/polygonized_{mask_hole}_{layer}.gpkg'
  gdf = polygonize(out_mask_path, out_vect_path)

  # Make thermal selection of regions within threshold range
  gdf_ws = gdf[(gdf['DN']>t2) & (gdf['DN']<100)]
  gdf_wl_therm = gdf[(gdf['DN']>=0) & (gdf['DN']<=t3)]

  # Apply intersection between WL prone ndvi, therm and dem
  gdf_wl = shpsIntersectionWL(gdf_wl_therm, mask_hole)

  print("\nExporting geometries within threshold range...")

  Waterstressdir = f'Waterstress_{mask_hole}'
  Waterloggeddir = f'Waterlogged_{mask_hole}'
  makeDir(os.path.join('./tmp/outputShpfiles','Waterstress'))
  makeDir(os.path.join('./tmp/outputShpfiles','Waterlogged'))
  out_path_ws = f'./tmp/outputShpfiles/Waterstress/{Waterstressdir}'
  out_path_wl = f'./tmp/outputShpfiles/Waterlogged/{Waterloggeddir}'
  deleteExistingFile(out_path_ws+".gpkg")
  deleteExistingFile(out_path_wl+".gpkg")
  gdf_ws.to_file(out_path_ws+".gpkg", driver="GPKG")
  gdf_wl.to_file(out_path_wl+".gpkg", driver="GPKG")

"""## Creating single shapefile for therm labels under date"""

def createSingleFileTherm():
    # To create a single shapefile for labels from Thermal
    filesToConcat = ['Waterstress', 'Waterlogged']

    for fileNames in filesToConcat:

        file = os.listdir(f'./tmp/outputShpfiles/{fileNames}')
        out_path = f'./output/thresholding/{fileNames}'
        path = [os.path.join(f'./tmp/outputShpfiles/{fileNames}', i) for i in file if ".gpkg" in i]

        gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in path],
                              ignore_index=True), crs=gpd.read_file(path[0]).crs)
        deleteExistingFile(out_path+'.gpkg')
        gdf.to_file(out_path+'.gpkg')
        shp_to_tif(out_path+'.gpkg', out_path+'.tif')


"""# Main"""

def threshold(src_ndvi, src_therm, hole_numbers):
    makeDir(os.path.join('./tmp','scaled'))
    makeDir(os.path.join('./tmp','masked'))
    makeDir(os.path.join('./tmp','polygonized'))
    makeDir(os.path.join('./tmp','outputShpfiles'))


    warnings.filterwarnings("ignore") # Suppress warning messages

    ###test_resample_rescale_pixel_vals(src_ndvi, './input/ndvi.tif')



    #src_ndvi = rio.open('./input/ndvi.tif') #INPUT arg
    #src_therm = rio.open('./input/therm.tif') #INPUT arg

    # Create rescaled images
    minndvi, maxndvi = resample_rescale_pixel_vals(src_ndvi, f"./tmp/scaled/scaled_ndvi.tif",downsample=True, rescale=[0,30])#from -1 to 1 (nodata = 0)
    mintemp, maxtemp = resample_rescale_pixel_vals(src_therm, f"./tmp/scaled/scaled_therm.tif",downsample=True, rescale=[0,800])#from (something>0) to 382.35 (nodata = 382.35)

    # Create shapefiles for NDVI and Therm
    get_shpfile_ndvi(minndvi, maxndvi, 'all')

    for holeNum in hole_numbers:
        try:
            get_shpfile_therm(holeNum, mintemp, maxtemp)
        except:
            print('\n'+ ('! '*35))
            print(f'Fairway no.{holeNum} does not overlap with raster for the date.')
            print(('! '*35)+'\n')

    createSingleFileTherm()
