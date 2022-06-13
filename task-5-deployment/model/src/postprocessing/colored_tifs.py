#!pip install rasterio fiona geopandas


# Geospatial raster packages
#import fiona
#import rasterio as rio
from osgeo import gdal, ogr, osr

# Arrays and DataFrames
import numpy as np
import geopandas as gpd

"""NDVI to RGB"""

def NDVIcolor(ndvi):
  min = -1
  max = 1
  color = 255 * ((ndvi-min) / (max-min))
  return int(np.round(color))

def NDVItoRGB(tiff_file, filename):
  #2.
  geotransform = tiff_file.GetGeoTransform()
  projection = tiff_file.GetProjection()
  band = tiff_file.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize

  #3.
  array = tiff_file.GetRasterBand(1).ReadAsArray()

  #4.
  print(array.min())
  print(array.max())
  array = np.interp(array, (array.min(), array.max()), (0, 255))

  #5.
  driver = gdal.GetDriverByName('GTiff')
  new_tiff = driver.Create(filename,xsize,ysize,1,gdal.GDT_Byte)
  new_tiff.SetGeoTransform(geotransform)
  new_tiff.SetProjection(projection)
  #new_tiff.GetRasterBand(1).SetNoDataValue(128) #optional if no-data transparent
  new_tiff.GetRasterBand(1).WriteArray(array)

  band = new_tiff.GetRasterBand(1)
  colors = gdal.ColorTable()

  #colors.SetColorEntry(128, (255, 255, 255)) #WHITE
  colors.CreateColorRamp(NDVIcolor(-1), (215, 25, 28), NDVIcolor(0.4), (215, 25, 28))
  colors.CreateColorRamp(NDVIcolor(0.4), (215, 25, 28), NDVIcolor(0.525), (253, 174, 97))
  colors.CreateColorRamp(NDVIcolor(0.525), (253, 174, 97), NDVIcolor(0.65), (255, 255, 192))
  colors.CreateColorRamp(NDVIcolor(0.65), (255, 255, 192), NDVIcolor(0.775), (166, 217, 106))
  colors.CreateColorRamp(NDVIcolor(0.775), (166, 217, 106), NDVIcolor(0.9), (26, 150, 65))
  colors.CreateColorRamp(NDVIcolor(0.9), (26, 150, 65), NDVIcolor(1), (26, 150, 65))

  band.SetRasterColorTable(colors)
  band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

  #new_tiff.FlushCache() #Saves to disk
  #new_tiff = None #closes the file

"""NDRE to RGB"""

def NDREcolor(ndre):
  min = -1
  max = 1
  color = 255 * ((ndre-min) / (max-min))
  return int(np.round(color))

def NDREtoRGB(tiff_file, filename):
  #2.
  geotransform = tiff_file.GetGeoTransform()
  projection = tiff_file.GetProjection()
  band = tiff_file.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize

  #3.
  array = tiff_file.GetRasterBand(1).ReadAsArray()

  #4.
  print(array.min())
  print(array.max())
  array = np.interp(array, (array.min(), array.max()), (0, 255))

  #5.
  driver = gdal.GetDriverByName('GTiff')
  new_tiff = driver.Create(filename,xsize,ysize,1,gdal.GDT_Byte)
  new_tiff.SetGeoTransform(geotransform)
  new_tiff.SetProjection(projection)
  #new_tiff.GetRasterBand(1).SetNoDataValue(128) #optional if no-data transparent
  new_tiff.GetRasterBand(1).WriteArray(array)

  band = new_tiff.GetRasterBand(1)
  colors = gdal.ColorTable()

  #colors.SetColorEntry(128, (255, 255, 255)) #WHITE
  colors.CreateColorRamp(NDREcolor(-1), (215, 25, 28), NDREcolor(0), (215, 25, 28))
  colors.CreateColorRamp(NDREcolor(0), (215, 25, 28), NDREcolor(0.15), (253, 174, 97))
  colors.CreateColorRamp(NDREcolor(0.15), (253, 174, 97), NDREcolor(0.3), (255, 255, 192))
  colors.CreateColorRamp(NDREcolor(0.3), (255, 255, 192), NDREcolor(0.45), (166, 217, 106))
  colors.CreateColorRamp(NDREcolor(0.45), (166, 217, 106), NDREcolor(0.6), (26, 150, 65))
  colors.CreateColorRamp(NDREcolor(0.6), (26, 150, 65), NDREcolor(1), (26, 150, 65))

  band.SetRasterColorTable(colors)
  band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

  #new_tiff.FlushCache() #Saves to disk
  #new_tiff = None #closes the file

"""Thermal to RGB"""

def THERMcolor(therm, min, max):
  color = 255 * ((therm-min) / (max-min))
  print(int(np.round(color)))
  return int(np.round(color))

def THERMtoRGB(tiff_file, filename):
  #2.
  geotransform = tiff_file.GetGeoTransform()
  projection = tiff_file.GetProjection()
  band = tiff_file.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize

  #3.
  array = tiff_file.GetRasterBand(1).ReadAsArray()
  array[array>60] = 60

  #4.
  m = array.min()
  x = array.max()
  print(m)
  print(x)
  array = np.interp(array, (m, x), (0, 255))

  #5.
  driver = gdal.GetDriverByName('GTiff')
  new_tiff = driver.Create(filename,xsize,ysize,1,gdal.GDT_Byte)
  new_tiff.SetGeoTransform(geotransform)
  new_tiff.SetProjection(projection)
  #new_tiff.GetRasterBand(1).SetNoDataValue(128) #optional if no-data transparent
  new_tiff.GetRasterBand(1).WriteArray(array)

  band = new_tiff.GetRasterBand(1)
  colors = gdal.ColorTable()

  #a, b, c, d, e, f, g = (m, m+3, m+5, m+15, m+20, m+25, x)
  a, b, c, d, e, f, g = (m, m+3, m+5, m+15, m+20, m+25, x)

  #colors.SetColorEntry(128, (255, 255, 255)) #WHITE
  colors.CreateColorRamp(THERMcolor(a, m, x), (43, 131, 186), THERMcolor(b, m, x), (43, 131, 186))
  colors.CreateColorRamp(THERMcolor(b, m, x), (43, 131, 186), THERMcolor(c, m, x), (171, 221, 164))
  colors.CreateColorRamp(THERMcolor(c, m, x), (171, 221, 164), THERMcolor(d, m, x), (255, 255, 191))
  colors.CreateColorRamp(THERMcolor(d, m, x), (255, 255, 191), THERMcolor(e, m, x), (253, 174, 97))
  colors.CreateColorRamp(THERMcolor(e, m, x), (253, 174, 97), THERMcolor(f, m, x), (215, 25, 28))
  colors.CreateColorRamp(THERMcolor(f, m, x), (215, 25, 28), THERMcolor(g, m, x), (215, 25, 28))

  band.SetRasterColorTable(colors)
  band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

  #new_tiff.FlushCache() #Saves to disk
  #new_tiff = None #closes the file

"""SMI to RGB"""

def SMIcolor(smi):
  imin = 0
  imax = 1
  color = 255 * ((smi-imin) / (imax-imin))
  print(int(np.round(color)))
  return int(np.round(color))

def SMItoRGB(tiff_file, filename):
  #2.
  geotransform = tiff_file.GetGeoTransform()
  projection = tiff_file.GetProjection()
  band = tiff_file.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize

  #3.
  array = tiff_file.GetRasterBand(1).ReadAsArray()
  #a = np.where(~np.isnan(array))
  array[np.isnan(array)] = 0
  print(array)

  #4.
  m = array.min()
  x = array.max()
  print(m)
  print(x)
  array = np.interp(array, (m, x), (0, 255))


  #5.
  driver = gdal.GetDriverByName('GTiff')
  new_tiff = driver.Create(filename,xsize,ysize,1,gdal.GDT_Byte)
  new_tiff.SetGeoTransform(geotransform)
  new_tiff.SetProjection(projection)
  #new_tiff.GetRasterBand(1).SetNoDataValue(128) #optional if no-data transparent
  new_tiff.GetRasterBand(1).WriteArray(array)

  band = new_tiff.GetRasterBand(1)
  colors = gdal.ColorTable()

  a, b, c, d, e, f, g = (0, 0.16, 0.33, 0.49, 0.66, 0.831, 1)
  #a, b, c, d, e, f, g = (0, 0.05, 0.125, 0.2, 0.275, 0.35, 1)

  #colors.SetColorEntry(128, (255, 255, 255)) #WHITE

  colors.CreateColorRamp(SMIcolor(a), (228,1,13), SMIcolor(b), (253, 148, 78))
  colors.CreateColorRamp(SMIcolor(b), (253, 148, 78), SMIcolor(c), (250, 233, 107))
  colors.CreateColorRamp(SMIcolor(c), (250, 233, 107), SMIcolor(d), (229, 230, 0))
  colors.CreateColorRamp(SMIcolor(d), (229, 230, 0), SMIcolor(e), (176, 191, 20))
  colors.CreateColorRamp(SMIcolor(e), (176, 191, 20), SMIcolor(f), (8,134,2))
  colors.CreateColorRamp(SMIcolor(f), (8,134,2), SMIcolor(g), (8,134,2))


  band.SetRasterColorTable(colors)
  band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

  #new_tiff.FlushCache() #Saves to disk
  #new_tiff = None #closes the file

"""Salinity to RGB"""

def SALINXcolor(salin, min, max):
  color = 255 * ((salin-min) / (max-min))
  print(int(np.round(color)))
  return int(np.round(color))

def SALINtoRGB(tiff_file, filename):
  #2.
  geotransform = tiff_file.GetGeoTransform()
  projection = tiff_file.GetProjection()
  band = tiff_file.GetRasterBand(1)
  xsize = band.XSize
  ysize = band.YSize

  #3.
  array = tiff_file.GetRasterBand(1).ReadAsArray()
  array[array>5] = 5
  array = np.around(array, 1)

  #4.
  m = array.min()
  x = array.max()
  print(m)
  print(x)
  array = np.interp(array, (m, x), (0, 255))

  #5.
  driver = gdal.GetDriverByName('GTiff')
  new_tiff = driver.Create(filename,xsize,ysize,1,gdal.GDT_Byte)
  new_tiff.SetGeoTransform(geotransform)
  new_tiff.SetProjection(projection)
  #new_tiff.GetRasterBand(1).SetNoDataValue(128) #optional if no-data transparent
  new_tiff.GetRasterBand(1).WriteArray(array)

  band = new_tiff.GetRasterBand(1)
  colors = gdal.ColorTable()

  a, b, c, d, e, f = (m, m+0.5, m+1, m+1.5, m+2.5, x)
  #a, b, c, d, e, f = (m, m+10000, m+50000, m+100000, m+200000, x)

  #colors.SetColorEntry(128, (255, 255, 255)) #WHITE
  colors.CreateColorRamp(SALINXcolor(a, m, x), (158, 202, 225), SALINXcolor(b, m, x), (158, 202, 225))
  colors.CreateColorRamp(SALINXcolor(b, m, x), (100, 161, 214), SALINXcolor(c, m, x), (100, 161, 214))
  colors.CreateColorRamp(SALINXcolor(c, m, x), (33, 113, 181), SALINXcolor(d, m, x), (33, 113, 181))
  colors.CreateColorRamp(SALINXcolor(d, m, x), (8, 81, 156), SALINXcolor(e, m, x), (8, 81, 156))
  colors.CreateColorRamp(SALINXcolor(e, m, x), (8, 48, 107), SALINXcolor(f, m, x), (8, 48, 107))

  band.SetRasterColorTable(colors)
  band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

  #new_tiff.FlushCache() #Saves to disk
  #new_tiff = None #closes the file

"""# Main"""

def colorTifs():
    tiff_file = gdal.Open('./input/NDVI.tif')
    NDVItoRGB(tiff_file, './output/postproc/coloredNDVI.tif')

    tiff_file = gdal.Open('./input/therm.tif')
    THERMtoRGB(tiff_file, './output/postproc/coloredTherm.tif')

    tiff_file = gdal.Open('./output/smi/smi.tif')
    SMItoRGB(tiff_file, './output/postproc/coloredSMI.tif')

    tiff_file = gdal.Open('./input/salin.tif')
    SALINtoRGB(tiff_file, './output/postproc/coloredSalin.tif')
