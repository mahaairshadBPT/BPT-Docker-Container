import os, sys, shutil
sys.path.insert(0, './smi')
sys.path.insert(1, './thresholding')
sys.path.insert(2, './utils')
sys.path.insert(3, './postprocessing')

from smi_estimation import SMIEstimation
from thresholding_model import threshold
from colored_tifs import colorTifs

import rasterio
import richdem as rd
import geopandas as gpd
import argparse

src_ndvi = rasterio.open('./input/NDVI.tif')
src_therm = rasterio.open('./input/therm.tif')
holes_outlines = gpd.read_file('./input/holes_outlines/holes_outlines.shp')
holes_outlines['Hole'] = holes_outlines['Hole'].astype(int)

os.makedirs('tmp', exist_ok=True) # used by thresholding and clustering models

shutil.rmtree('output/smi', ignore_errors=True)
shutil.rmtree('output/thresholding', ignore_errors=True)

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', nargs='+', choices=['smi','thresholding'], default=['smi','thresholding'])
args = parser.parse_args()


smi = True if 'smi' in args.analysis else False
thresholding = True if 'thresholding' in args.analysis else False

if smi == True:
    print('Running SMI Model')
    os.makedirs('output/smi', exist_ok=True)
    # SMI Model
    ##smi_model = SMIEstimation('./input/holes_outlines/', './input', outline='holes_outlines')
    smi_model = SMIEstimation('./input/holes_outlines/', './input', outline='holes_outlines')
    smi_model.save_smi_raster('./output/smi')


if thresholding == True:
    print('Running Thresholding Model')
    os.makedirs('output/thresholding', exist_ok=True)
    # Thresholding Model
    hole_numbers = holes_outlines['Hole'].tolist()
    threshold(src_ndvi, src_therm, hole_numbers)

os.makedirs('output/postproc', exist_ok=True)
colorTifs()
