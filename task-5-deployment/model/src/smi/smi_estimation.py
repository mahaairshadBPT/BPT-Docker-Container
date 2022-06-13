import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import os

from raster_loader import RasterLoader
from polygon_loader import PolygonLoader
from envelope_detection import EnvelopeDetection


class SMIEstimation:
    def __init__(self, source_path_outlines, source_path_rasters, outline=None, reduction=9):
        self.date = ''
        self.field = ''
        self.outline = outline
        self.source_path_outlines = source_path_outlines
        self.source_path_rasters = source_path_rasters

        # (1-2) load the rasters, and crop ROI
        self.ndvi, self.therm = self._load_the_rasters()

        # (3) Smoothen and downsample the rasters
        # TODO: need to investigate ndvi downsampling vs nir and red downsampling --> define an experiment
        self.ndvi_downsampled, self.therm_downsampled = self._downsample_rasters(method='Average', reduction=reduction)

        # (5) Determine Wet and Dry edges
        # TODO: for now hardcoded edges !!
        #  (1) can be done manually: e.g. easily through interactive plot
        #  (2) or (later), through an algorithm
        self._get_wet_dry_edges()


        # (6) Calculate SMI
        # TODO: do we want SMI to be calculated on the original or downsampled raster? Could be both, and then whatever
        #  is needed, e.g. for validation can then pick what is suitable.
        self.smi = self._get_smi()

    def _load_the_rasters(self):
        # Load the outlines, shape files (only once)
        outline_shapes = PolygonLoader(path=self.source_path_outlines, outline=self.outline)

        # Load the rasters for several bands, for a certain date, and possibly cropped based on an outline
        ndvi = RasterLoader(band='ndvi',
                            outline=outline_shapes.get_outlines(),
                            crop_data=False, base_path=self.source_path_rasters)
        therm = RasterLoader(band='therm',
                             outline=outline_shapes.get_outlines(),
                             crop_data=False, base_path=self.source_path_rasters)
        return ndvi, therm

    def _downsample_rasters(self, method='Average', reduction=9):#method Gaussian or Average
        """Now, downsample the data

        Why?
        * Why is smoothing useful? Well, for the wet/dry edgesnalysis, we are combining values from different bands to make
        the scatter (so ndvi vs thermal). Sensors are by nature noisy, also the response from the soil/vegetation may have
        some variance, and also the rasters we get are stitched from individual drone images, so also there some noise can
        be introduced in this process --> so instead of plotting a cell(=pixel) of ndvi vs a cell of thermal, the idea is
        to plot e.g. 0.5m² of ndvi vs 0.5m² of thermal info, to mitigate some of the noise and local variations.
        This is of course a design parameter, so too much smoothing may remove too much useful info, while too little
        smoothing may cause the scatter plot containing too much noise; so a tradeoff

        How?
        * we have to respect spatial info of course
        * is it correct to do this on the already calculated channels (instead of the raw bands)? --> point of discussion
        * I did it now with a convolution based method, but that's also a design choice
        """
        # TODO: implement reduction in a better way (in downsample_raster)
        ndvi_downsampled = self.ndvi.downsample_raster(method=method, reduction=reduction)
        therm_downsampled = self.therm.downsample_raster(method=method, reduction=reduction)
        print('test_THERM: ', np.where(therm_downsampled>300))
        print('test_NDVI: ', np.where(ndvi_downsampled==0))

        return ndvi_downsampled, therm_downsampled

    def _get_wet_dry_edges(self):

        envelope = EnvelopeDetection(x=self.ndvi_downsampled.flatten(), y=self.therm_downsampled.flatten())
        print('SMI_Estimation')
        x = np.array([-1, 1])
        #self.wet_edge = np.array([[-1,1], [0,0]]) #MI removed
        self.wet_edge = np.array([x,(x * envelope.slope_bottom) + envelope.intercept_bottom])
        print('wet slope: ', envelope.slope_bottom)
        print('wet interc: ', envelope.intercept_bottom)

        x = np.array([0.75, 1]) #changed starting point from 0.5 to 0.75

        #print('xline: ', x)
        print('dry slope: ', envelope.slope)
        print('dry interc: ', envelope.intercept)
        print('dry y: ', x * envelope.slope + envelope.intercept)
        self.dry_edge = np.array([x,(x * envelope.slope) + envelope.intercept]) #removed intercept+5 MI
        #self.dry_edge = np.array([x,x * envelope.slope + envelope.intercept+5.])

    def show_scatter_plot(self):

        imgDF = pd.DataFrame()
        # imgDF['location'] = location
        imgDF['Thermal'] = self.therm_downsampled.flatten() # therm.raster[0, :, :].flatten()
        imgDF['NDVI'] = self.ndvi_downsampled.flatten() # ndvi.raster[0, :, :].flatten()
        # imgDF['labels'] = labels.flatten()
        imgDF['Temperature (Celsius)'] = imgDF['Thermal']


        fig, axis = plt.subplots()

        axis.set_title('Temperature vs. NDVI', fontsize=10)
        axis.set_ylabel('Temperature (Celsius)', fontsize=10)
        axis.set_xlabel('NDVI', fontsize=10)

        Y = imgDF['Temperature (Celsius)']
        X = imgDF['NDVI']


        # for now, no plotting of dry and wet edge, as this is hardcoded and makes no sense for a random raster anyhow
        if True:
            print('x: ', self.dry_edge[0])
            print('y: ', self.dry_edge[1])
            plt.plot(self.dry_edge[0], self.dry_edge[1], color='orange', linewidth=2, label="warm edge={} *NDVI + {}".format(round(getGrad(self.dry_edge),1), round(getInter(self.dry_edge),1))) # warm edge
            plt.plot(self.wet_edge[0], self.wet_edge[1], color='blue', linewidth=2, label="cold edge={} *NDVI + {}".format(round(getGrad(self.wet_edge),1), round(getInter(self.wet_edge),1))) # cold edge


        plt.axis([-1,1,0, 45]) # try linear reression, r coefficient

        axis.scatter(X, Y,s = 0.5)# c = toPlot['labels'].map(colors))
        plt.legend()
        plt.show()

    def show_smi(self):
        plt.imshow(self.smi, cmap='Greens')
        plt.title("SMI | " + self.date + '_' + self.field, loc='left')
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.show()

        plt.hist(self.smi[~np.isnan(self.smi)], bins=np.linspace(-2, 1.5, num=50))
        plt.show()

    def _get_t_i_max(self, warmEdge, raster_ndvi):
        return getGrad(warmEdge) * raster_ndvi + getInter(warmEdge)

    def _get_t_i_min(self, coldEdge, raster_ndvi):
        return getGrad(coldEdge) * raster_ndvi + getInter(coldEdge)

    def _get_smi(self):
        t_i_max = self._get_t_i_max(self.dry_edge, self.ndvi.get_raster())
        t_i_min = self._get_t_i_min(self.wet_edge, self.ndvi.get_raster())
        smi = (t_i_max - self.therm.get_raster()) / (t_i_max - t_i_min)

        # clip the outliers (some points which are outside the [0, 1]-range are probably points which are outside the
        #  the wet and dry edge, can be clipped)
        smi[smi < 0] = 0
        smi[smi > 1] = 1

        return smi

    def save_smi_raster(self,destination_path):

        # copy meta info from one of the source rasters (should be same for any; need to check?)
        out_meta = self.ndvi.get_raster_meta()

        # get a name (including (rel) path) for the output file
        out_fullfilename = self._create_file_name_smi_raster(destination_path)
        # first create the folder (in case it wouldn't exist yet)
        os.makedirs(os.path.dirname(out_fullfilename), exist_ok=True)

        with rasterio.open(
                out_fullfilename,
                'w',
                **out_meta
        ) as dest_file:
            dest_file.write(self.smi, 1)
        dest_file.close()

    def _create_file_name_smi_raster(self, path_name):
        file_name = 'smi.tif'
        return path_name + "/" + file_name




def getGrad(edge):
    return (edge[1,1] - edge[1,0]) / (edge[0,1] - edge[0,0])

def getInter(edge):
    grad = getGrad(edge)
    return edge[1,0] - grad * edge[0,0]
