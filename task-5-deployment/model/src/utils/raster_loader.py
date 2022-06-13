# import gdal
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sns
import numpy as np
import math
from collections import OrderedDict

# Enforce rasterio 1.2.10
# this seems not to work for Radikha (due to Mac?), as long as this is not understood, the enforcement is elevated
#import pkg_resources
# pkg_resources.require("rasterio==1.2.10")
import rasterio
rasterio_version = rasterio.__version__
if rasterio_version != '1.2.10':
    print("Rasterio version: " + rasterio.__version__ + " is not matching the required version 1.2.10 --> no guarantees the code will work!")

import rasterio.mask
import glob
from pathlib import Path

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

import cv2


# Prettier plotting with seaborn
# sns.set(font_scale=1.5)


class RasterLoader:
    def __init__(self, date=None, band='NDVI', field='F1', outline=None, crop_data=False,
                 base_path='../../task-0-raw-data/data/'):

        # Remark: the base_path should be something like, level after that is constructed!!
        # > '../../task-0-raw-data/data/'
        # > or, "../../task-1-preprocessing/compression/compressed-tifs/

        self.date = date  # should be in yyyy-mm-dd format --> check to be added!

        # if band not in ['Therm', 'NDVI', 'NDRE', 'DEM', 'Multispectral', 'SMI']:
        #     raise AttributeError('Unsupported band: ' + band)
        self.band = band

        if field not in ['F1', 'F2']:
            raise AttributeError('Unsupported field: ' + field)
        self.field = field

        self.outline_shapes = outline

        self.crop_data = crop_data

        # User responsibility to specify an absolute or relative path to the location of the raster
        self.base_path = base_path

        method = 'RASTERIO'
        if method == 'RASTERIO':
            # Load the raster
            self._load_raster()
            # Correct for faulty values, outliers,...
            self._correct_raster()
        else:
            raise Exception('Only rasterio supported, gdal not')
            # gdal not supported anymore, given issues with setting up the right python environment with gdal package...
            # self.raster = self._load_w_gdal()

    def get_raster(self):
        """ Public method such that the user can fetch the raster """
        if self.band == 'Multispectral':
            return self.raster
        else:
            return self.raster[0, :, :]

    def get_raster_meta(self):
        """ Public method such that the user can fetch the meta info from the raster (with crs etc.) """
        return self.raster_meta

    def _get_filename(self) -> str:
        return self.band + ".tif"
        # if self.band == 'DEM':
        #     # DEM has no data, as this is static info for the site
        #     return self.field + "_" + self.band.lower() + ".tif"
        # else:
        #     return self.date + "_" + self.field + "_" + self.band.lower() + ".tif"

    def _get_full_filename(self):
        # if 'task-0' in self.base_path:
        #     # for task-0 folder, need to add band as folder name
        #     base_path_ext = self.base_path + "/" + self.band + "/"
        # else:
        #     # for task-1, task-2 data folders, need to add date as folder layer in between (not the band)
        #     base_path_ext = self.base_path + "/" + self.date + "/"

        filepath = glob.glob(self.base_path + self.band + ".tif")
        if len(filepath) == 1:
            return filepath[0]
        else:
            # If above method didn't work, try in another way: find a unique filename in the base_path
            rel_paths = Path(self.base_path).rglob(self._get_filename())
            rel_paths_sorted = sorted(rel_paths)

            if len(rel_paths_sorted) == 0:
                raise FileNotFoundError(self._get_filename() + ' not found within ' + self.base_path)
            elif len(rel_paths_sorted) == 1:
                return rel_paths_sorted[0].as_posix()
            else:
                raise FileExistsError(self._get_filename() + ' not unique within ' + self.base_path + '; please make sure the base path is specific enough')

    def _load_raster(self):
        if self.outline_shapes is not None:
            with rasterio.open(self._get_full_filename()) as src:
                # Important to set crop=False, in case you want to retain the geo-information
                # (because if set to True, it will shift the raster possibly, and then the geo reference may be lost)
                try:
                    if self._get_raster_dtype(src.dtypes) == 'uint16':
                        nodata = 0
                    else:
                        nodata = np.nan
                    out_image, out_transform = rasterio.mask.mask(src, self.outline_shapes, crop=self.crop_data,
                                                                  nodata=nodata)
                except ValueError:
                    # probably Input shapes do not overlap with raster
                    # should we handle this? for now just rethrow the error
                    raise

                out_meta = src.meta
                # update the meta-info with the transform AFTER masking
                out_meta['transform'] = out_transform
                out_meta['width'] = out_image.shape[2]
                out_meta['height'] = out_image.shape[1]

        else:
            with rasterio.open(self._get_full_filename()) as src:
                out_image = src.read()
                out_meta = src.meta

        self.raster = out_image
        self.raster_meta = out_meta

    # def _load_w_gdal(self):
    #     ds = gdal.Open(self.base_path + "/" + self._get_filename())
    #     dsClip = gdal.Warp("dsClip.tif", ds, cutlineDSName = self.base_path + "/Outlines/"+self.outline+".shp",
    #                        cropToCutline = True, dstNodata = np.nan)
    #
    #     return dsClip

    def get_raster_mask(self):
        # TODO: now had to implement it in this dirty way, that we load the file once more. Can we do better?
        #  Or is there no signficant impact on execution time doing it this way?
        if self.outline_shapes is not None:
            with rasterio.open(self._get_full_filename()) as src:
                mask, _, _ = rasterio.mask.raster_geometry_mask(src, self.outline_shapes, all_touched=True,
                                                                crop=self.crop_data)
                return mask

        return None

    def _get_raster_dtype(self, raster_dtype):
        if isinstance(raster_dtype, tuple) or isinstance(raster_dtype, list):
            # TODO: now we return datatype of one of the layers, but in case that wouldn't be the same accross the
            #  layers, one should actually check for the worst case dtype (uint16 is more worst case than double e.g.)
            return raster_dtype[0]
        elif isinstance(raster_dtype, str):
            return raster_dtype
        else:
            raise ValueError('Unknown format of raster datatype: ' + raster_dtype)

    def _correct_raster(self):
        if self.band == "Therm":
            # replace invalid measurements with nan
            self.raster[self.raster > 300] = np.nan
        elif self.band == "DEM":
            # replace none values (apparently -34000 for DEM) with nan
            self.raster[self.raster == -34000.] = np.nan

        # TODO: to be done for other rasters

    def plot_raster(self):
        # remark: we provide self.raster, and not self.get_raster(), to assure that the first dimension represents
        # the number of layers/channels (which may not be the case for self.get_raster() for a single layer)
        self._plot_raster_generic(self.raster, title_prefix="")

    def plot_downsampled_raster(self):
        downsampled_raster_exp = self.downsampled_raster[None, :, :]
        self._plot_raster_generic(downsampled_raster_exp, title_prefix="DOWNSAMPLED | ")

    def _plot_raster_generic(self, raster_vals, title_prefix=""):
        # TODO: need to add the "natural" colors for every band
        if self.band == "Therm":
            cmap = 'coolwarm'
        elif self.band == 'NDVI':
            cmap = 'RdYlGn'
        elif self.band == 'Multispectral':
            cmap = 'gray'
            # remark: could also be specified as a list, for each channel!
            # example of a random list cmap = ['gray', 'BrBG', 'CMRmap', 'Dark2_r', 'YlGn', 'coolwarm', 'RdYlGn']
        else:
            cmap = 'YlGn'

        nr_subplots = raster_vals.shape[0]
        nr_columns = math.ceil(math.sqrt(nr_subplots))
        nr_rows = nr_columns

        fig, axes = plt.subplots(nr_rows, nr_columns)
        # manipulate the axes, always to a 2-dimensional array, such that we can generically iterate over it
        if nr_rows == 1 and nr_columns == 1:
            axes = np.array(axes)
            axes = np.expand_dims(axes, axis=(0, 1))
        elif nr_rows == 1 and nr_columns > 1:
            axes = np.expand_dims(axes, axis=0)
        elif nr_rows > 1 and nr_columns ==1:
            axes = np.expand_dims(axes, axis=1)
        else:
            pass # do nothing

        idx = 0
        for row in axes:
            for ax in row:
                if idx < nr_subplots:
                    if isinstance(cmap, list):
                        cmap_tmp = cmap[idx]
                    else:
                        cmap_tmp = cmap
                    im = ax.imshow(raster_vals[idx, :, :], cmap=cmap_tmp)

                    # Create divider for existing axes instance
                    divider = make_axes_locatable(ax)
                    # Append axes to the right of ax3, with 20% width of ax3
                    cax = divider.append_axes("right", size="10%", pad=0.05)
                    # Create colorbar in the appended axes
                    # Tick locations can be set with the kwarg `ticks`
                    # and the format of the ticklabels with kwarg `format`
                    plt.colorbar(im, cax=cax) #, ticks=MultipleLocator(0.2), format="%.2f")

                    if self.band == 'Multispectral':
                        subplot_title = 'Multispectral[{idx:d}]'.format(idx=idx)
                    else:
                        subplot_title = self.band
                    ax.set_title(subplot_title, loc='left')

                    idx += 1
                else:
                    # These are unpopulated subplots --> remove them
                    fig.delaxes(ax)

        fig.suptitle(title_prefix + self._get_filename())
        plt.tight_layout()
        plt.show()

    def plot_hist(self):
        _ = plt.hist(self.raster.flatten(), bins='auto')
        plt.show()

    def get_raster_stats(self):
        return None

    def downsample_raster(self, method='Average', reduction=9):
        if self.band == 'Multispectral':
            raise ValueError("Downsampling of multispectral raster(given multi dimensionality) not yet supported !")

        # this is actually a slow implementation... trying astropy instead

        # Remark first used  https://stackoverflow.com/questions/38318362/2d-convolution-in-python-with-missing-data/40416633
        # but turns out to be crazily slow, so moved to astropy and stepwise approach(for >10 reduction)

        """
        convolved = convolve2d(self.raster[0, :, :], kernel, max_missing=0.4, verbose=True)
        downsampled = convolved[::n, ::n] / n
        """

        if reduction < 10:
            if method == 'Gaussian':
                kernel = Gaussian2DKernel(x_stddev=1)
                # TODO: need to still check how the kernel size can be set for gaussiankernel (it is related to this stddev)
            elif method == "Average":
                kernel = np.ones((reduction, reduction))
            else:
                raise AttributeError('Unsupported method for downsampling')

            n = kernel.shape[0]
            convolved = convolve(self.get_raster(), kernel)
            downsampled = convolved[::n, ::n]

        else:
            # split reduction in 2 steps (how?)
            reduction = np.int(np.floor(np.sqrt(reduction)))
            if np.mod(reduction, 2) == 0:
                reduction += 1

            if method == 'Gaussian':
                kernel = Gaussian2DKernel(x_stddev=1)
                # TODO: need to still check how the kernel size can be set for gaussiankernel (it is related to this stddev)
            elif method == "Average":
                kernel = np.ones((reduction, reduction))
            else:
                raise AttributeError('Unsupported method for downsampling')

            n = kernel.shape[0]

            convolved = convolve(self.get_raster(), kernel)
            downsampled = convolved[::n, ::n]

            # repeat once more
            # TODO: we do this because convolution seems quite computationally hungry, so if you want to have a reduction of
            #  let's say 25, it may be better to do a reduction of 5, and then another one of 5... so how to deal with this?
            convolved = convolve(downsampled, kernel)
            downsampled = convolved[::n, ::n]

        """
        if np.mod(reduction, 2) == 0:
            n = reduction + 1
        else:
            n = reduction
            kernel = np.ones((n, n))

        downsampled = convolve2d(convolved, kernel, mode='valid')
        downsampled = convolved[::n, ::n] / n
        """
        self.downsampled_raster = downsampled

        return downsampled


class RasterStacker:
    def __init__(self, date, field, channels=None, outline=None, base_paths=None):
        """ First positional argument: date --> there always needs to be a date! """
        self.date = date

        """ Second positional argument: field """
        self.field = field

        """ Optional argument channels. If set to none, it is defaulted to below list """
        if channels is None:
            # default channels
            channels = ['Multispectral', 'NDVI', 'NDRE', 'Therm']

        self.channels = channels

        self.outline = outline

        path_dict = {'NDVI': '../../task-1-preprocessing/',
                     'NDRE': '../../task-1-preprocessing/',
                     'Therm': '../../task-1-preprocessing/',
                     'Multispectral': '../../task-1-preprocessing/',
                     'DEM': '../../task-0-raw-data/',
                     'SMI': '../../task-2-add-metrics-soil/'}
        if base_paths is None:
            self.base_paths = path_dict
        elif isinstance(base_paths, dict):
            self.base_paths = base_paths
        else:
            raise AttributeError("Unsupported base_paths input (should be None or dict): " + base_paths)

        # check that every provided channel, has a path
        for channel in self.channels:
            if channel not in list(self.base_paths.keys()):
                raise AttributeError('Channel {ch:} has no corresponding path in the provided dictionary!'.format(ch=channel))

        self.stack_dict, self.mask = self._load_and_stack_rasters()

    def get_mask(self):
        return self.mask

    def get_full_stack(self):
        # Full stack
        return self._make_stack(self.channels)

    def get_sub_stack(self, selected_channels=None):
        # stack some specific channels provided by application
        # TODO: support also separate channels of multispectral layer to be selectable!
        if selected_channels is None:
            selected_channels = ['NDVI', 'Therm']
        return self._make_stack(selected_channels)

    def get_transform(self, band=None):
        """ Public method such that user can fetch the transformation info (CRS Affine transformation)
            If a band is specified, than the transform for a specific band is returned, otherwise a transform
            at random is chosen (normally they should also correspond, otherwise the stack would not be meaningful)
         """
        meta = self.get_meta(band=band)

        return meta['transform']

    def get_crs(self, band=None):
        meta = self.get_meta(band=band)

        return meta['crs']

    def get_meta(self, band=None):
        raster = None

        # Try to get a raster
        if band is None:
            for key, value in self.stack_dict.items():
                if key != 'DEM':
                    raster = value
                    break
        else:
            try:
                raster = self.stack_dict[band]
            except:
                pass

        if raster is None:
            raise ValueError('Could not find a valid band (!=DEM) to fetch the meta-info!')

        meta = raster.get_raster_meta()
        return meta

    def _make_stack(self, selected_channels):

        stack = None
        channels = []
        for key, value in self.stack_dict.items():
            if key in selected_channels:
                if key == 'Multispectral':
                    # Select 5 layers: R, G, B, NIR, RED_EDGE
                    raster_layer = value.raster[0:5, :, :]
                    channel = ['R', 'G', 'B', 'NIR', 'RED_EDGE']
                elif key == 'DEM':
                    # DEM is handled seperately as well, as it needs to be resampled
                    channel = [key]
                    if stack is not None:
                        width = stack.shape[2]
                        height = stack.shape[1]
                        target_dim = (width, height)
                        resampled_DEM = cv2.resize(value.raster[0, :, :], dsize=target_dim, interpolation=cv2.INTER_NEAREST)
                        raster_layer = np.expand_dims(resampled_DEM, axis=0)
                    else:
                        raise Exception("DEM may not be specified as first channel, as it will use other channels to resample to!")

                else:
                    # other rasters are assumed to have only one layer
                    raster_layer = value.raster
                    channel = [key]

                if stack is None:
                    stack = np.vstack([raster_layer])
                else:
                    stack = np.vstack([stack, raster_layer])

                channels = channels + channel

        # move "channel" dimension to last dimension
        stack = stack.transpose((1, 2, 0))
        return stack, channels

    def get_channel_by_name(self):
        # get a single channel from the stack
        return None

    def _is_valid_channels(self):
        # TODO: check that provided channels are supported, and in right format
        return None

    def _load_and_stack_rasters(self):
        # Make empty dictionary of rasters
        # We have chosen to use an ordered dictionary, such that DEM is not suddenly the first layer (which is undesired
        #  from resampling purposes). An ordered dictionary has same functionality as a normal dict, but keeps the order
        #  of inserting
        stack = OrderedDict()
        mask = None

        for band in self.channels:

            if band == 'DEM':
                date = None
            else:
                date = self.date

            # TODO: for now hardcoded to cropping, may not always be desired !!
            raster = RasterLoader(band=band, date=date, field=self.field,
                                  outline=self.outline,
                                  crop_data=True, base_path=self.base_paths[band])
            # Add to dictionary
            stack[band] = raster

            # Once, get a mask (to be improved)
            if mask is None:
                mask = raster.get_raster_mask()

        return stack, mask

    def plot_stack(self):
        for _, value in self.stack_dict.items():
            value.plot_raster()