import fiona
import geopandas as gpd


class PolygonLoader:
    def __init__(self, path="../../task-0-raw-data/data/Outlines/", outline=None):
        """
        Load polygons based on shape files. With @param::outline you can pre-select which outlines to load.
        Remark that when nothing is specified, it will load all polygons (from v2)
        """
        self.base_path = path

        # check if outline is supported
        if self._is_valid_outline(outline):
            # make sure to put it in list format, if not yet
            if outline is None:
                # If nothing is specified, load them all (and use v2 if possible)
                self.outline = ['Fairways_v2', 'teesF1', 'teesF2', 'Greens_v2', 'lakes', 'holes_outlines']
            elif type(outline) is list:
                if 'tees' in outline:
                    outline.remove('tees')
                    outline.append('teesF1')
                    outline.append('teesF2')
                self.outline = outline
            else:
                if outline == 'tees':
                    self.outline = [outline + 'F1', outline + 'F2']
                else:
                    self.outline = [outline]
        else:
            raise AttributeError('Not supported outline')

        # load the shape files
        # fiona replaced with geopandas // self._load_outline_fiona()
        self._load_outline_gpd()

    def get_outlines(self):
        """ Public method, to return all the outline polygons in a list """
        return self.get_outline_on_spec(field=None, hole=None, outline_type=None)

    def get_outline_on_idx(self, outline_index):
        """ Public method, to return the a single outline polygon (still in list), based on an index """
        outline_shapes = self.get_outlines()
        if 0 <= outline_index < len(outline_shapes):
            return [outline_shapes[outline_index]]
        else:
            raise OverflowError("Index " + outline_index + " out of range for outlines-list of length: "
                                + len(outline_shapes))

    def get_outline_on_spec(self, field=None, hole=None, outline_type=None):
        """ Public method, to return a list of outline polygons, based upon the optional selection criteria """
        outline_selected = self.outline_df
        # optional filter on field
        if field is not None:
            if field in ['F1', 'F2']:
                outline_selected = outline_selected.loc[(outline_selected['Field'] == field)]
            else:
                raise AttributeError('Provided field ' + field + ' is not supported')
        # optional filter on hole
        if hole is not None:
            outline_selected = outline_selected[(outline_selected['hole'] == hole)]
        # optional filter on type
        if outline_type is not None:
            if outline_type in ['fairways', 'greens', 'tees', 'Lakes']:
                outline_selected = outline_selected[(outline_selected['Type'] == outline_type)]
            else:
                raise AttributeError('Provided outline type ' + outline_type + ' is not supported')

        geoms = []
        for _, outline in outline_selected.iterrows():
            geoms.append(outline['geometry'])

        return geoms

    def _is_valid_outline(self, outline):
        """ Return boolean which indicates if provided outlines are supported """

        supported_outlines = ['fairways', 'Fairways_v2', 'tees', 'greens', 'Greens_v2', 'lakes', 'holes_outlines']

        if type(outline) is list:
            # TODO: maybe we need to check that fairways and Fairways_v2 are not selected at same time (should be
            #  exclusive selection)
            return set(outline).issubset(supported_outlines)
        elif type(outline) is str:
            return outline in supported_outlines
        elif outline is None:
            return True
        else:
            return False

    # Not used anymore... replaced by geopandas loader
    def _load_outline_fiona(self):
        """ Private method to load the outlines, based on the spec """

        if self.outline is not None:
            self.outline_shapes = []
            for outline_type in self.outline:
                with fiona.open(self.base_path + _get_filename(outline_type), "r") as shapefile:
                    # keep joining the lists of geometries, for every shape file
                    self.outline_shapes += [feature["geometry"] for feature in shapefile]
        else:
            self.outline_shapes = None

    def _load_outline_gpd(self):
        """ Private method to load the outlines, based on the spec - using geopandas
            Also some manipulations on the data or executed (adding field, adding type if not yet present)
        """

        tmp = None
        if self.outline is not None:
            for outline_type in self.outline:
                shapefile = gpd.read_file(self.base_path + _get_filename(outline_type))
                shapefile = shapefile.rename({'Hole':'hole'}, axis=1)

                # if shapefile has no Field column, try to add field based on outline_type
                if 'Field' not in shapefile.columns.to_list():
                    shapefile = shapefile.assign(Field=_extract_field_from_filename(outline_type))
                # add polygon type (fairway, tee,...) to df, if not in yet
                if 'Type' not in shapefile.columns.to_list():
                    shapefile = shapefile.assign(Type=_get_nice_outline_name(outline_type))

                if tmp is None:
                    tmp = shapefile
                else:
                    tmp = tmp.append(shapefile)
        else:
            self.outline_df = None

        self.outline_df = tmp


def _extract_field_from_filename(outline):
    idx_start = outline.find('F')
    if idx_start > 0:
        return outline[idx_start:]
    else:
        print('Warning!! Cannot find fieldname in : ' + outline + '. Unknown field name added to dataframe.')
        return 'Unknown'


def _get_filename(outline):
    return outline + ".shp"


def _get_nice_outline_name(outline):
    if outline == 'Fairways_v2':
        return 'fairways'
    elif outline == 'Greens_v2':
        return 'greens'
    elif 'F' in outline:
        return outline[0:outline.find('F')]
    else:
        return outline
