from typing import Sequence, Optional, Tuple

import numpy as np

from ....core.structs.structured_elements.structured_grid import StructuredGrid
from ....optional_requirements import require_rasterio
from ....core.structs import StructuredData, UnstructuredData
from ....core.utils.utils_core import get_extension
from ....core.reader_helpers.readers_data import GenericReaderFilesHelper
from ....core.reader_helpers.reader_unstruct import ReaderUnstructuredHelper
from ..mesh.surfaces_api import read_2d_mesh_to_unstruct


def read_structured_topography(path, crop_to_extent: Optional[Sequence] = None, band: Optional[int] = None) -> StructuredData:
    rasterio = require_rasterio()

    extension = get_extension(path)
    if extension == '.tif':
        structured_data = rasterio_dataset_to_structured_data(
            dataset=rasterio.open(path),
            crop_to_extent=crop_to_extent,
            band=band
        )
    else:
        raise NotImplementedError('The extension given cannot be read yet')

    return structured_data


def read_structured_topography_to_unstructured(path) -> UnstructuredData:
    structured_data = read_structured_topography(path)
    return topography_to_unstructured_data(structured_data)


def rasterio_dataset_to_structured_data(dataset, crop_to_extent: Optional[Sequence] = None, band: Optional[int] = None):
    rasterio = require_rasterio()

    if crop_to_extent is not None:
        window = _get_raster_window(crop_to_extent, dataset)
    else:
        window = None

    band = _select_raster_band(dataset, window, band)
    data = _read_raster_band_as_float(dataset, window, band)
    data = np.fliplr(data.T)
    shape = data.shape

    x, y = _get_raster_center_coords(dataset, shape, window)
    if dataset.crs is not None and dataset.crs.is_geographic:
        x, y = _geographic_coords_to_metric_coords(x, y, dataset.crs)

    coords = {
            'x': x,
            'y': y
    }
    structured_data = StructuredData.from_numpy(data, data_array_name='topography', coords=coords)
    return structured_data


def _select_raster_band(dataset, window=None, band: Optional[int] = None):
    if band is not None:
        return band

    if dataset.count == 1:
        return 1

    band_scores = []
    for band_index in dataset.indexes:
        data = _read_raster_band_as_float(dataset, window, band_index)
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            band_scores.append((-1, -1, -1, band_index))
            continue

        sample = valid_data[::max(valid_data.size // 10000, 1)]
        unique_values = np.unique(sample).size
        value_range = float(np.nanmax(valid_data) - np.nanmin(valid_data))
        band_scores.append((unique_values, value_range, valid_data.size, band_index))

    return max(band_scores)[-1]


def _read_raster_band_as_float(dataset, window=None, band: int = 1):
    data = dataset.read(band, window=window, masked=True)
    data = data.astype(float).filled(np.nan)

    if dataset.nodatavals[band - 1] is None and np.issubdtype(dataset.dtypes[band - 1], np.unsignedinteger):
        unsigned_max = np.iinfo(dataset.dtypes[band - 1]).max
        data[data == unsigned_max] = np.nan

    return data


def _get_raster_center_coords(dataset, shape: Tuple[int, int], window=None):
    rasterio = require_rasterio()

    if window is not None:
        transform = rasterio.windows.transform(window, dataset.transform)
    else:
        transform = dataset.transform

    x = rasterio.transform.xy(transform, 0, np.arange(shape[0]), offset='center')[0]
    y = rasterio.transform.xy(transform, np.arange(shape[1] - 1, -1, -1), 0, offset='center')[1]
    return np.asarray(x), np.asarray(y)


def _geographic_coords_to_metric_coords(x, y, crs):
    rasterio = require_rasterio()
    from rasterio import warp

    longitude_center = float(np.nanmean(x))
    latitude_center = float(np.nanmean(y))
    zone = int((longitude_center + 180) // 6) + 1
    epsg = 32600 + zone if latitude_center >= 0 else 32700 + zone
    metric_crs = rasterio.crs.CRS.from_epsg(epsg)

    x_metric, _ = warp.transform(
        src_crs=crs,
        dst_crs=metric_crs,
        xs=x,
        ys=np.full_like(x, latitude_center, dtype=float)
    )
    _, y_metric = warp.transform(
        src_crs=crs,
        dst_crs=metric_crs,
        xs=np.full_like(y, longitude_center, dtype=float),
        ys=y
    )
    return np.asarray(x_metric), np.asarray(y_metric)


def rasterio_dataset_to_structured_data_(dataset, crop_to_extent: Optional[Sequence] = None):
    if crop_to_extent is not None:
        window = _get_raster_window(crop_to_extent, dataset)
    else:
        window = None

    data = dataset.read(1, window=window)
    data = np.fliplr(data.T)
    shape = data.shape

    # TODO: ===================
    # TODO: Add the option to crop
    # TODO: Resample

    coords = {
            'x': np.linspace(
                dataset.bounds.left,
                dataset.bounds.right,
                shape[0]
            ),
            'y': np.linspace(
                dataset.bounds.bottom,
                dataset.bounds.top,
                shape[1]
            )
    }
    structured_data = StructuredData.from_numpy(data, data_array_name='topography', coords=coords)
    return structured_data


def read_unstructured_topography(path, additional_reader_kwargs: Optional[dict] = None) -> UnstructuredData:
    """For example, a dxf file"""

    additional_reader_kwargs = additional_reader_kwargs or {}
    helper = GenericReaderFilesHelper(
        file_or_buffer=path,
        additional_reader_kwargs=additional_reader_kwargs or {}
    )
    unstructured_helper = ReaderUnstructuredHelper(helper)
    unstruct: UnstructuredData = read_2d_mesh_to_unstruct(unstructured_helper)
    return unstruct


def topography_to_unstructured_data(structured_data: StructuredData) -> UnstructuredData:
    from subsurface.modules.visualization import to_pyvista_grid

    sg = StructuredGrid(structured_data)
    s = to_pyvista_grid(sg, data_order='C', data_set_name='topography')
    un_s = s.cast_to_unstructured_grid()
    un_s.triangulate(inplace=True)
    vertex = un_s.points
    cells = un_s.cells.reshape(-1, 4)[:, 1:]

    unstructured_data = UnstructuredData.from_array(vertex, cells)
    return unstructured_data


def _get_raster_window(crop_to_extent, dataset):
    from rasterio.windows import Window
    # TODO: Add None check
    # Get the indices of the window
    left, bottom, right, top = crop_to_extent
    row_start, col_start = dataset.index(left, top)
    row_stop, col_stop = dataset.index(right, bottom)
    # Read the data in the window
    window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
    return window
