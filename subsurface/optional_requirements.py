﻿def require_omf():
    try:
        import omfvista
    except ImportError:
        raise ImportError("The omfvista package is required to run this function.")
    return omfvista


def require_rasterio():
    try:
        import rasterio
    except ImportError:
        raise ImportError("The rasterio package is required to run this function.")
    return rasterio


def require_pyvista():
    try:
        import pyvista
    except ImportError:
        raise ImportError("The pyvista package is required to run this function.")
    return pyvista


def require_imageio():
    try:
        import imageio
    except ImportError:
        raise ImportError("The imageio package is required to run this function.")
    return imageio


def require_geopandas():
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("The geopandas package is required to run this function.")
    return gpd


def require_segyio():
    try:
        import segyio
    except ImportError:
        raise ImportError("The segyio package is required to run this function.")
    return segyio


def require_welly():
    try:
        import welly 
    except ImportError:
        raise ImportError("The welly package is required to run this function.")
    return welly


def require_striplog():
    try:
        import striplog
    except ImportError:
        raise ImportError("The striplog package is required to run this function.")
    return striplog


def require_ezdxf():
    try:
        import ezdxf
    except ImportError:
        raise ImportError("The ezdxf package is required to run this function.")
    return ezdxf


def require_trimesh():
    try:
        import trimesh
    except ImportError:
        raise ImportError("The trimesh package is required to run this function.")
    return trimesh


def require_scipy():
    try:
        import scipy
    except ImportError:
        raise ImportError("The scipy package is required to run this function.")
    return scipy


def require_pandas():
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("The pandas package is required to run this function.")
    return pd


def require_wellpathpy():
    try:
        import wellpathpy
    except ImportError:
        raise ImportError("The wellpathpy package is required to run this function.")
    return wellpathpy


def require_subsurface():
    try:
        import subsurface
    except ImportError:
        raise ImportError("The subsurface package is required to run this function.")
    return subsurface

def require_matplotlib():
    try:
        import matplotlib
    except ImportError:
        raise ImportError("The matplotlib package is required to run this function.")
    return matplotlib