from subsurface import PointSet
from subsurface.reader import borehole_location_to_unstruct
from subsurface.visualization import to_pyvista_points, pv_plot
import pandas as pd


def test_borehole_location_to_unstruct(data_path):
    us = borehole_location_to_unstruct(
        data_path + '/borehole/borehole_collar.xlsx', {
            'columns_map': {1: 'x',
                            2: 'y',
                            3: 'altitude',
                            }
        })
    point_set = PointSet(us)
    s = to_pyvista_points(point_set)


def test_generate_tops(data_path):
    d = pd.read_csv(data_path + '/borehole/no_tops.csv')
    d['_'] = d['Z'] - d['Altitude']
    Z_shift = d.groupby('Index')['_'].shift(1)
    Z_0 = Z_shift.fillna(0)
    v = Z_0 + d['Altitude']

    d['top'] = v