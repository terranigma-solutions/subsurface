import json

from tests.conftest import RequirementsLevel, check_requirements

import pytest
import numpy as np
import pandas as pd

from subsurface import UnstructuredData, StructuredData, optional_requirements
from subsurface.core.structs.unstructured_elements import TriSurf
from subsurface.modules.reader.read_netcdf import read_unstruct
from subsurface.modules.reader.profiles.profiles_core import create_mesh_from_trace
from subsurface.modules.visualization import to_pyvista_mesh_and_texture


@pytest.fixture(scope='module')
def wells(data_path):
    us = read_unstruct(data_path + '/wells.nc')
    return us


@pytest.mark.skip("Run only explicitly. Is giving OS issues.")
def test_wells_to_binary(wells):
    bytearray_le, header = wells.to_binary_legacy()

    with open('well_f.json', 'w') as outfile:
        json.dump(header, outfile)

    new_file = open("wells_f.le", "wb")
    new_file.write(bytearray_le)


@pytest.mark.skipif(check_requirements(RequirementsLevel.GEOSPATIAL), reason="Geopandas is not imported ")
def test_profile_to_binary(data_path):
    gpd = optional_requirements.require_geopandas()
    traces = gpd.read_file(data_path + '/profiles/Traces.shp')
    v, e = create_mesh_from_trace(
        traces.loc[0, 'geometry'],
        traces.loc[0, 'zmax'],
        traces.loc[0, 'zmin']
    )

    unstruct_temp = UnstructuredData.from_array(v, e)

    imageio = optional_requirements.require_imageio()
    cross = imageio.imread(data_path + '/profiles/Profil1_cropped.png')
    struct = StructuredData.from_numpy(np.array(cross))
    texture_binary, texture_header = struct.default_data_array_to_binary_legacy()

    origin = [
            traces.loc[0, 'geometry'].xy[0][0],
            traces.loc[0, 'geometry'].xy[1][0],
            int(traces.loc[0, 'zmin'])
    ]
    point_u = [
            traces.loc[0, 'geometry'].xy[0][-1],
            traces.loc[0, 'geometry'].xy[1][-1],
            int(traces.loc[0, 'zmin'])
    ]
    point_v = [
            traces.loc[0, 'geometry'].xy[0][0],
            traces.loc[0, 'geometry'].xy[1][0],
            int(traces.loc[0, 'zmax'])
    ]

    texture_header['texture_origin'] = origin
    texture_header['texture_point_u'] = point_u
    texture_header['texture_point_v'] = point_v

    ts = TriSurf(
        mesh=unstruct_temp,
        texture=struct,
        texture_origin=origin,
        texture_point_u=point_u,
        texture_point_v=point_v
    )

    _, uv = to_pyvista_mesh_and_texture(ts)
    import pandas as pd

    unstruct = UnstructuredData.from_array(v, e, vertex_attr=pd.DataFrame(uv, columns=['u', 'v']))
    mesh_binary, mesh_header = unstruct.to_binary_legacy()

    with open('mesh_uv.json', 'w') as outfile:
        import json
        json.dump(mesh_header, outfile)

    with open('texture.json', 'w') as outfile:
        json.dump(texture_header, outfile)

    new_file = open("mesh_uv_f.le", "wb")
    new_file.write(mesh_binary)

    new_file = open("texture_f.le", "wb")
    new_file.write(texture_binary)

    return mesh_binary


class TestFilterNumericColumns:
    def test_keeps_numeric_columns_when_mixed_with_object(self):
        """Regression: object-dtype columns containing only numeric values
        must be included in _filter_numeric_columns output with a numeric dtype."""
        from subsurface.core.structs.base_structures._liquid_earth_mesh import _filter_numeric_columns

        df = pd.DataFrame({
            "well_id": pd.Series([0, 1, 2], dtype=object),
            "depth": pd.Series([1.5, 2.0, 3.0], dtype=object),
            "Cu_pct": pd.Series(["4.0", "3.0", "2.0"], dtype=object),
            "As_pct": pd.Series(["n.a.", "<0.005", "0.001"], dtype=object),
        })

        result = _filter_numeric_columns(df)

        assert "well_id" in result.columns
        assert "depth" in result.columns
        assert "Cu_pct" in result.columns
        assert "As_pct" not in result.columns

        assert pd.api.types.is_numeric_dtype(result["Cu_pct"])
        assert result["Cu_pct"].tolist() == [4.0, 3.0, 2.0]

    def test_keeps_mixed_int_float_numeric(self):
        from subsurface.core.structs.base_structures._liquid_earth_mesh import _filter_numeric_columns

        df = pd.DataFrame({
            "a": pd.Series([1, 2, 3], dtype=np.int32),
            "b": pd.Series([1.1, 2.2, 3.3], dtype=np.float64),
            "c": pd.Series([True, False, True], dtype=bool),
        })

        result = _filter_numeric_columns(df)
        assert list(result.columns) == ["a", "b", "c"]

    def test_excludes_all_text_when_no_numeric(self):
        from subsurface.core.structs.base_structures._liquid_earth_mesh import _filter_numeric_columns

        df = pd.DataFrame({
            "text": pd.Series(["foo", "bar", "baz"], dtype=object),
        })

        result = _filter_numeric_columns(df)
        assert result.empty

    def test_round_trip_unstructured_data_with_mixed_attrs(self):
        df_vertex = pd.DataFrame({
            "well_id": pd.Series([0, 0, 1, 1], dtype=object),
            "depth": pd.Series([0.0, 1.0, 0.0, 1.0], dtype=object),
            "Cu_pct": pd.Series([4.0, 3.0, 2.0, 1.0], dtype=object),
            "notes": pd.Series(["", "", "lost", ""], dtype=object),
        })

        unstr = UnstructuredData.from_array(
            vertex=np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]], dtype=np.float32),
            cells=np.array([[0, 1], [2, 3]], dtype=np.int32),
            vertex_attr=df_vertex,
        )

        header = unstr._set_binary_header()
        attr_names = [m["name"] for m in header["vertex_attrs"]]
        assert "well_id" in attr_names
        assert "depth" in attr_names
        assert "Cu_pct" in attr_names
        assert "notes" not in attr_names

        binary = unstr.to_binary()
        assert len(binary) > 0
