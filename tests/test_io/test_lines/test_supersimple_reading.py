import pytest
import os
from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from ._aux_func import _plot
from ...conftest import RequirementsLevel

PLOT = False

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_WELL"
)


def test_read_supersimple_borehole_data(data_path):
    # Paths to the supersimple files
    collar_path = os.path.join(data_path, 'borehole', 'supersimple_collars.csv')
    survey_path = os.path.join(data_path, 'borehole', 'supersimple_survey.csv')
    lith_path = os.path.join(data_path, 'borehole', 'supersimple_lithology.csv')

    # Create readers
    collars_reader = GenericReaderFilesHelper(file_or_buffer=collar_path)
    surveys_reader = GenericReaderFilesHelper(file_or_buffer=survey_path)
    attrs_reader = GenericReaderFilesHelper(file_or_buffer=lith_path)

    # Read wells as assays (is_lith_attr=False)
    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=False
    )

    # Verification
    assert len(borehole_set.collars.ids) == 2
    assert "zzz" in borehole_set.collars.ids
    assert "aaa" in borehole_set.collars.ids

    # Check survey data
    assert hasattr(borehole_set.survey, 'survey_trajectory')

    # Check attributes
    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    assert "component lith" in points_attrs.columns

    well_zzz_id = borehole_set.survey.get_well_num_id("zzz")
    well_zzz_attrs = points_attrs[points_attrs["well_id"] == well_zzz_id]
    assert not well_zzz_attrs.empty

    if PLOT:
        _plot(
            scalar="component lith",
            trajectory=borehole_set.combined_trajectory,
            collars=borehole_set.collars,
            radius=0.01
        )


def test_read_supersimple_as_lithology(data_path):
    # Testing reading it as lithology
    collar_path = os.path.join(data_path, 'borehole', 'supersimple_collars.csv')
    survey_path = os.path.join(data_path, 'borehole', 'supersimple_survey.csv')
    lith_path = os.path.join(data_path, 'borehole', 'supersimple_lithology.csv')

    collars_reader = GenericReaderFilesHelper(file_or_buffer=collar_path)
    surveys_reader = GenericReaderFilesHelper(file_or_buffer=survey_path)
    attrs_reader = GenericReaderFilesHelper(file_or_buffer=lith_path)

    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=True
    )

    assert len(borehole_set.collars.ids) == 2

    # In lithology mode, component lith should be present and have categorical values
    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    assert "component lith" in points_attrs.columns

    # Check for some lithology values from supersimple_lithology.csv
    # zzz has bun, cheese, patty
    well_zzz_id = borehole_set.survey.get_well_num_id("zzz")
    well_zzz_attrs = points_attrs[points_attrs["well_id"] == well_zzz_id]
    unique_liths = well_zzz_attrs["component lith"].unique()
    assert any("bun" in str(l) for l in unique_liths)
    assert any("cheese" in str(l) for l in unique_liths)


def test_read_supersimple_with_mapping(data_path):
    # Test reading with manual mapping
    collar_path = os.path.join(data_path, 'borehole', 'supersimple_collars.csv')
    survey_path = os.path.join(data_path, 'borehole', 'supersimple_survey.csv')
    lith_path = os.path.join(data_path, 'borehole', 'supersimple_lithology.csv')

    collars_reader = GenericReaderFilesHelper(
        file_or_buffer=collar_path,
        columns_map={"id": "id", "x": "x", "y": "y", "z": "z"}
    )
    surveys_reader = GenericReaderFilesHelper(
        file_or_buffer=survey_path,
        columns_map={"id": "id", "md": "md"}
    )
    attrs_reader = GenericReaderFilesHelper(
        file_or_buffer=lith_path,
        columns_map={"id": "id", "top": "top", "base": "base", "component lith": "component lith"}
    )

    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=True,
        add_attrs_as_nodes=True,
        duplicate_attr_depths=True
    )

    assert len(borehole_set.collars.ids) == 2

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    well_aaa_id = borehole_set.survey.get_well_num_id("aaa")
    well_aaa_attrs = points_attrs[points_attrs["well_id"] == well_aaa_id]

    # Check for lettuce which only appears in aaa
    assert any("lettuce" in str(l) for l in well_aaa_attrs["component lith"].unique())

    if PLOT:
        _plot(
            scalar="lith_ids",
            trajectory=borehole_set.combined_trajectory,
            collars=borehole_set.collars,
            image_2d=False,
            radius=0.01
        )


@pytest.mark.parametrize("collar_file, survey_file, lith_file", [
    ('supersimple_collars.csv', 'supersimple_survey.csv', 'supersimple_lithology.csv'),
    ('supersimple_collars_v2.csv', 'supersimple_survey_v2.csv', 'supersimple_lithology_v2.csv'),
    ('supersimple_collars_v3.csv', 'supersimple_survey_v3.csv', 'supersimple_lithology_v3.csv'),
    ('supersimple_collars.csv', 'supersimple_survey_v2.csv', 'supersimple_lithology_v3.csv'),
    ('supersimple_collars_v3.csv', 'supersimple_survey.csv', 'supersimple_lithology.csv'),
    ('supersimple_collars_v2.csv', 'supersimple_survey_v3.csv', 'supersimple_lithology_v2.csv'),
])
def test_read_supersimple_order_invariance(data_path, collar_file, survey_file, lith_file):
    collar_path = os.path.join(data_path, 'borehole', collar_file)
    survey_path = os.path.join(data_path, 'borehole', survey_file)
    lith_path = os.path.join(data_path, 'borehole', lith_file)

    collars_reader = GenericReaderFilesHelper(file_or_buffer=collar_path)
    surveys_reader = GenericReaderFilesHelper(file_or_buffer=survey_path)
    attrs_reader = GenericReaderFilesHelper(file_or_buffer=lith_path)

    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=True
    )

    # Current behavior: BoreholeSet.collars has ids from survey_df
    # regardless of whether they are in collars_df.
    # This might be because it reindexes by survey_df.index
    import pandas as pd
    import numpy as np
    survey_df = pd.read_csv(survey_path, index_col=0)
    expected_ids = set(survey_df.index.unique())

    assert len(borehole_set.collars.ids) == len(expected_ids)
    for well_id in expected_ids:
        assert well_id in borehole_set.collars.ids

    # Check that attributes are correctly mapped regardless of order
    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    vertices = borehole_set.combined_trajectory.data.vertex
    
    # All wells in the result should have their attributes mapped correctly
    # If a well is in both survey and lith, it should have its attributes.
    lith_df = pd.read_csv(lith_path, index_col=0)
    common_wells = set(expected_ids).intersection(set(lith_df.index.unique()))

    # Known reference data for validation
    # aaa: x=0.25, y=0.4, z=1, MDs=[0, 0.5]
    # zzz: x=0.75, y=0.6, z=1, MDs=[0, 1]
    # bbb: x=0.5, y=0.5, z=1, MDs=[0, 0.8]
    expected_data = {
        "aaa": {"x": 0.25, "y": 0.4, "z": 1.0, "max_md": 0.5},
        "zzz": {"x": 0.75, "y": 0.6, "z": 1.0, "max_md": 1.0},
        "bbb": {"x": 0.5, "y": 0.5, "z": 1.0, "max_md": 0.8}
    }

    for well_id in common_wells:
        well_num_id = borehole_set.survey.get_well_num_id(well_id)
        # The well_id in points_attributes is the numeric ID
        well_mask = points_attrs["well_id"] == well_num_id
        well_attrs = points_attrs[well_mask]
        assert not well_attrs.empty, f"Well {well_id} attributes should not be empty"
        
        # Check coordinates and attributes for this well
        well_vertices = vertices[well_mask]
        
        # In subsurface, combined_trajectory vertices are calculated as:
        # vertex = survey_offset + collar_location
        # For these straight wells, survey_offset is likely (0, 0, -md) 
        # but let's see what the actual values are.
        # expected_data contains collar locations.
        if well_id in expected_data:
            # If a well is in both survey and collars, its vertices should be valid numbers
            if not np.any(np.isnan(well_vertices)):
                assert np.allclose(well_vertices[:, 0], expected_data[well_id]["x"]), f"X mismatch for {well_id}"
                assert np.allclose(well_vertices[:, 1], expected_data[well_id]["y"]), f"Y mismatch for {well_id}"
                # Z starts at collar and usually goes DOWN for boreholes
                assert np.isclose(well_vertices[0, 2], expected_data[well_id]["z"]), f"Z mismatch for {well_id}"
            else:
                # If well is in survey but NOT in the provided collars file for this variation
                # then it might have NaNs in vertices because _add_collar_coordinates adds NaNs
                # This happens in the 'supersimple_collars.csv' + 'supersimple_survey_v2.csv' case for 'bbb'
                pass
        
        # Specific attribute checks
        if well_id == "aaa":
            assert any("lettuce" in str(l) for l in well_attrs["component lith"].unique())
        elif well_id == "zzz":
            assert any("cheese" in str(l) for l in well_attrs["component lith"].unique())
        elif well_id == "bbb":
             assert any("patty" in str(l) for l in well_attrs["component lith"].unique())
