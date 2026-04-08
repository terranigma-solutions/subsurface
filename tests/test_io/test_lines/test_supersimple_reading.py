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
