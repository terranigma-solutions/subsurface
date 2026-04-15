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


def test_read_dflt_borehole_data(data_path):
    # Paths to the dflt space files
    collar_path = os.path.join(data_path, 'borehole', 'dlft_space_collars.csv')
    survey_path = os.path.join(data_path, 'borehole', 'dflt_space_survey.csv')
    attr_path = os.path.join(data_path, 'borehole', 'dflt_space_attributes.csv')

    # Create readers for canonical format (dflt files seem to follow it)
    collars_reader = GenericReaderFilesHelper(file_or_buffer=collar_path)
    surveys_reader = GenericReaderFilesHelper(file_or_buffer=survey_path)
    attrs_reader = GenericReaderFilesHelper(file_or_buffer=attr_path)

    # Read wells as assays (is_lith_attr=False)
    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=False
    )

    # Verification
    assert len(borehole_set.collars.ids) == 2
    assert "well_01" in borehole_set.collars.ids
    assert "well_02" in borehole_set.collars.ids

    # Check survey data (well_02 has a deviation)
    # The survey trajectory should be a LineSet
    assert hasattr(borehole_set.survey, 'survey_trajectory')

    # Check attributes
    # The attributes are stored in the combined_trajectory.data.data
    # We can use the helper method _merge_vertex_data_arrays_to_dataframe or look at points_attributes

    # borehole_set.combined_trajectory is a LineSet
    # borehole_set.combined_trajectory.data is an UnstructuredData
    # borehole_set.combined_trajectory.data.points_attributes is a pandas DataFrame

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    assert "Cu" in points_attrs.columns
    assert "Au" in points_attrs.columns

    well_01_id = borehole_set.survey.get_well_num_id("well_01")
    well_01_attrs = points_attrs[points_attrs["well_id"] == well_01_id]
    assert not well_01_attrs.empty

    # In default mode (add_attrs_as_nodes=False), attributes are interpolated 
    # and might not match the input values exactly at the nodes.
    # We just check that we have some data.

    if PLOT:
        _plot(
            scalar="Cu",
            trajectory=borehole_set.combined_trajectory,
            collars=borehole_set.collars,
            radius=0.01
        )


def test_read_dflt_as_lithology(data_path):
    # Testing the case where we might want to read it as lithology
    # However, the dflt_space_attributes.csv does NOT have 'component lith' column.
    # According to GUIDE.md, this should raise an ValueError (wrapped around AttributeError) if is_lith_attr=True.

    collar_path = os.path.join(data_path, 'borehole', 'dlft_space_collars.csv')
    survey_path = os.path.join(data_path, 'borehole', 'dflt_space_survey.csv')
    attr_path = os.path.join(data_path, 'borehole', 'dflt_space_attributes.csv')

    collars_reader = GenericReaderFilesHelper(file_or_buffer=collar_path)
    surveys_reader = GenericReaderFilesHelper(file_or_buffer=survey_path)
    attrs_reader = GenericReaderFilesHelper(file_or_buffer=attr_path)

    with pytest.raises(ValueError, match="component lith"):
        read_wells(
            collars_reader=collars_reader,
            surveys_reader=surveys_reader,
            attrs_reader=attrs_reader,
            is_lith_attr=True
        )


def test_read_dflt_with_mapping(data_path):
    # Test reading with manual mapping even if it follows canonical
    collar_path = os.path.join(data_path, 'borehole', 'dlft_space_collars.csv')
    survey_path = os.path.join(data_path, 'borehole', 'dflt_space_survey.csv')
    attr_path = os.path.join(data_path, 'borehole', 'dflt_space_attributes.csv')

    collars_reader = GenericReaderFilesHelper(
        file_or_buffer=collar_path,
        columns_map={"id": "id", "x": "x", "y": "y", "z": "z"}
    )
    surveys_reader = GenericReaderFilesHelper(
        file_or_buffer=survey_path,
        columns_map={"id": "id", "md": "md", "inc": "inc", "azi": "azi"}
    )
    attrs_reader = GenericReaderFilesHelper(
        file_or_buffer=attr_path,
        columns_map={"id": "id", "top": "top", "base": "base", "Cu": "Cu", "Au": "Au"}
    )

    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=False,
        add_attrs_as_nodes=True
    )

    assert len(borehole_set.collars.ids) == 2

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    well_01_id = borehole_set.survey.get_well_num_id("well_01")
    well_01_attrs = points_attrs[points_attrs["well_id"] == well_01_id]

    # When add_attrs_as_nodes=True, the nodes should be at the attribute depths
    # Check if some Cu values from dflt_space_attributes.csv are present
    # In this case, 1.0 is a value that appears in the input and should be preserved
    assert 1.0 in well_01_attrs["Cu"].values

    if PLOT:
        _plot(
            scalar="Au",
            trajectory=borehole_set.combined_trajectory,
            collars=borehole_set.collars,
            image_2d=False,
            radius=0.01
        )



def test_read_dflt_with_mapping_tops_and_bottoms(data_path):
    # Test reading with manual mapping even if it follows canonical
    collar_path = os.path.join(data_path, 'borehole', 'dlft_space_collars.csv')
    survey_path = os.path.join(data_path, 'borehole', 'dflt_space_survey.csv')
    attr_path = os.path.join(data_path, 'borehole', 'dflt_space_attributes_discontiniued.csv')

    collars_reader = GenericReaderFilesHelper(
        file_or_buffer=collar_path,
        columns_map={"id": "id", "x": "x", "y": "y", "z": "z"}
    )
    surveys_reader = GenericReaderFilesHelper(
        file_or_buffer=survey_path,
        columns_map={"id": "id", "md": "md", "inc": "inc", "azi": "azi"}
    )
    attrs_reader = GenericReaderFilesHelper(
        file_or_buffer=attr_path,
        columns_map={"id": "id", "top": "top", "base": "base", "Cu": "Cu", "Au": "Au"}
    )

    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=False,
        add_attrs_as_nodes=True,
        duplicate_attr_depths=True,
        number_nodes=0
    )

    assert len(borehole_set.collars.ids) == 2

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    well_01_id = borehole_set.survey.get_well_num_id("well_01")
    well_01_attrs = points_attrs[points_attrs["well_id"] == well_01_id]
    well_02_id = borehole_set.survey.get_well_num_id("well_02")
    well_02_attrs = points_attrs[points_attrs["well_id"] == well_02_id]
    print(well_01_attrs)

    print(well_02_attrs)

    # When add_attrs_as_nodes=True, the nodes should be at the attribute depths
    # Check if some Cu values from dflt_space_attributes.csv are present
    # In this case, 1.0 is a value that appears in the input and should be preserved
    assert 1.0 in well_01_attrs["Cu"].values


    if PLOT:
        _plot(
            scalar="Au",
            trajectory=borehole_set.combined_trajectory,
            collars=borehole_set.collars,
            image_2d=False,
            radius=0.01
        )