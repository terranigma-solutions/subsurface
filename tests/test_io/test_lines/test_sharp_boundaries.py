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


def test_sharp_lith_boundaies(data_path):
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
        duplicate_attr_depths=True,
        number_nodes=0
    )

    assert len(borehole_set.collars.ids) == 2

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    well_aaa_id = borehole_set.survey.get_well_num_id("aaa")
    well_aaa_attrs = points_attrs[points_attrs["well_id"] == well_aaa_id]

    # Check for lettuce which only appears in aaa
    assert any("lettuce" in str(l) for l in well_aaa_attrs["component lith"].unique())
    
    print(borehole_set.combined_trajectory.data.points_attributes)

    if PLOT or True:
        _plot(
            scalar="lith_ids",
            trajectory=borehole_set.combined_trajectory,
            collars=borehole_set.collars,
            image_2d=False,
            radius=0.01
        )
