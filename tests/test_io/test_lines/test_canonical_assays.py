import os

import pytest

from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from tests.conftest import RequirementsLevel

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_WELL"
)


def test_canonical_assays():
    # Define paths
    base_path = "examples/data/borehole_canonical/"
    collars_path = os.path.join(base_path, "collars.csv")
    survey_path = os.path.join(base_path, "survey.csv")
    assays_path = os.path.join(base_path, "assays.csv")

    # Readers
    collars_reader = GenericReaderFilesHelper(file_or_buffer=collars_path)
    survey_reader = GenericReaderFilesHelper(file_or_buffer=survey_path)
    assays_reader = GenericReaderFilesHelper(file_or_buffer=assays_path)

    # Read wells with assays (not lith)
    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=survey_reader,
        attrs_reader=assays_reader,
        is_lith_attr=False,
        add_attrs_as_nodes=True,
        number_nodes=5,
        duplicate_attr_depths=True
    )

    print("BoreholeSet created successfully from canonical files with assays")
    print(f"Boreholes: {borehole_set.survey.ids}")

    # Check if assays are present in combined_trajectory
    data = borehole_set.combined_trajectory.data.data
    print(f"Available attributes: {data.keys()}")

    # Verify Cu and Au are in vertex_attr if they were mapped
    # Actually read_wells uses read_attributes which returns the whole df if not lith
    # then survey.update_survey_with_lith(attrs) is called (it seems it's used for both lith and attrs in read_wells, though the name is misleading)

    assert "well_id" in data.vertex_attr.values
    # Cu and Au should be there as well
    assert "Cu" in data.vertex_attr.values
    assert "Au" in data.vertex_attr.values


if __name__ == "__main__":
    test_canonical_assays()
