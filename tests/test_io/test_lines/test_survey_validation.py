import warnings

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.wells.read_borehole_interface import (
    read_attributes,
    read_survey,
)


def test_validate_survey_keeps_multistation_and_total_depth_records():
    reader = GenericReaderFilesHelper(
        file_or_buffer={
            "data": [
                [0.0, 0.0, 180.0],
                [10.0, 5.0, 175.0],
                [8.0, 0.0, 180.0],
                [0.0, 0.0, 180.0],
            ],
            "columns": ["md", "azi", "inc"],
            "index": ["multi", "multi", "total-depth", "zero-depth"],
        }
    )

    validated = read_survey(reader)

    assert validated.index.tolist() == ["multi", "multi", "total-depth"]
    assert validated.loc["total-depth", "md"] == 8.0


def test_validate_canonical_lithology_does_not_warn():
    reader = GenericReaderFilesHelper(
        file_or_buffer={
            "data": [[0.0, 5.0, "sand"], [5.0, 10.0, "clay"]],
            "columns": ["top", "base", "component lith"],
            "index": ["well", "well"],
        }
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        attributes = read_attributes(reader, is_lith=True)

    assert not caught_warnings
    assert attributes["component lith"].cat.categories.tolist() == ["clay", "sand"]
