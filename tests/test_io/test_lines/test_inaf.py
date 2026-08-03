import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subsurface.api.reader.read_wells import read_wells
from subsurface.core.geological_formats import BoreholeSet, Collars, Survey
from subsurface.core.geological_formats.boreholes.boreholes import MergeOptions
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.modules.reader.wells.read_borehole_interface import (
    read_attributes,
    read_collar,
    read_survey,
)
from tests.conftest import RequirementsLevel
from tests.test_io.test_lines._aux_func import _plot

_DEVOPS_PATH = Path(
    os.getenv("TERRA_PATH_DEVOPS", "/home/leguark/DevOps/SubsurfaceTestData")
)
_DATA_PATH = _DEVOPS_PATH / "boreholes" / "INAF"

pytestmark = [
        pytest.mark.skipif(
            RequirementsLevel.READ_WELL
            not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
            reason="Need to set the READ_WELL requirement level",
        ),
        pytest.mark.skipif(
            not _DATA_PATH.is_dir(),
            reason=f"INAF test data not found at {_DATA_PATH}",
        ),
]


def _reader(filename: str) -> GenericReaderFilesHelper:
    return GenericReaderFilesHelper(file_or_buffer=_DATA_PATH / filename)


def _read_inaf_without_single_station_validation() -> BoreholeSet:
    collars_df = read_collar(_reader("collars.csv"))
    survey_df = read_survey(_reader("survey.csv"), validate_survey=False)
    attributes_df = read_attributes(_reader("attributes.csv"), is_lith=True)

    survey = Survey.from_df(
        survey_df=survey_df,
        attr_df=attributes_df,
        number_nodes=2,
    )
    survey.update_survey_with_lith(attributes_df)

    return BoreholeSet(
        collars=Collars.from_df(collars_df),
        survey=survey,
        merge_option=MergeOptions.INTERSECT,
    )


def test_inaf_source_data_characteristics():
    collars = pd.read_csv(_DATA_PATH / "collars.csv")
    survey = pd.read_csv(_DATA_PATH / "survey.csv")
    attributes = pd.read_csv(_DATA_PATH / "attributes.csv")

    assert len(collars) == 156
    assert len(survey) == 156
    assert len(attributes) == 7514
    assert set(collars["id"]) == set(survey["id"]) == set(attributes["id"])

    # These are total-depth records, not conventional multi-station surveys.
    assert survey.groupby("id").size().eq(1).all()
    assert survey["md"].gt(0).all()
    assert survey["azi"].eq(0).all()
    assert survey["inc"].eq(180).all()

    # The lithology records include point observations, gaps, and overlaps.
    assert attributes["top"].le(attributes["base"]).all()
    assert attributes["top"].eq(attributes["base"]).sum() == 1504
    assert attributes.duplicated(["id", "top", "base"], keep=False).sum() == 6

    interval_stats = []
    for _, intervals in attributes.sort_values(["id", "top", "base"]).groupby("id"):
        previous_base = intervals["base"].cummax().shift()
        interval_stats.append(
            (
                    (intervals["top"] < previous_base).any(),
                    (intervals["top"] > previous_base).any(),
            )
        )

    assert sum(has_overlap for has_overlap, _ in interval_stats) == 35
    assert sum(has_gap for _, has_gap in interval_stats) == 94


def test_inaf_public_reader():
    with pytest.warns(
            UserWarning,
            match=(
                    "1504 zero-thickness intervals, overlapping intervals in 35 wells, "
                    "and 3 interval keys with conflicting lithologies"
            ),
    ):
        boreholes = read_wells(
            collars_reader=_reader("collars.csv"),
            surveys_reader=_reader("survey.csv"),
            attrs_reader=_reader("attributes.csv"),
            is_lith_attr=True,
            add_attrs_as_nodes=True,
        )

    assert len(boreholes.survey.ids) == 156
    assert len(boreholes.collars.ids) == 156
    assert (
            "component lith"
            in boreholes.combined_trajectory.data.points_attributes.columns
    )


def test_inaf_import_without_single_station_validation():
    with pytest.warns(UserWarning, match="1504 zero-thickness intervals"):
        boreholes = _read_inaf_without_single_station_validation()

    assert len(boreholes.collars.ids) == 156
    assert len(boreholes.survey.ids) == 156

    trajectory = boreholes.combined_trajectory.data
    point_attributes = trajectory.points_attributes
    assert np.isfinite(trajectory.vertex).all()
    assert point_attributes["component lith"].nunique() == 11
    assert point_attributes["component lith"].notna().all()
    assert "lith_ids" in point_attributes.columns
    assert "lith_id_mapper" in trajectory.data.attrs

    first_well = "MCDH_002_1_33"
    first_well_id = boreholes.survey.get_well_num_id(first_well)
    first_well_mask = point_attributes["well_id"].eq(first_well_id).to_numpy()
    first_well_vertices = trajectory.vertex[first_well_mask]

    assert np.allclose(first_well_vertices[:, 0], 497939.0)
    assert np.allclose(first_well_vertices[:, 1], 3875303.0)
    assert np.isclose(first_well_vertices[:, 2].max(), 1713.5672607421875)
    assert np.isclose(
        first_well_vertices[:, 2].min(),
        1713.5672607421875 - 33.22,
    )
    if PLOT := False:
        _plot(
            scalar="lith_ids",
            trajectory=boreholes.combined_trajectory,
            collars=boreholes.collars,
            image_2d=False,
            ve=10
        )
