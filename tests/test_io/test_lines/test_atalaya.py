import os

import numpy as np
import pandas as pd
import pytest

from subsurface.core.geological_formats.boreholes.boreholes import BoreholeSet, MergeOptions
from subsurface.core.geological_formats.boreholes.collars import Collars
from subsurface.core.geological_formats.boreholes.survey import Survey
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
from subsurface.api.reader.read_wells import read_wells
from tests.conftest import RequirementsLevel

_TERRA_PATH = os.getenv("TERRA_PATH_DEVOPS")
_DATA_PATH = os.path.join(_TERRA_PATH, "boreholes", "Atalaya")

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_WELL) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set the READ_WELL",
)


def _read_collar_df():
    df = pd.read_csv(
        os.path.join(_DATA_PATH, "Collar_Mojarra.csv"),
        sep=";",
        encoding="latin-1",
        index_col=0,
    )
    df.rename(
        columns={
            "BHID": "id",
            "XCOLLAR": "x",
            "YCOLLAR": "y",
            "ZCOLLAR": "z",
        },
        inplace=True,
    )
    return df


def _read_survey_df():
    df = pd.read_csv(
        os.path.join(_DATA_PATH, "Survey_Mojarra.csv"),
        sep=";",
        index_col=0,
    )
    df.rename(
        columns={
            "AT": "md",
            "BRG": "azi",
            "DIP": "dip",
        },
        inplace=True,
    )
    df["inc"] = 90 - df["dip"]
    df.drop(columns=["dip"], inplace=True)
    df = df[df.index.duplicated(keep=False)]
    df.dropna(subset=["inc", "azi"], inplace=True)
    return df


def _read_lith_df():
    df = pd.read_csv(
        os.path.join(_DATA_PATH, "Litho_Mojarra.csv"),
        sep=";",
        encoding="latin-1",
        index_col=0,
    )
    df.rename(
        columns={
            "BHID": "id",
            "FROM": "top",
            "TO": "base",
            "LITHO": "component lith",
        },
        inplace=True,
    )
    return df


def _read_assay_df():
    df = pd.read_csv(
        os.path.join(_DATA_PATH, "Assay_Mojarra.csv"),
        sep=";",
        index_col=0,
    )
    df.rename(
        columns={
            "HoleID": "id",
            "From": "top",
            "To": "base",
        },
        inplace=True,
    )
    return df


def _df_to_dict(df: pd.DataFrame) -> dict:
    return {
        "data": df.values.tolist(),
        "columns": list(df.columns),
        "index": df.index.tolist(),
    }


def _clean_assay_numerics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in ("HoleID", "From", "To"):
            continue
        df[col] = pd.to_numeric(
            df[col].replace(r"^(n\.a\.|<.*)$", None, regex=True),
            errors="coerce",
        )
    return df


def test_read_atalaya_collar():
    df = _read_collar_df()
    assert not df.empty
    assert len(df) == 29
    assert "x" in df.columns
    assert "y" in df.columns
    assert "z" in df.columns

    assert "EST01" in df.index
    assert "MR01" in df.index
    assert "MR24" in df.index

    collars = Collars.from_df(df)
    assert collars.collar_loc.n_points == 29
    assert collars.collar_loc.n_points == len(collars.ids)


def test_read_atalaya_survey():
    df = _read_survey_df()
    assert not df.empty
    assert "md" in df.columns
    assert "inc" in df.columns
    assert "azi" in df.columns

    survey = Survey.from_df(df)
    assert survey.survey_trajectory.data.n_points > 0


def test_read_atalaya_full_lithology():
    collar_df = _read_collar_df()
    survey_df = _read_survey_df()
    lith_df = _read_lith_df()

    survey = Survey.from_df(
        survey_df=survey_df,
        attr_df=lith_df,
        number_nodes=10,
        duplicate_attr_depths=True,
    )
    survey.update_survey_with_lith(lith_df)

    collars = Collars.from_df(collar_df)
    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT,
    )

    assert "EST01" in borehole_set.collars.ids
    assert "MR24" in borehole_set.collars.ids

    collar_count = len(borehole_set.collars.ids)
    assert collar_count >= 29

    assert hasattr(borehole_set.survey, "survey_trajectory")

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    assert "component lith" in points_attrs.columns
    assert "lith_ids" in points_attrs.columns

    est01_id = borehole_set.survey.get_well_num_id("EST01")
    est01_attrs = points_attrs[points_attrs["well_id"] == est01_id]
    assert not est01_attrs.empty

    lith_codes = est01_attrs["component lith"].dropna().unique()
    assert len(lith_codes) > 0
    assert any("VR" in str(c) or c == "VR" for c in lith_codes)
    assert any("VT" in str(c) or c == "VT" for c in lith_codes)


def test_read_atalaya_assays():
    collar_df = _read_collar_df()
    survey_df = _read_survey_df()
    attrs = _read_assay_df()

    assert not attrs.empty
    raw_assay_cols = [c for c in attrs.columns if c not in ("top", "base")]
    assert len(raw_assay_cols) > 0
    assert "Cu %" in raw_assay_cols
    assert "Au ppm" in raw_assay_cols

    for col in attrs.columns:
        if col not in ("top", "base"):
            attrs[col] = pd.to_numeric(attrs[col], errors="coerce")

    collars = Collars.from_df(collar_df)
    survey = Survey.from_df(
        survey_df=survey_df,
        attr_df=attrs,
        number_nodes=10,
        duplicate_attr_depths=True,
    )
    survey.update_survey_with_attr(attrs)

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT,
    )

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    assert "Cu %" in points_attrs.columns
    assert "Au ppm" in points_attrs.columns

    mr01_id = borehole_set.survey.get_well_num_id("MR01")
    mr01_attrs = points_attrs[points_attrs["well_id"] == mr01_id]
    assert not mr01_attrs.empty

    cu_values = mr01_attrs["Cu %"].dropna()
    assert len(cu_values) > 0

    mr01_vertices = borehole_set.combined_trajectory.data.vertex[
        points_attrs["well_id"] == mr01_id
    ]
    assert mr01_vertices.shape[0] > 0

    expected_x = 692622.949
    expected_y = 4158049.67
    assert np.isclose(mr01_vertices[0, 0], expected_x, rtol=1e-5)
    assert np.isclose(mr01_vertices[0, 1], expected_y, rtol=1e-5)


def test_read_atalaya_lithology_with_explicit_mapping():
    collar_df = _read_collar_df()
    survey_df = _read_survey_df()
    lith_df = _read_lith_df()

    survey = Survey.from_df(
        survey_df=survey_df,
        attr_df=lith_df,
        number_nodes=0,
        duplicate_attr_depths=True,
    )
    survey.update_survey_with_lith(lith_df)

    collars = Collars.from_df(collar_df)
    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT,
    )

    assert len(borehole_set.collars.ids) >= 29

    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    assert "component lith" in points_attrs.columns
    assert "lith_ids" in points_attrs.columns

    vertices = borehole_set.combined_trajectory.data.vertex
    n = vertices.shape[0]
    assert n > 0

    unique_lith_ids = points_attrs["lith_ids"].unique()
    assert len(unique_lith_ids) > 1


def test_read_atalaya_assays_via_dict():
    collar_df = pd.read_csv(
        os.path.join(_DATA_PATH, "Collar_Mojarra.csv"),
        sep=";",
        encoding="latin-1",
        index_col=0,
    )
    survey_df = pd.read_csv(
        os.path.join(_DATA_PATH, "Survey_Mojarra.csv"),
        sep=";",
        index_col=0,
    )
    survey_df.index = survey_df.index.where(survey_df.index.notna(), None)
    survey_df = survey_df[survey_df.index.notna() & survey_df.index.notnull()]
    survey_df = survey_df[survey_df.index.duplicated(keep=False)]

    raw_assay_df = pd.read_csv(
        os.path.join(_DATA_PATH, "Assay_Mojarra.csv"),
        sep=";",
        index_col=0,
    )
    assay_df = _clean_assay_numerics(raw_assay_df)

    collar_dict = _df_to_dict(collar_df)
    survey_dict = _df_to_dict(survey_df)
    assay_dict = _df_to_dict(assay_df)

    collar_reader = GenericReaderFilesHelper(
        file_or_buffer=collar_dict,
        columns_map={
            "BHID": "id",
            "XCOLLAR": "x",
            "YCOLLAR": "y",
            "ZCOLLAR": "z",
        },
    )
    survey_reader = GenericReaderFilesHelper(
        file_or_buffer=survey_dict,
        columns_map={
            "AT": "md",
            "BRG": "azi",
            "DIP": "dip",
        },
    )
    assay_reader = GenericReaderFilesHelper(
        file_or_buffer=assay_dict,
        columns_map={
            "HoleID": "id",
            "From": "top",
            "To": "base",
        },
    )

    borehole_set = read_wells(
        collars_reader=collar_reader,
        surveys_reader=survey_reader,
        attrs_reader=assay_reader,
        is_lith_attr=False,
        add_attrs_as_nodes=True,
    )

    assert "Cu %" in borehole_set.combined_trajectory.data.points_attributes.columns
    assert "Au ppm" in borehole_set.combined_trajectory.data.points_attributes.columns

    mr01_id = borehole_set.survey.get_well_num_id("MR01")
    points_attrs = borehole_set.combined_trajectory.data.points_attributes
    mr01_attrs = points_attrs[points_attrs["well_id"] == mr01_id]
    assert not mr01_attrs.empty

    cu_values = mr01_attrs["Cu %"].dropna()
    assert len(cu_values) > 0
    au_values = mr01_attrs["Au ppm"].dropna()
    assert len(au_values) > 0


def _make_raw_assay_dict():
    raw = pd.read_csv(
        os.path.join(_DATA_PATH, "Assay_Mojarra.csv"),
        sep=";",
        index_col=0,
    )
    return _df_to_dict(raw)


class TestDiagnosticRawDictGaps:
    """
    Diagnostic tests showing the behavior of read_wells with raw (uncleaned)
    dict/JSON input, and how the coerce_numeric fix resolves the issues.

    FIXED: raw survey no longer crashes (NaN rows are dropped in _validate_survey_data).
           raw assay with coerce_numeric=True now interpolates correctly.
    """

    def test_raw_survey_no_longer_crashes(self):
        raw_survey = pd.read_csv(
            os.path.join(_DATA_PATH, "Survey_Mojarra.csv"),
            sep=";",
            index_col=0,
        )
        survey_dict = _df_to_dict(raw_survey)

        collar_reader = GenericReaderFilesHelper(
            file_or_buffer=_df_to_dict(pd.read_csv(
                os.path.join(_DATA_PATH, "Collar_Mojarra.csv"),
                sep=";", encoding="latin-1", index_col=0,
            )),
            columns_map={"BHID": "id", "XCOLLAR": "x", "YCOLLAR": "y", "ZCOLLAR": "z"},
        )
        survey_reader = GenericReaderFilesHelper(
            file_or_buffer=survey_dict,
            columns_map={"AT": "md", "BRG": "azi", "DIP": "dip"},
        )
        attr_reader = GenericReaderFilesHelper(
            file_or_buffer=_make_raw_assay_dict(),
            columns_map={"HoleID": "id", "From": "top", "To": "base"},
        )

        # Previously crashed in _correct_angles, now succeeds.
        borehole_set = read_wells(
            collars_reader=collar_reader,
            surveys_reader=survey_reader,
            attrs_reader=attr_reader,
            is_lith_attr=False,
            add_attrs_as_nodes=True,
        )
        assert borehole_set.combined_trajectory.data.n_points > 0

    def test_raw_assay_columns_exist_but_all_none(self):
        raw_survey = pd.read_csv(
            os.path.join(_DATA_PATH, "Survey_Mojarra.csv"),
            sep=";",
            index_col=0,
        )
        raw_survey.index = raw_survey.index.where(raw_survey.index.notna(), None)
        raw_survey = raw_survey[raw_survey.index.notna() & raw_survey.index.notnull()]
        raw_survey = raw_survey[raw_survey.index.duplicated(keep=False)]
        survey_dict = _df_to_dict(raw_survey)

        collar_reader = GenericReaderFilesHelper(
            file_or_buffer=_df_to_dict(pd.read_csv(
                os.path.join(_DATA_PATH, "Collar_Mojarra.csv"),
                sep=";", encoding="latin-1", index_col=0,
            )),
            columns_map={"BHID": "id", "XCOLLAR": "x", "YCOLLAR": "y", "ZCOLLAR": "z"},
        )
        survey_reader = GenericReaderFilesHelper(
            file_or_buffer=survey_dict,
            columns_map={"AT": "md", "BRG": "azi", "DIP": "dip"},
        )
        attr_reader = GenericReaderFilesHelper(
            file_or_buffer=_make_raw_assay_dict(),
            columns_map={"HoleID": "id", "From": "top", "To": "base"},
        )

        with pytest.warns(UserWarning, match="non-numeric dtype"):
            borehole_set = read_wells(
                collars_reader=collar_reader,
                surveys_reader=survey_reader,
                attrs_reader=attr_reader,
                is_lith_attr=False,
                add_attrs_as_nodes=True,
            )

        points_attrs = borehole_set.combined_trajectory.data.points_attributes
        assert "Cu %" in points_attrs.columns
        assert points_attrs["Cu %"].isna().all()
        assert "Au ppm" in points_attrs.columns
        assert points_attrs["Au ppm"].isna().all()

    def test_raw_assay_with_coerce_numeric_works(self):
        raw_survey = pd.read_csv(
            os.path.join(_DATA_PATH, "Survey_Mojarra.csv"),
            sep=";",
            index_col=0,
        )
        raw_survey.index = raw_survey.index.where(raw_survey.index.notna(), None)
        raw_survey = raw_survey[raw_survey.index.notna() & raw_survey.index.notnull()]
        raw_survey = raw_survey[raw_survey.index.duplicated(keep=False)]
        survey_dict = _df_to_dict(raw_survey)

        collar_reader = GenericReaderFilesHelper(
            file_or_buffer=_df_to_dict(pd.read_csv(
                os.path.join(_DATA_PATH, "Collar_Mojarra.csv"),
                sep=";", encoding="latin-1", index_col=0,
            )),
            columns_map={"BHID": "id", "XCOLLAR": "x", "YCOLLAR": "y", "ZCOLLAR": "z"},
        )
        survey_reader = GenericReaderFilesHelper(
            file_or_buffer=survey_dict,
            columns_map={"AT": "md", "BRG": "azi", "DIP": "dip"},
        )
        attr_reader = GenericReaderFilesHelper(
            file_or_buffer=_make_raw_assay_dict(),
            columns_map={"HoleID": "id", "From": "top", "To": "base"},
            coerce_numeric=[
                "Cu %", "S %", "Pb %", "Zn %", "As %", "Sb %", "Bi %",
                "Hg ppm", "Fe %", "Th  ppm", "Ag ppm", "Cd %",
                "Co %", "Ni %", "Mn %", "Al %", "Ca %", "Mg %",
                "Sn ppm", "Mo %", "P %", "Tl %", "Au ppm",
            ],
        )

        borehole_set = read_wells(
            collars_reader=collar_reader,
            surveys_reader=survey_reader,
            attrs_reader=attr_reader,
            is_lith_attr=False,
            add_attrs_as_nodes=True,
        )

        points_attrs = borehole_set.combined_trajectory.data.points_attributes
        assert "Cu %" in points_attrs.columns
        assert "Au ppm" in points_attrs.columns

        mr01_id = borehole_set.survey.get_well_num_id("MR01")
        mr01_attrs = points_attrs[points_attrs["well_id"] == mr01_id]
        assert not mr01_attrs.empty

        cu_values = mr01_attrs["Cu %"].dropna()
        assert len(cu_values) > 0
        au_values = mr01_attrs["Au ppm"].dropna()
        assert len(au_values) > 0
