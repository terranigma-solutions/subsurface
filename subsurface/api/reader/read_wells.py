from subsurface.core.geological_formats.boreholes.boreholes import MergeOptions

from subsurface.core.structs.base_structures.base_structures_enum import SpecialCellCase

from subsurface.core.structs import UnstructuredData, PointSet

from subsurface import optional_requirements
from subsurface.modules.reader.wells.read_borehole_interface import read_collar, read_survey, read_lith, read_attributes

from subsurface.core.geological_formats import BoreholeSet, Collars, Survey

from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper


def read_wells(
        collars_reader: GenericReaderFilesHelper,
        surveys_reader: GenericReaderFilesHelper,
        attrs_reader: GenericReaderFilesHelper,
        is_lith_attr: bool
) -> BoreholeSet:
    # ! FIGUROUT IF WE NEED LITH
    
    
    pd = optional_requirements.require_pandas()
    collar_df: pd.DataFrame = read_collar(collars_reader)
    # TODO: df to unstruct
    unstruc: UnstructuredData = UnstructuredData.from_array(
        vertex=collar_df[["x", "y", "z"]].values,
        cells=SpecialCellCase.POINTS
    )
    points = PointSet(data=unstruc)
    collars: Collars = Collars(
        ids=collar_df.index.to_list(),
        collar_loc=points
    )

    survey_df: pd.DataFrame = read_survey(surveys_reader)

    survey: Survey = Survey.from_df(survey_df)

    # Check if component lith is in columns or columns_map
    lith: pd.DataFrame = read_attributes(attrs_reader, is_lith=is_lith_attr)

    survey.update_survey_with_lith(lith)

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    return borehole_set
