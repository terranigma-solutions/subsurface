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
        is_lith_attr: bool,
        number_nodes: int = 10,
        add_attrs_as_nodes: bool = False,
        duplicate_attr_depths: bool = False 
) -> BoreholeSet:
    # ! FIGUROUT IF WE NEED LITH
    
    
    pd = optional_requirements.require_pandas()
    try:
        collar_df: pd.DataFrame = read_collar(collars_reader)
        unstruc: UnstructuredData = UnstructuredData.from_array(
            vertex=collar_df[["x", "y", "z"]].values,
            cells=SpecialCellCase.POINTS
        )
        points = PointSet(data=unstruc)
        collars: Collars = Collars(
            ids=collar_df.index.to_list(),
            collar_loc=points
        )
    except Exception as e:
        raise ValueError(f"Error while reading collars: {e}")   

    
    try:
        survey_df: pd.DataFrame = read_survey(surveys_reader)
    except Exception as e:
        raise ValueError(f"Error while reading surveys: {e}")
    
    try:
        attrs: pd.DataFrame = read_attributes(attrs_reader, is_lith=is_lith_attr)
    except Exception as e:
        raise ValueError(f"Error while reading attributes: {e}")
    
    try:
        if add_attrs_as_nodes:
            attr_df = attrs
        else:
            attr_df = None

        survey: Survey = Survey.from_df(
            survey_df=survey_df,
            attr_df=attr_df,
            number_nodes=number_nodes,
            duplicate_attr_depths=duplicate_attr_depths
        )

        # Check if component lith is in columns or columns_map
        survey.update_survey_with_lith(attrs)
    except Exception as e:
        raise ValueError(f"Error while creating survey: {e}")

    borehole_set = BoreholeSet(
        collars=collars,
        survey=survey,
        merge_option=MergeOptions.INTERSECT
    )

    return borehole_set
