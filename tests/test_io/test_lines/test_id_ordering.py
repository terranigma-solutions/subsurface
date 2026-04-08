import pandas as pd
import numpy as np
import os
import pytest
from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper

def test_borehole_id_ordering_bug():
    """Test that borehole data is correctly mapped to IDs even if the order in files is different.
    
    This targets a bug where 'aaa' and 'zzz' were swapped if their order in the collars file
    differed from the survey file's implicit order.
    """
    # Create data where alphabetical order and file order are different
    # aaa is shorter (0.5), zzz is longer (1.0)
    
    # Collars: zzz then aaa
    collars_df = pd.DataFrame({
        'x': [10, 0],
        'y': [0, 0],
        'z': [0, 0]
    }, index=['zzz', 'aaa'])
    
    # Survey: zzz then aaa
    survey_df = pd.DataFrame({
        'md': [0.0, 0.5, 1.0, 0.0, 0.5],
        'inc': [180, 180, 180, 180, 180],
        'azi': [0, 0, 0, 0, 0]
    }, index=['zzz', 'zzz', 'zzz', 'aaa', 'aaa'])
    
    # Lith: zzz then aaa
    lith_df = pd.DataFrame({
        'top': [0.0, 0.5, 0.0],
        'base': [0.5, 1.0, 0.5],
        'component lith': ['L1', 'L2', 'L1']
    }, index=['zzz', 'zzz', 'aaa'])

    collars_file = 'test_ids_collars.csv'
    survey_file = 'test_ids_survey.csv'
    lith_file = 'test_ids_lith.csv'
    
    collars_df.to_csv(collars_file)
    survey_df.to_csv(survey_file)
    lith_df.to_csv(lith_file)

    try:
        borehole_set = read_wells(
            collars_reader=GenericReaderFilesHelper(file_or_buffer=collars_file),
            surveys_reader=GenericReaderFilesHelper(file_or_buffer=survey_file),
            attrs_reader=GenericReaderFilesHelper(file_or_buffer=lith_file),
            is_lith_attr=True,
            add_attrs_as_nodes=True
        )

        # Check lengths
        traj = borehole_set.combined_trajectory.data
        vertex_attrs = traj.points_attributes
        
        # Borehole 'aaa' should have max MD 0.5
        aaa_id = borehole_set.survey.get_well_num_id('aaa')
        aaa_points = vertex_attrs[vertex_attrs['well_id'] == aaa_id]
        assert aaa_points['measured_depths'].max() == 0.5
        
        # Borehole 'zzz' should have max MD 1.0
        zzz_id = borehole_set.survey.get_well_num_id('zzz')
        zzz_points = vertex_attrs[vertex_attrs['well_id'] == zzz_id]
        assert zzz_points['measured_depths'].max() == 1.0
        
    finally:
        for f in [collars_file, survey_file, lith_file]:
            if os.path.exists(f):
                os.remove(f)
