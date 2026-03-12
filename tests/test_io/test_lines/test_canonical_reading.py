import pandas as pd
from subsurface.api.reader.read_wells import read_wells
from subsurface.core.reader_helpers.readers_data import GenericReaderFilesHelper
import os

def test_read_canonical():
    # Paths to canonical files
    collar_path = "examples/data/borehole_canonical/collars.csv"
    survey_path = "examples/data/borehole_canonical/survey.csv"
    lith_path = "examples/data/borehole_canonical/lithology.csv"

    # Create readers with NO arguments except file path
    # GenericReaderFilesHelper has some defaults but we want to see if they match canonical
    # We'll use defaults as much as possible
    collars_reader = GenericReaderFilesHelper(file_or_buffer=collar_path)
    surveys_reader = GenericReaderFilesHelper(file_or_buffer=survey_path)
    attrs_reader = GenericReaderFilesHelper(file_or_buffer=lith_path)

    # Read wells
    borehole_set = read_wells(
        collars_reader=collars_reader,
        surveys_reader=surveys_reader,
        attrs_reader=attrs_reader,
        is_lith_attr=True
    )

    print("BoreholeSet successfully created from canonical files!")
    print(f"Boreholes: {borehole_set.collars.ids}")
    
    # Verify some data
    assert len(borehole_set.collars.ids) == 2
    assert "well_01" in borehole_set.collars.ids
    assert "well_02" in borehole_set.collars.ids
    
    # Check if survey was correctly read (inc should be present)
    # The read_wells function calls Survey.from_df which corrects angles and sets up trajectories
    
    print("Verification complete.")

if __name__ == "__main__":
    test_read_canonical()
