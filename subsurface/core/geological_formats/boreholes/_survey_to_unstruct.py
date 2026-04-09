from typing import Hashable, Optional

import numpy as np
import pandas as pd

from subsurface import optional_requirements
from ...structs.base_structures import UnstructuredData


def data_frame_to_unstructured_data(survey_df: 'pd.DataFrame', number_nodes: int, attr_df: Optional['pd.DataFrame'] = None,
                                     duplicate_attr_depths: bool = False) -> UnstructuredData:
    wp = optional_requirements.require_wellpathpy()

    cum_vertex: np.ndarray = np.empty((0, 3), dtype=np.float32)
    cells: np.ndarray = np.empty((0, 2), dtype=np.int_)
    cell_attr: pd.DataFrame = pd.DataFrame(columns=['well_id'], dtype=np.float32)
    vertex_attr: pd.DataFrame = pd.DataFrame()

    # Create well_id_mapper based on the order of groupby to ensure consistent e indexing
    well_id_mapper = {borehole_id: e for e, (borehole_id, _) in enumerate(survey_df.groupby(level=0, sort=True))}

    for e, (borehole_id, data) in enumerate(survey_df.groupby(level=0, sort=True)):
        md = data['md'].values
        inc = data['inc'].values
        azi = data['azi'].values

        if md[0] != 0:
            md = np.insert(md, 0, 0)
            inc = np.insert(inc, 0, inc[0])
            azi = np.insert(azi, 0, azi[0])

        dev = wp.deviation(
            md=md,
            inc=inc,
            azi=azi
        )

        md_min = dev.md.min()
        md_max = dev.md.max()

        attr_depths = _grab_depths_from_attr(
            attr_df=attr_df,
            borehole_id=borehole_id,
            duplicate_attr_depths=duplicate_attr_depths,
            md_max=md_max,
            md_min=md_min
        )

        # Now combine attr_depths with depths
        md_min = dev.md.min()
        md_max = dev.md.max()
        depths = np.linspace(md_min, md_max, number_nodes)
        depths = np.union1d(depths, attr_depths)
        depths.sort()

        # Resample positions at depths
        pos = dev.minimum_curvature().resample(depths=depths)
        vertex_count = cum_vertex.shape[0]

        this_well_vertex = np.vstack([pos.easting, pos.northing, pos.depth]).T
        cum_vertex = np.vstack([cum_vertex, this_well_vertex])
        measured_depths = _calculate_distances(
            array_of_vertices=this_well_vertex,
            start_md = depths[0]
        )

        n_vertex_shift_0 = np.arange(0, len(pos.depth) - 1, dtype=np.int_)
        n_vertex_shift_1 = np.arange(1, len(pos.depth), dtype=np.int_)
        cell_per_well = np.vstack([n_vertex_shift_0, n_vertex_shift_1]).T + vertex_count
        cells = np.vstack([cells, cell_per_well])

        attribute_values = np.isin(depths, attr_depths)

        vertex_attr_per_well = pd.DataFrame({
                'well_id'        : [e] * len(pos.depth),
                'measured_depths': measured_depths,
                'is_attr_point'  : attribute_values,
        })

        vertex_attr = pd.concat([vertex_attr, vertex_attr_per_well], ignore_index=True)

        # Add the id (e), to cell_attr
        cell_attr = pd.concat([cell_attr, pd.DataFrame({'well_id': [e] * len(cell_per_well)})], ignore_index=True)

    unstruct = UnstructuredData.from_array(
        vertex=cum_vertex,
        cells=cells.astype(int),
        vertex_attr=vertex_attr.reset_index(drop=True),
        cells_attr=cell_attr.reset_index(drop=True)
    )

    unstruct.data.attrs["well_id_mapper"] = well_id_mapper

    return unstruct


def _grab_depths_from_attr(
        attr_df: Optional[pd.DataFrame],
        borehole_id: Hashable,
        duplicate_attr_depths: bool,
        md_max: float,
        md_min: float
) -> np.ndarray:
    # Initialize attr_depths as empty array
    attr_depths = np.array([], dtype=float)

    if attr_df is None or ("top" not in attr_df.columns and "base" not in attr_df.columns):
        return attr_depths

    try:
        vals = attr_df.loc[borehole_id]

        tops = np.array([], dtype=float)
        bases = np.array([], dtype=float)

        if 'top' in vals:
            if isinstance(vals, pd.DataFrame):
                tops = vals['top'].values.flatten()
            else:
                tops = np.array([vals['top']])
            # Convert to float and remove NaNs
            tops = tops.astype(float)
            tops = tops[~np.isnan(tops)]
            # Clip to within md range
            tops = tops[(tops >= md_min) & (tops <= md_max)]

        if 'base' in vals:
            if isinstance(vals, pd.DataFrame):
                bases = vals['base'].values.flatten()
            else:
                bases = np.array([vals['base']])
            # Convert to float and remove NaNs
            bases = bases.astype(float)
            bases = bases[~np.isnan(bases)]
            # Clip to within md range
            bases = bases[(bases >= md_min) & (bases <= md_max)]

        # Combine tops and bases into attr_depths
        attr_depths = np.unique(np.concatenate((tops, bases)))

    except KeyError:
        # No attributes for this borehole_id or missing columns
        attr_depths = np.array([], dtype=float)

    # If duplicate_attr_depths is True, duplicate every attr_depth with a tiny offset 
    # to fall into both sides of each boundary
    if duplicate_attr_depths and len(attr_depths) > 0:
        tiny_offset = (md_max - md_min) * 1e-6  # A tiny fraction of the depth range
        # Combine original depths, plus offsets, and minus offsets
        duplicated_attr_depths = np.concatenate([
            attr_depths,
            attr_depths + tiny_offset,
            attr_depths - tiny_offset
        ])
        # Ensure the duplicated depths are within the md range
        valid_indices = (duplicated_attr_depths >= md_min) & (duplicated_attr_depths <= md_max)
        attr_depths = np.unique(duplicated_attr_depths[valid_indices])

    return attr_depths


def _calculate_distances(array_of_vertices: np.ndarray, start_md: float = 0) -> np.ndarray:
    # Calculate the differences between consecutive points
    differences = np.diff(array_of_vertices, axis=0)

    # Calculate the Euclidean distance for each pair of consecutive points
    distances = np.linalg.norm(differences, axis=1)
    # Start from start_md instead of 0
    measured_depths = np.insert(np.cumsum(distances) + start_md, 0, start_md)
    return measured_depths
