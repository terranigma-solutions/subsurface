import pathlib
from typing import TextIO, Union

import numpy as np

from .... import optional_requirements


def dxf_from_file_to_vertex(file_path: str):
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.readfile(file_path)
    vertex = []
    entity = dataset.modelspace()
    for e in entity:
        vertex.append(e[0])
        vertex.append(e[1])
        vertex.append(e[2])
    vertex = np.array(vertex)
    vertex = np.unique(vertex, axis=0)
    return vertex


def dxf_from_stream_to_vertex(stream: TextIO):
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.read(stream)
    vertex = []
    entity = dataset.modelspace()
    for e in entity:
        vertex.append(e[0])
        vertex.append(e[1])
        vertex.append(e[2])
    vertex = np.array(vertex)
    vertex = np.unique(vertex, axis=0)
    return vertex


def dxf_file_to_unstruct_input(file: Union[str, pathlib.Path]):
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.readfile(file)
    cell_attr_int, cell_attr_map, cells, vertex = _dxf_dataset_to_unstruct_input(dataset)
    # Check if empty and raise error
    if vertex.size == 0:
        raise ValueError("The dxf file does not contain any 3DFACE entities.")
    
    return vertex, cells, cell_attr_int, cell_attr_map


def dxf_stream_to_unstruct_input(stream: TextIO):
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.read(stream)
    cell_attr_int, cell_attr_map, cells, vertex = _dxf_dataset_to_unstruct_input(dataset)

    return vertex, cells, cell_attr_int, cell_attr_map


def _dxf_dataset_to_unstruct_input(dataset):
    vertex = []
    cell_attr = []
    entity = dataset.modelspace()
    for e in entity:
        if e.dxftype() != "3DFACE":
            continue
        vertex.append(e[0])
        vertex.append(e[1])
        vertex.append(e[2])
        cell_attr.append(e.dxf.get("layer"))
    vertex = np.array(vertex)
    cells = np.arange(0, vertex.shape[0]).reshape(-1, 3)
    cell_attr_int, cell_attr_map = _map_cell_attr_strings_to_integers(cell_attr)
    return cell_attr_int, cell_attr_map, cells, vertex


def _map_cell_attr_strings_to_integers(cell_attr):
    # Get unique sorted values from the cell_attr array
    unique_values = np.unique(cell_attr)
    sorted_unique_values = sorted(unique_values)

    # Create a mapping from string values to integers starting from 1
    value_to_int_mapping = {str(value): index + 1 for index, value in enumerate(sorted_unique_values)}

    # Map the original cell_attr values to their corresponding integers
    cell_attr_int = np.array([value_to_int_mapping[value] for value in cell_attr])

    return cell_attr_int, value_to_int_mapping
