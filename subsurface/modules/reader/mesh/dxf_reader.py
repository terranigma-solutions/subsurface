import pathlib
import warnings
from enum import Flag
from typing import TextIO, Union

import numpy as np

from .... import optional_requirements


class DXFEntityType(Flag):
    """Decides which entity types should be extracted."""
    FACE3D = 2 ** 0  # Extract only 3DFACE
    POLYLINE = 2 ** 1  # Extract only POLYLINE
    ALL = FACE3D | POLYLINE  # Extract all entity types


def dxf_from_file_to_vertex(
        file_path: Union[str, pathlib.Path],
        entity_type: DXFEntityType = DXFEntityType.ALL,
) -> np.ndarray:
    """
    Extract vertices from a DXF file, according to the chosen entity_type.
    
    :param file_path: Path to the DXF file.
    :param entity_type: Controls which entity types to extract.
    :return: Unique vertex array [N, 3].
    """
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.readfile(file_path)
    return _extract_vertices_from_dataset(dataset, entity_type)


def dxf_from_stream_to_vertex(
        stream: TextIO,
        entity_type: DXFEntityType = DXFEntityType.ALL,
) -> np.ndarray:
    """
    Extract vertices from a DXF stream, according to the chosen entity_type.
    
    :param stream: A file-like object containing the DXF data.
    :param entity_type: Controls which entity types to extract.
    :return: Unique vertex array [N, 3].
    """
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.read(stream)
    return _extract_vertices_from_dataset(dataset, entity_type)


def dxf_file_to_unstruct_input(
        file: Union[str, pathlib.Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Extract unstructured-mesh-like data (vertex, cells, cell_attr_int, cell_attr_map) 
    from a DXF file using only 3DFACE entities.
    
    :param file: Path to the DXF file.
    :return: (vertex, cells, cell_attr_int, cell_attr_map)
    """
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.readfile(file)
    vertex, cells, cell_attr_int, cell_attr_map = _dxf_dataset_to_unstruct_input(dataset)

    if vertex.size == 0:
        raise ValueError("The DXF file does not contain any 3DFACE entities.")

    return vertex, cells, cell_attr_int, cell_attr_map


def dxf_stream_to_unstruct_input(
        stream: TextIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Extract unstructured-mesh-like data (vertex, cells, cell_attr_int, cell_attr_map) 
    from a DXF stream using only 3DFACE entities.
    
    :param stream: A file-like object containing the DXF data.
    :return: (vertex, cells, cell_attr_int, cell_attr_map)
    """
    ezdxf = optional_requirements.require_ezdxf()
    dataset = ezdxf.read(stream)
    return _dxf_dataset_to_unstruct_input(dataset)


def _extract_vertices_from_dataset(
        dataset,
        entity_type: DXFEntityType = DXFEntityType.ALL
) -> np.ndarray:
    """
    Collect unique vertices from the dataset's modelspace.
    The entity_type flag dictates which entities are extracted.
    """
    vertices = []

    for e in dataset.modelspace():
        match e.dxftype():
            # 3DFACE
            case "3DFACE" if DXFEntityType.FACE3D in entity_type:
                # A 3DFACE entity typically has three corners: e[0], e[1], e[2]
                # Sometimes it can have a fourth corner e[3], which might be equal to e[2].
                # Adjust accordingly if needed:
                vertices.extend([e[0], e[1], e[2]])
                # If you'd like to handle the potential 4th corner,
                # you could do: vertices.append(e[3]) if e[3] != e[2] else None

            # POLYLINE
            case "POLYLINE" if DXFEntityType.POLYLINE in entity_type:
                for v in e.vertices:
                    x, y, z = v.dxf.location.xyz
                    vertices.append([x, y, z])

            # Other / unsupported
            case _:
                # If it doesn't match the chosen entity type(s), we skip or warn.
                # But if you'd prefer to only warn when an entity isn't recognized at all,
                # you can do so here:
                warnings.warn(f"Entity type '{e.dxftype()}' not extracted (flag: {entity_type}).")
                continue

    # Convert to numpy array and ensure uniqueness
    vertices = np.array(vertices)
    if vertices.size == 0:
        return vertices
    return np.unique(vertices, axis=0)


def _dxf_dataset_to_unstruct_input(dataset: 'ezdxf.drawing.Drawing') -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build unstructured-mesh-like data from 3DFACE entities in a dataset:
      - vertex coordinates
      - connectivity in 'cells' array
      - cell attributes in both integer-coded and mapping (string->int) forms

    """
    """
    Build unstructured-mesh-like data from 3DFACE entities in a dataset:
      - vertex coordinates
      - connectivity in 'cells' array
      - cell attributes in both integer-coded and mapping (string->int) forms

    Returns (cell_attr_int, cell_attr_map, cells, vertex).
    """

    def _map_cell_attr_strings_to_integers(cell_attr):
        """
        Map string layer names (or other string attributes) to integer IDs, starting from 1.
        """
        unique_values = np.unique(cell_attr)
        sorted_unique_values = sorted(unique_values)

        # Create mapping from string values to integers (start at 1, 2, 3, ...)
        value_to_int_mapping = {
                str(value): index + 1 for index, value in enumerate(sorted_unique_values)
        }

        cell_attr_int = np.array([value_to_int_mapping[value] for value in cell_attr])
        return cell_attr_int, value_to_int_mapping

    vertex_list = []
    cell_attr = []

    for e in dataset.modelspace():
        match e.dxftype():
            case "3DFACE":
                # Typically 3 corners, but can have 4th corner repeated
                vertex_list.extend([e[0], e[1], e[2]])
                cell_attr.append(e.dxf.get("layer"))
            case _:
                # We ignore other entities in unstructured input
                continue

    vertices = np.array(vertex_list)
    # For each triple (3DFACE), build a cell (triangle)
    cells = np.arange(0, vertices.shape[0]).reshape(-1, 3)

    cell_attr_int, cell_attr_map = _map_cell_attr_strings_to_integers(cell_attr)
    return vertices, cells, cell_attr_int, cell_attr_map
