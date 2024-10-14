import re
import warnings
from typing import Optional, TextIO
import numpy as np
from subsurface.core.structs import UnstructuredData
from ._GOCAD_mesh import GOCADMesh


def mx_to_unstruc_from_binary(stream: TextIO) -> UnstructuredData:
    content = stream.read()
    goCADMeshes = _parse_lines(content)
    unstruct_data = _meshes_to_unstruct(goCADMeshes)
    return unstruct_data


def mx_to_unstruct_from_file(filename: str) -> UnstructuredData:
    # Split the file into meshes
    with open(filename, 'r') as f:
        content: str = f.read()
    goCAD_meshes = _parse_lines(content)
    unstruct_data = _meshes_to_unstruct(goCAD_meshes[0:3:1])

    return unstruct_data


def _parse_lines(content: str) -> list[GOCADMesh]:
    meshes = []
    mesh_blocks = re.split(r'(?=GOCAD TSurf 1)', content)
    # Remove any empty strings from the list

    mesh_blocks = [block for block in mesh_blocks if block.strip()]
    # Use multiprocessing Pool to parse meshes in parallel
    # with Pool(processes=cpu_count()) as pool:
    #     meshes = pool.map(process_mesh, mesh_blocks)
    for mesh_block in mesh_blocks:
        mesh_lines = mesh_block.split('\n')
        mesh = _process_mesh(mesh_lines)
        if mesh:
            meshes.append(mesh)
    return meshes


def _meshes_to_unstruct(meshes: list[GOCADMesh]) -> UnstructuredData:
    # ? I added this function to the Solutions class
    n_meshes = len(meshes)

    vertex_array = np.concatenate([meshes[i].vertices for i in range(n_meshes)])
    simplex_array = np.concatenate([meshes[i].vectorized_edges for i in range(n_meshes)])
    unc, count = np.unique(simplex_array, axis=0, return_counts=True)

    # * Prepare the simplex array
    simplex_array = meshes[0].vectorized_edges
    for i in range(1, n_meshes):
        adder = np.max(meshes[i - 1].vectorized_edges) + 1
        add_mesh = meshes[i].vectorized_edges + adder
        simplex_array = np.append(simplex_array, add_mesh, axis=0)

    # * Prepare the cells_attr array
    ids_array = np.ones(simplex_array.shape[0])
    l0 = 0
    id = 1
    for mesh in meshes:
        l1 = l0 + mesh.vectorized_edges.shape[0]
        ids_array[l0:l1] = id
        l0 = l1
        id += 1

    # * Create the unstructured data
    import pandas as pd
    unstructured_data = UnstructuredData.from_array(
        vertex=vertex_array,
        cells=simplex_array,
        cells_attr=pd.DataFrame(ids_array, columns=['id'])  # TODO: We have to create an array with the shape of simplex array with the id of each simplex
    )

    return unstructured_data


def _process_mesh(mesh_lines) -> Optional[GOCADMesh]:
    mesh = GOCADMesh()
    in_header = False
    in_coord_sys = False
    in_property_class_header = False
    in_tface = False
    in_atom = False
    current_property_class_header = {}
    vertex_list = []
    vertex_indices = []
    triangle_list = []

    for line in mesh_lines:
        line = line.strip()

        if line.startswith('HEADER {'):
            in_header = True
            continue

        if in_header:
            if line == '}':
                in_header = False
            else:
                key_value = line.split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    mesh.header[key.strip()] = value.strip()
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        key, value = parts
                        mesh.header[key.strip()] = value.strip()
                    else:
                        mesh.header[line.strip()] = None
            continue

        if line.startswith('GOCAD_ORIGINAL_COORDINATE_SYSTEM'):
            in_coord_sys = True
            continue
        if in_coord_sys:
            if line == 'END_ORIGINAL_COORDINATE_SYSTEM':
                in_coord_sys = False
            else:
                key_value = line.split(' ', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    mesh.coordinate_system[key.strip()] = value.strip()
                else:
                    mesh.coordinate_system[line.strip()] = None
            continue

        if line.startswith('PROPERTY_CLASS_HEADER'):
            in_property_class_header = True
            current_property_class_header = {'name': line[len('PROPERTY_CLASS_HEADER'):].strip()}
            continue

        if in_property_class_header:
            if line == '}':
                in_property_class_header = False
                mesh.property_class_headers.append(current_property_class_header)
                current_property_class_header = {}
            else:
                key_value = line.split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    current_property_class_header[key.strip()] = value.strip()
                else:
                    key_value = line.split(' ', 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        current_property_class_header[key.strip()] = value.strip()
                    else:
                        current_property_class_header[line.strip()] = None
            continue

        if line == 'TFACE':
            in_tface = True
            continue

        if line.startswith("ATOM"):
            in_atom = True
            continue

        if in_tface:
            if line.startswith('VRTX'):
                # Parse vertex line
                parts = line.split()
                if len(parts) >= 5:
                    _, vid, x, y, z = parts[:5]
                    vertex_indices.append(int(vid))
                    vertex_list.append([float(x), float(y), float(z)])
            elif line.startswith('TRGL'):
                # Parse triangle line
                parts = line.split()
                if len(parts) == 4:
                    _, v1, v2, v3 = parts
                elif len(parts) == 3:
                    v1, v2, v3 = parts
                else:
                    continue
                triangle_list.append([int(v1), int(v2), int(v3)])
            elif line.startswith('BSTONE'):
                _, value = line.split()
                mesh.bstones.append(int(value))
            elif line.startswith('BORDER'):
                parts = line.split()
                if len(parts) >= 4:
                    _, bid, v1, v2 = parts[:4]
                    mesh.borders.append({'id': int(bid), 'v1': int(v1), 'v2': int(v2)})
            elif line == 'END':
                in_tface = False
            else:
                pass
            continue

        # Other lines, possibly store as metadata
        if line:
            key_value = line.split(':', 1)
            if len(key_value) == 2:
                key, value = key_value
                mesh.metadata[key.strip()] = value.strip()
            else:
                pass
        if in_atom:
            warnings.warn("ATOM not implemented yet")
            return None

        # Convert lists to NumPy arrays
    mesh.vertices = np.array(vertex_list)
    mesh.vertex_indices = np.array(vertex_indices)
    mesh.edges = np.array(triangle_list)

    # Check the number of vertices, the indices and the triangles max value to avoid errors

    return mesh
