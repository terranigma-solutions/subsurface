import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyvista as pv


@dataclass
class Mesh:
    header: dict = field(default_factory=dict)
    coordinate_system: dict = field(default_factory=dict)
    property_class_headers: list = field(default_factory=list)
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    vertex_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    bstones: list = field(default_factory=list)
    borders: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


    @property
    def color(self):
        """Try to get the color from the metadata. Can be in the form of:
        *solid*color: #87ceeb or *solid*color: 0 0 1 1 (rgba).
        Returns a color in the format acceptable by PyVista.
        """
        color = None

        # First try to find a color value from the header
        for key, value in self.header.items():
            if 'color' in key.lower():
                color = value.strip()
                break

        # If no color was found, return None
        if not color:
            return None

        # Handle hexadecimal color string (e.g., #87ceeb)
        if color.startswith('#') and len(color) == 7:
            return color  # already valid as a hex string

        # Handle space-separated RGBA or RGB values (e.g., "0 0 1 1")
        if ' ' in color:
            color_vals = [float(c) for c in color.split()]
            if len(color_vals) == 4:  # RGBA
                return color_vals[:3]  # ignore the alpha channel for now, as PyVista handles RGB
            elif len(color_vals) == 3:  # RGB
                return color_vals  # already in the right format

        # Fallback: if none of the formats match, return None
        return None


def parse_gocad_mx_file(filename):
    meshes = []
    current_mesh = None
    in_header = False
    in_coord_sys = False
    in_property_class_header = False
    in_tface = False

    # Split the file into meshes
    with open(filename, 'r') as f:
        content = f.read()

    mesh_blocks = re.split(r'(?=GOCAD TSurf 1)', content)

    # Remove any empty strings from the list
    mesh_blocks = [block for block in mesh_blocks if block.strip()]

    # Use multiprocessing Pool to parse meshes in parallel
    # with Pool(processes=cpu_count()) as pool:
    #     meshes = pool.map(process_mesh, mesh_blocks)

    for mesh_block in mesh_blocks:
        mesh_lines = mesh_block.split('\n')
        mesh = process_mesh(mesh_lines)
        if mesh:
            meshes.append(mesh)

    return meshes


def process_mesh(mesh_lines) -> Optional[Mesh]:
    mesh = Mesh()
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
    mesh.triangles = np.array(triangle_list)

    # Check the number of vertices, the indices and the triangles max value to avoid errors

    return mesh


def meshes_to_pyvista(meshes):
    pyvista_meshes = []
    for mesh in meshes:
        # Create index mapping from original to zero-based indices
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(mesh.vertex_indices)}

        # Map triangle indices
        try:
            triangles_mapped = np.vectorize(idx_map.get)(mesh.triangles)
        except TypeError as e:
            print(f"Error mapping indices for mesh: {e}")
            continue

        # Create faces array for PyVista
        faces = np.hstack([np.full((triangles_mapped.shape[0], 1), 3), triangles_mapped]).flatten()

        # Create PyVista mesh
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        pv_mesh.color = mesh.color
        pyvista_meshes.append(pv_mesh)
    return pyvista_meshes
