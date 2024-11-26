from dataclasses import dataclass, field

import numpy as np


@dataclass
class GOCADMesh:
    header: dict = field(default_factory=dict)
    coordinate_system: dict = field(default_factory=dict)
    property_class_headers: list = field(default_factory=list)
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    vertex_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    bstones: list = field(default_factory=list)
    borders: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def vectorized_edges(self):
        # Create index mapping from original to zero-based indices
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.vertex_indices)}
        # Map triangle indices
        try:
            triangles_mapped = np.vectorize(idx_map.get)(self.edges)
        except TypeError as e:
            self._verbose_debugging()
            raise f"Error mapping indices for mesh: {e}"

        return triangles_mapped

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

    def _verbose_debugging(self):
        # Create index mapping from original to zero-based indices
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(self.vertex_indices)}

        # Check for missing indices
        unique_edge_indices = np.unique(self.edges)
        missing_indices = set(unique_edge_indices) - set(idx_map.keys())
        if missing_indices:
            raise ValueError(f"Edges contain indices not found in vertex_indices: {missing_indices}")

        # Map triangle indices using a list comprehension
        try:
            edges_flat = self.edges.flatten()
            mapped_flat = [idx_map[idx] for idx in edges_flat]
            triangles_mapped = np.array(mapped_flat).reshape(self.edges.shape)
        except Exception as e:
            raise Exception(f"Error mapping indices for mesh: {e}")

        return triangles_mapped
