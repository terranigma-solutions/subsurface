import numpy as np

from subsurface.modules.visualization import to_pyvista_line, init_plotter, to_pyvista_points, pyvista_to_matplotlib


def _plot(scalar, trajectory, collars=None, lut:int=100, image_2d=True):
    s = to_pyvista_line(
        line_set=trajectory,
        active_scalar=scalar,
        radius=10
    )

    s = clip_nan_points(s, scalar_name=scalar)
    p = init_plotter(image_2d=image_2d)
    import matplotlib.pyplot as plt
    boring_cmap = plt.get_cmap("viridis", lut)
    p.add_mesh(s, cmap=boring_cmap)
    p.add_axes()
    # Clip nans

    if collars is not None:
        collar_mesh = to_pyvista_points(collars.collar_loc)
        p.add_mesh(collar_mesh, render_points_as_spheres=True)
        p.add_point_labels(
            points=collars.collar_loc.points,
            labels=collars.ids,
            point_size=10,
            shape_opacity=0.5,
            font_size=12,
            bold=True
        )
    if image_2d:
        f = pyvista_to_matplotlib(p)
        p.close()
    else:
        p.show()

def clip_nan_points(mesh, scalar_name):
    import pyvista as pv
    # Extract the scalar array
    scalars = mesh.point_data[scalar_name]
    
    # Identify valid (non-NaN) points
    valid_mask = ~np.isnan(scalars)
    
    # Apply the mask to filter points and scalars
    filtered_points = mesh.points[valid_mask]
    filtered_scalars = scalars[valid_mask]
    
    # Create a new PyVista mesh with the filtered data
    filtered_mesh = pv.PolyData(filtered_points)
    filtered_mesh.point_data[scalar_name] = filtered_scalars
    
    return filtered_mesh

# Example usage