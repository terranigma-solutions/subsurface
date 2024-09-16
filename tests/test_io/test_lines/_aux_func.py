from subsurface.modules.visualization import to_pyvista_line, init_plotter, to_pyvista_points, pyvista_to_matplotlib


def _plot(scalar, trajectory, collars=None, lut:int=100, image_2d=True):
    s = to_pyvista_line(
        line_set=trajectory,
        active_scalar=scalar,
        radius=40
    )
    
    
    p = init_plotter()
    import matplotlib.pyplot as plt
    boring_cmap = plt.get_cmap("viridis", lut)
    p.add_mesh(s, cmap=boring_cmap)

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
    else:
        p.show()
