from subsurface.modules.visualization import to_pyvista_line, init_plotter, to_pyvista_points


def _plot(scalar, trajectory, collars=None):
    s = to_pyvista_line(
        line_set=trajectory,
        active_scalar=scalar,
        radius=40
    )
    p = init_plotter()
    import matplotlib.pyplot as plt
    boring_cmap = plt.get_cmap("viridis", 8)
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
    p.show()
