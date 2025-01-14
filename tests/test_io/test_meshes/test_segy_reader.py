import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytest

from subsurface import TriSurf, optional_requirements
from subsurface.core.structs.base_structures import StructuredData, UnstructuredData
from subsurface.modules.reader.volume import segy_reader
from subsurface.modules.visualization import to_pyvista_mesh, pv_plot, to_pyvista_grid
from tests.conftest import RequirementsLevel

pytestmark = pytest.mark.skipif(
    condition=(RequirementsLevel.READ_VOLUME) not in RequirementsLevel.REQUIREMENT_LEVEL_TO_TEST(),
    reason="Need to set READ_VOLUME"
)

pv = optional_requirements.require_pyvista()


def _read_segy_file(file_path) -> dict:
    with open(file_path, 'r') as cf:
        lines = cf.readlines()
        xs = []
        ys = []
        for line in lines:
            x = line.split()[1]
            y = line.split()[2]
            xs.append(x)
            ys.append(y)
    return {'x': xs, 'y': ys}


input_path = os.path.dirname(__file__) + '/../../data/segy'
files = ['/E5_MIG_DMO_FINAL.sgy', '/E5_MIG_DMO_FINAL_DEPTH.sgy', '/E5_STACK_DMO_FINAL.sgy', '/test.segy', '/Linie01.segy']
images = ['/myplot_cropped.png', '/myplot2_cropped.png', '/myplot3_cropped.png', '/myplot4_cropped.png']
coords = _read_segy_file(input_path + '/E5_CMP_COORDS.txt')


@pytest.fixture(scope="module")
def get_structured_data() -> List[StructuredData]:
    file_array = [input_path + x for x in files]
    sd_array = [segy_reader.read_in_segy(fp) for fp in file_array]
    return sd_array


@pytest.fixture(scope="module")
def get_images() -> List[str]:
    image_array = [input_path + y for y in images]
    return image_array


def test_converted_to_structured_data(get_structured_data):
    for x in get_structured_data:
        assert isinstance(x, StructuredData)
        x.active_data_array.plot()
        plt.show(block=False)


def test_pyvista_grid(get_structured_data, get_images):
    for s, t in zip(get_structured_data, get_images):

        x = s.data['x']
        y = s.data['y']

        x2, y2 = np.meshgrid(x, y)
        print(x2, y2)
        tex = pv.read_texture(t)
        z = np.zeros((len(y), len(x)))
        # z.reshape(z, (-1, 1101))
        print(x2.shape, y2.shape, z.shape)

        # create a surface to host this texture
        surf = pv.StructuredGrid(z, x2, y2)
        print(surf)

        surf.texture_map_to_plane(inplace=True)
        if False:  # Taking screenshots of pyvista is not handle well by pycharm
            pv_plot([surf], image_2d=True)
            time.sleep(2)
        # use Trisurf with Structured Data for texture and UnstructuredData for geometry


def test_read_segy_to_struct_data_imageio(get_structured_data, get_images):
    imageio = optional_requirements.require_imageio()
    for x, image in zip(get_structured_data, get_images):
        vertex = np.array([[0, x.data['x'][0], x.data['y'][0]], [0, x.data['x'][-1], x.data['y'][0]], [0, x.data['x'][0], x.data['y'][-1]], [0, x.data['x'][-1], x.data['y'][-1]]])
        a = pv.PolyData(vertex)
        b = a.delaunay_2d().faces
        cells = b.reshape(-1, 4)[:, 1:]
        print('cells', cells)
        struct = StructuredData.from_numpy(np.array(imageio.imread(image)))
        unstruct = UnstructuredData.from_array(vertex, cells)
        ts = TriSurf(
            mesh=unstruct,
            texture=struct
        )

        s = to_pyvista_mesh(ts)

        if False:  # Taking screenshots of pyvista is not handle well by pycharm
            pv_plot([s], image_2d=True)
            time.sleep(2)


def test_plot_segy_as_struct_data_with_coords_dict(get_structured_data, get_images):
    imageio = optional_requirements.require_imageio()
    s = []
    i = 0
    for x, image in zip(get_structured_data, get_images):
        zmin = -6000.0
        zmax = 0.0
        v, e = segy_reader.create_mesh_from_coords(coords, zmin, zmax)
        v[:, 0] = + i * 1000

        struct = StructuredData.from_numpy(np.array(imageio.imread(image)))
        print(struct)  # normalize to number of samples
        unstruct = UnstructuredData.from_array(v, e)

        origin = [float(coords['x'][0]), float(coords['y'][0]), zmin]
        point_u = [float(coords['x'][-1]), float(coords['y'][-1]), zmin]
        point_v = [float(coords['x'][0]), float(coords['y'][0]), zmax]
        ts = TriSurf(
            mesh=unstruct,
            texture=struct,
            texture_origin=origin,
            texture_point_u=point_u,
            texture_point_v=point_v
        )

        s.append(to_pyvista_mesh(ts))
        i += 1

    if True:  # Taking screenshots of pyvista is not handle well by pycharm
        pv_plot(meshes=s, image_2d=False)
        time.sleep(2)




@pytest.mark.skip(reason="This test should only being run explicitly")
def test_segy_3d_segy_seg() -> None:
    import xarray as xr
    from segysak.segy import segy_header_scan

    # default just needs the file name
    scan = segy_header_scan(os.getenv("PATH_TO_SEISMIC_3D"))
    scan

    V3D = xr.open_dataset(
        filename_or_obj=os.getenv("PATH_TO_SEISMIC_3D"),
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 73, "cdp_y": 77},
    )
    fig, ax1 = plt.subplots(ncols=1, figsize=(15, 8))
    iline_sel = 10093
    V3D.data.transpose(
        "samples", "iline", "xline",
        transpose_coords=True
    ).sel(
        iline=iline_sel,
        method="nearest"
    ).plot(yincrease=False, cmap="seismic_r")
    plt.grid("grey")
    plt.ylabel("TWT")
    plt.xlabel("XLINE")
    plt.show()

    V3D.to_netcdf("data/V3D.nc")
    pass


def test_segy_3d_segy__volume_segsak_II() -> None:
    import os
    import xarray as xr
    from segysak.segy import segy_header_scan

    # Scan the SEG-Y file headers
    scan = segy_header_scan(os.getenv("PATH_TO_SEISMIC_3D"))
    print(scan)

    # Read the SEG-Y into an xarray Dataset
    # Adjust these byte positions (dim_byte_fields, extra_byte_fields) to match your file
    V3D = xr.open_dataset(
        filename_or_obj=os.getenv("PATH_TO_SEISMIC_3D"),
        dim_byte_fields={"iline": 189, "xline": 193},  # might differ for your data
        extra_byte_fields={"cdp_x": 73, "cdp_y": 77},  # might differ for your data
    )

    # Quick sanity check: plot one inline
    import matplotlib.pyplot as plt

    iline_sel = 10093
    # Note: "data" may be named differently in your xarray
    V3D.data.transpose("samples", "iline", "xline", transpose_coords=True).sel(
        iline=iline_sel,
        method="nearest"
    ).plot(yincrease=False, cmap="seismic_r"
           )
    plt.grid("grey")
    plt.ylabel("TWT (samples)")
    plt.xlabel("XLINE")
    plt.title(f"Inline = {iline_sel}")
    plt.show()

    # Optionally save as NetCDF
    # V3D.to_netcdf("data/V3D.nc")

    # Step II

    # Ensure the order is (samples, iline, xline)
    # Some xarray objects might already be in that order
    data_da = V3D.data.transpose("samples", "iline", "xline")

    # Convert to a NumPy array
    data_3d = data_da.values  # shape: (nz, ny, nx)
    nz, ny, nx = data_3d.shape
    print("3D Data Shape:", data_3d.shape)

    import pyvista as pv

    # Spacing in each dimension (index spacing = 1 by default, or define actual spacing if known)
    dx = 1.0  # spacing along xline
    dy = 1.0  # spacing along iline
    dz = 1.0  # spacing along samples (e.g., 2 ms, or convert to depth if you have a velocity model)

    # Origin (0,0,0) or shift as needed
    origin = (0, 0, 0)

    # Create the UniformGrid
    grid = pv.UniformGrid()
    grid.origin = origin  # bottom-left of the dataset
    grid.spacing = (dx, dy, dz)  # distance between points along each axis
    grid.dimensions = (nx, ny, nz)  # note the order: (x, y, z)
    # Flatten the data in x-fastest order (Fortran order) if needed
    grid.point_data["Amplitude"] = data_3d.ravel(order="F")

    # Volume Rendering
    plotter = pv.Plotter()
    plotter.add_volume(grid, cmap="seismic", opacity="sigmoid")
    plotter.show_grid()
    plotter.show()


def test_segy_3d_segy_segsak_III() -> None:
    import os
    import xarray as xr
    from segysak.segy import segy_header_scan
    import matplotlib.pyplot as plt

    # 1. (Optional) Scan headers to find byte fields
    scan_info = segy_header_scan(os.getenv("PATH_TO_SEISMIC_3D"))
    print(scan_info)

    # 2. Open SEG-Y as an xarray Dataset
    V3D = xr.open_dataset(
        filename_or_obj=os.getenv("PATH_TO_SEISMIC_3D"),
        dim_byte_fields={"iline": 189, "xline": 193},  # may differ for your file
        extra_byte_fields={"cdp_x": 73, "cdp_y": 77},  # may differ for your file
    )
    # 2) Coarsen that region by a factor of 2 in each dimension
    coarsen_factor = 10
    V3D_coarse = V3D.coarsen(
        samples=coarsen_factor,
        iline=coarsen_factor,
        xline=coarsen_factor,
        boundary="trim"
    ).mean()

    V3D.close()
    V3D = V3D_coarse

    # 3. Quick check: Plot a single inline
    inline_example = V3D.iline.values[0]  # pick first or any inline
    V3D.data.transpose("samples", "iline", "xline").sel(
        iline=inline_example,
        method="nearest"
    ).plot(
        yincrease=False, cmap="seismic_r"
    )
    plt.title(f"Inline = {inline_example}")
    plt.show()

    import numpy as np

    # Extract data in a consistent order: (samples, iline, xline)
    data_da = V3D.data.transpose("samples", "iline", "xline")
    data_3d = data_da.values  # shape: (nz, ny, nx)

    nz, ny, nx = data_3d.shape
    print("Seismic cube shape:", data_3d.shape)

    # Extract the sample values (e.g., time in ms or sample indices)
    samples_1d = V3D.samples.values  # shape (nz,)

    # Extract cdp_x, cdp_y (2D arrays for each (iline, xline))
    cdp_x_2d = V3D.cdp_x.values  # shape: (ny, nx)
    cdp_y_2d = V3D.cdp_y.values  # shape: (ny, nx)
    struct = StructuredData.from_numpy(
        array=data_3d,  # data_3d is shape (nz, ny, nx)
        coords={
                'z': np.linspace(
                    start=samples_1d.min(),
                    stop=samples_1d.max(),
                    num=nz
                ),
                'y': np.linspace(
                    start=cdp_y_2d.min(),
                    stop=cdp_y_2d.max(),
                    num=ny
                ),
                'x': np.linspace(
                    start=cdp_x_2d.min(),
                    stop=cdp_x_2d.max(),
                    num=nx
                ),
        },
        dim_names=["z", "y", "x"]  # IMPORTANT: match the shape (nz, ny, nx)
    )

    struct.to_netcdf("data/3D_seismic.nc")
    pass


def test_segy_3d_segy_viz() -> None:
    import subsurface
    struct = StructuredData.from_netcdf("data/3D_seismic.nc")
    sg: subsurface.StructuredGrid = subsurface.StructuredGrid(struct)
    grid = to_pyvista_grid(sg)

    plotter = pv.Plotter()

    plotter.add_volume(
        grid,
        cmap="seismic",
        opacity="sigmoid_20",  # or "linear", "exp", etc.
        shade=False,         # sometimes turning shading off is clearer for seismic
        # Add min max to 50 -50
        clim=(-50, 50)
    )
    
    p = plotter

    if image_2d := True is False:
        p.show()
        return p
    else:
        from subsurface.modules.visualization import to_pyvista
        fig = to_pyvista.pyvista_to_matplotlib(p)
        return fig
