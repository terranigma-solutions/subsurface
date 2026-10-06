"""Export three synthetic 3D temperature frames, with no external data dependency.

    python examples/time_series_volumes.py output/volumes

The named time coordinate contains naive UTC values, so the exporter receives
source_timezone='UTC' explicitly. It slices time before writing ordinary 3D files.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from subsurface.core.structs.base_structures import StructuredData


def synthetic_volume_series():
    """Build a small regular grid with a stable active field and numeric dtype."""
    x = np.linspace(0, 100, 8)
    y = np.linspace(0, 60, 6)
    z = np.linspace(-50, 0, 5)
    times = np.array(["2025-12-12T10:45:39", "2025-12-12T11:45:39",
                      "2025-12-12T12:45:39"], dtype="datetime64[s]")
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    base = 12 + 0.01 * xx + 0.02 * yy - 0.03 * zz
    values = np.stack([base + offset for offset in (0, 0.5, 1)]).astype("float32")
    data = xr.Dataset(
        {"temperature": (("time", "x", "y", "z"), values)},
        coords={"time": times, "x": x, "y": y, "z": z},
        attrs={"attribute_units": {"temperature": "degC"}, "source": "synthetic example"},
    )
    data["temperature"].attrs["units"] = "degC"
    for axis in ("x", "y", "z"):
        data[axis].attrs["units"] = "m"
    return StructuredData(data, "temperature", dtype="float32")


def main(argv=None):
    """Write synthetic volume snapshots and their discovery index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--time-series-id", default="synthetic-temperature")
    args = parser.parse_args(argv)
    from subsurface.modules.writer.time_series import export_volume_time_series

    index = export_volume_time_series(
        synthetic_volume_series(), args.output,
        time_series_id=args.time_series_id, source_timezone="UTC",
    )
    print(index)


if __name__ == "__main__":
    main()
