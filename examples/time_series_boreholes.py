"""Export fixed XYZ measurements without collars, resampling, or inferred well IDs.

Run from an environment with subsurface installed::

    python examples/time_series_boreholes.py \
        /home/leguark/DevOps/SubsurfaceTestData/boreholes/2terranigma output/boreholes \
        --source-timezone Europe/Berlin --single-trajectory

Europe/Berlin is illustrative: confirm the source timezone before exporting.
The affirmation also requires confirming that sequential samples form one line.
No CRS or interpretation of the supplied change columns is inferred.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from subsurface.core.structs.base_structures import UnstructuredData


FILENAME_FORMAT = "%Y-%m-%d_%H-%M-%S_strain_and_temperature.csv"
XYZ_COLUMNS = ["X [m]", "Y [m]", "Z [m]"]
# Source labels retain their Unicode units; exported attribute names are ASCII.
FIELDS = {
    "position [m]": ("position", "m"),
    "brillouin_strain [Ghz]": ("brillouin_strain", "GHz"),
    "raw_strain [\u00b5m/m]": ("raw_strain", "um/m"),
    "strain_change [\u00b5m/m]": ("strain_change", "um/m"),
    "brillouin_temp [Ghz]": ("brillouin_temp", "GHz"),
    "temperature [\u00b0C]": ("temperature", "degC"),
    "temperature_change [\u00b0C]": ("temperature_change", "degC"),
}


def iter_borehole_frames(directory, *, source_timezone, time_series_id,
                         single_trajectory=False):
    """Yield (UTC timestamp, snapshot) pairs with one CSV and reference geometry.

    Missing measurements remain NaN. Geometry and position must match exactly;
    this example deliberately does not interpolate or infer survey geometry.
    """
    if not single_trajectory:
        raise ValueError("Explicit single_trajectory affirmation is required: source has no IDs")
    if not source_timezone:
        raise ValueError("source_timezone is required for naive filename timestamps")
    try:
        zone = ZoneInfo(source_timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Invalid source timezone: {source_timezone}") from exc
    if not isinstance(time_series_id, str) or not time_series_id.strip():
        raise ValueError("time_series_id must be a nonempty string")
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a dataset directory: {directory}")

    files = []
    for path in directory.glob("*.csv"):
        try:
            local = datetime.strptime(path.name, FILENAME_FORMAT)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp filename: {path.name}") from exc
        # Round trips distinguish DST gaps from folds without choosing either fold.
        candidates = set()
        for fold in (0, 1):
            utc = local.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
            if utc.astimezone(zone).replace(tzinfo=None) == local:
                candidates.add(utc)
        if not candidates:
            raise ValueError(f"Nonexistent local timestamp: {path.name} ({source_timezone})")
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous local timestamp: {path.name} ({source_timezone})")
        files.append((candidates.pop(), path))
    if not files:
        raise ValueError(f"No CSV frames found in {directory}")
    files.sort(key=lambda item: item[0])
    if len({timestamp for timestamp, _ in files}) != len(files):
        raise ValueError("Duplicate normalized timestamps")

    reference_xyz = reference_position = cells = None
    expected_columns = set(XYZ_COLUMNS) | set(FIELDS)
    for timestamp, path in files:
        table = pd.read_csv(path)
        if set(table.columns) != expected_columns:
            raise ValueError(f"Inconsistent source fields/units in {path.name}; expected {sorted(expected_columns)}")
        table = table.apply(pd.to_numeric, errors="raise")
        xyz = table[XYZ_COLUMNS].to_numpy(dtype=float)
        position = table["position [m]"].to_numpy(dtype=float)
        if len(position) < 2:
            raise ValueError(f"At least two trajectory samples are required: {path.name}")
        if not np.isfinite(xyz).all() or not np.isfinite(position).all():
            raise ValueError(f"XYZ and position must be finite: {path.name}")
        if not (np.diff(position) > 0).all():
            raise ValueError(f"Position must be strictly increasing and unique: {path.name}")
        if reference_xyz is None:
            reference_xyz, reference_position = xyz.copy(), position.copy()
        elif not (np.array_equal(xyz, reference_xyz)
                  and np.array_equal(position, reference_position)):
            raise ValueError(f"XYZ/position geometry differs from first frame: {path.name}")
        attributes = table[list(FIELDS)].rename(columns={key: value[0] for key, value in FIELDS.items()})
        metadata = {
            "time_series_id": time_series_id,
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "source_filename": path.name,
            "source_timestamp": datetime.strptime(path.name, FILENAME_FORMAT).isoformat(),
            "source_timezone": source_timezone,
            "single_trajectory_confirmed": True,
            "coordinate_units": "m",
            "attribute_units": {name: unit for name, unit in FIELDS.values()},
            "source_fields": {name: source for source, (name, _) in FIELDS.items()},
        }
        frame = UnstructuredData.from_array(
            vertex=xyz, cells="lines" if cells is None else cells,
            vertex_attr=attributes, xarray_attributes=metadata,
        )
        if cells is None:
            cells = frame.cells
        yield metadata["timestamp"], frame


def main(argv=None):
    """Export the explicitly confirmed trajectory through the time-series API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-timezone", required=True, help="Confirmed IANA timezone, e.g. UTC")
    parser.add_argument("--time-series-id", default="2terranigma-strain-temperature")
    parser.add_argument("--single-trajectory", action="store_true", required=True,
                        help="Affirm all ordered samples belong to one connected trajectory")
    args = parser.parse_args(argv)
    from subsurface.modules.writer.time_series import export_time_series

    frames = iter_borehole_frames(
        args.directory, source_timezone=args.source_timezone,
        time_series_id=args.time_series_id, single_trajectory=args.single_trajectory,
    )
    index = export_time_series(frames, args.output, time_series_id=args.time_series_id,
                               kind="trajectory")
    print(index)


if __name__ == "__main__":
    main()
