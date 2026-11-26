"""Export fixed-geometry observations as ordinary Liquid Earth snapshots."""

import hashlib
import json
import re
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from subsurface.core.structs.base_structures import StructuredData, UnstructuredData

INDEX_NAME = "series.json"
MASTER_SCHEMA_VERSION = 2


def _timestamp(value):
    if not isinstance(value, (datetime, str)):
        raise TypeError("timestamp must be a timezone-aware datetime or ISO string")
    if isinstance(value, str):
        # Timestamp also accepts non-ISO date strings; do not accept those here.
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    stamp = stamp.tz_convert("UTC")
    return stamp, stamp.isoformat().replace("+00:00", "Z")


def _array_schema(array, fixed=False):
    schema = [array.dims, array.shape, str(array.dtype), array.attrs]
    if fixed:
        values = array.values
        if values.dtype.hasobject:
            content = json.dumps(values.tolist(), sort_keys=True).encode("utf-8")
        else:
            content = np.ascontiguousarray(values).tobytes()
        schema.append(hashlib.sha256(content).hexdigest())
    return schema


def _schema(data, format_version=1):
    ds = data.data
    if "time" in ds.dims:
        raise ValueError("snapshots must not contain a time dimension")
    schema = {"coords": {name: _array_schema(coord, True)
                         for name, coord in ds.coords.items()},
              "units": ds.attrs.get("attribute_units", ds.attrs.get("units"))}
    if isinstance(data, StructuredData):
        array = data.active_data_array
        # Format 2 stores every variable, so vectors may add a dimension after x, y, z.
        spatial = array.dims[:3] if format_version == 2 else array.dims
        if len(spatial) != 3 or any(size == 0 for size in array.shape):
            raise ValueError("volume snapshots must have three nonempty spatial dimensions")
        if spatial != ("x", "y", "z"):
            raise ValueError("volume spatial dimensions must be ordered x, y, z")
        for coord in array.coords.values():
            if np.issubdtype(coord.dtype, np.number) and not np.isfinite(coord.values).all():
                raise ValueError("spatial coordinates must be finite")
        schema.update(field=data.active_data_array_name, array=_array_schema(array),
                      bounds=data.bounds, type=data.type.name, dtype=data.dtype)
        if format_version == 2:
            if data._bounds is not None:
                raise ValueError("format 2 frames take their bounds from the spatial coordinates; "
                                 "drop the bounds override")
            schema["variables"] = {name: _array_schema(variable) for name, variable in ds.data_vars.items()}
    else:
        vertex, cells = data.vertex, data.cells
        if vertex.ndim != 2 or vertex.shape[1] != 3 or not len(vertex) or not np.isfinite(vertex).all():
            raise ValueError("trajectory XYZ must be nonempty and finite")
        if (cells.ndim != 2 or cells.shape[1] != 2 or
                not np.issubdtype(cells.dtype, np.integer) or
                np.any(cells < 0) or np.any(cells >= len(vertex))):
            raise ValueError("trajectory cells must contain valid line connectivity")
        schema["variables"] = {name: _array_schema(array, name in ("vertex", "cells"))
                               for name, array in ds.data_vars.items()}
        schema["well_id_mapper"] = ds.attrs.get("well_id_mapper")
        for name, attrs in (("cell", data.attributes), ("vertex", data.points_attributes)):
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in attrs.dtypes):
                raise ValueError("trajectory sample attributes must be numeric")
            schema[name + "_attributes"] = [(str(column), str(attrs[column].dtype))
                                               for column in attrs.columns]
            for column in ("position", "well_id", "measured_depths", "is_attr_point"):
                if column in attrs:
                    values = attrs[column].to_numpy()
                    if column == "position" and not np.isfinite(values).all():
                        raise ValueError("sample position must be finite")
                    schema[name + "_" + column] = hashlib.sha256(values.tobytes()).hexdigest()
    # Canonical JSON also checks metadata before writing any snapshot bytes.
    return json.dumps(schema, sort_keys=True, allow_nan=False)


def export_time_series(frames, output_directory, *, time_series_id, kind, format_version=1):
    """Write an iterable of ``(aware datetime or ISO timestamp, data)`` pairs.

    ``kind`` is ``volume`` for StructuredData or ``trajectory`` for
    UnstructuredData line meshes. Geometry, sample order, and attribute schemas
    must remain fixed, including borehole identity attributes and well ID mapping.
    Volume spatial dimensions must be ordered x, y, z. Measurement values
    (including NaNs) may change. Input objects are not modified. The output
    directory must not already exist.

    ``format_version=2`` writes volume frames in structured .le format 2, with
    every data variable (see ``docs/le_spec.md``). Every variable must then stay
    the same across frames, and bounds come from the spatial coordinates.

    Frames are consumed once, with only frame metadata retained. The index is
    sorted chronologically, published last, and returned as a pathlib.Path.
    Failed exports remove their newly created directory, never an existing one.
    """
    if not isinstance(time_series_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", time_series_id):
        raise ValueError("time_series_id must be a nonempty stable ID using letters, digits, _, . or -")
    if kind not in ("volume", "trajectory"):
        raise ValueError("kind must be volume or trajectory")
    if format_version not in (1, 2) or type(format_version) is not int:
        raise ValueError("format_version must be 1 or 2")
    if format_version == 2 and kind != "volume":
        raise ValueError("format 2 is a structured .le format; only volume frames use it")
    expected_type = StructuredData if kind == "volume" else UnstructuredData
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError("output directory already exists: {}".format(directory))
    entries, seen = [], set()
    baseline = None
    created = False
    try:
        for timestamp, data in frames:
            stamp, text = _timestamp(timestamp)
            if stamp in seen:
                raise ValueError("duplicate timestamp: {}".format(text))
            if not isinstance(data, expected_type):
                raise TypeError("{} frames must be {}".format(kind, expected_type.__name__))
            schema = _schema(data, format_version)
            if baseline is not None and schema != baseline:
                raise ValueError("frames must share fixed geometry, sample identity and attribute schema")
            baseline = schema
            ds = data.data.copy(deep=False)
            ds.attrs = dict(data.data.attrs, time_series_id=time_series_id, timestamp=text)
            json.dumps(ds.attrs, allow_nan=False)
            snapshot = replace(data, data=ds)
            if not created:
                directory.mkdir(parents=True, exist_ok=False)
                created = True
            name = "{}_{:04d}.le".format(kind, len(entries))
            with (directory / name).open("xb") as stream:
                stream.write(snapshot.to_binary(format_version=format_version) if format_version == 2
                             else snapshot.to_binary())
            entries.append((stamp, {"timestamp": text, "path": name}))
            seen.add(stamp)
        if not entries:
            raise ValueError("a time series must contain at least one frame")
        index = {"schema_version": 1, "time_series_id": time_series_id,
                 "kind": kind, "frames": [entry for _, entry in sorted(entries, key=lambda item: item[0])]}
        temporary = directory / (INDEX_NAME + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(index, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.rename(directory / INDEX_NAME)
    except BaseException:
        if created:
            shutil.rmtree(directory)
        raise
    return directory / INDEX_NAME


def time_series_master(latest, index, *, directory, revision, format_version=1):
    """The master file of a time series: the latest frame, listing every frame for a client to pull.

    ``latest`` is the data of the chronologically last frame of ``index`` (the
    dict that :func:`export_time_series` writes as ``series.json``).
    ``directory`` is where the frames are published, relative to the container,
    and ``revision`` identifies that publication. The manifest goes in the
    master's ``xarray_attrs.time_series`` (schema 2); ``time_series_id`` and
    ``timestamp`` stay at the top level, as in the frames. Returns the .le bytes.
    """
    kind, frames = index["kind"], index["frames"]
    expected_type = StructuredData if kind == "volume" else UnstructuredData
    if not isinstance(latest, expected_type):
        raise TypeError("{} masters must be {}".format(kind, expected_type.__name__))
    if not frames:
        raise ValueError("a time series must contain at least one frame")
    if format_version == 2 and kind != "volume":
        raise ValueError("format 2 is a structured .le format; only volume masters use it")
    time_series_id = index["time_series_id"]
    ds = latest.data.copy(deep=False)
    ds.attrs = dict(ds.attrs, time_series_id=time_series_id, timestamp=frames[-1]["timestamp"], time_series={
            "schema_version": MASTER_SCHEMA_VERSION,
            "role"          : "master",
            "time_series_id": time_series_id,
            "revision"      : revision,
            "kind"          : kind,
            "directory"     : directory,
            "index"         : INDEX_NAME,
            "frames"        : frames,
    })
    if kind == "trajectory":
        ds.attrs["well_id_mapper"] = {time_series_id: 0}  # one well, named after the series
    json.dumps(ds.attrs, allow_nan=False)
    master = replace(latest, data=ds)
    return master.to_binary(format_version=format_version) if format_version == 2 else master.to_binary()


def export_volume_time_series(data, output_directory, *, time_series_id, source_timezone=None, format_version=1):
    """Slice a StructuredData time axis and export ordinary 3D volume frames.

    Naive time coordinates require an explicit source_timezone (timezone name
    or tzinfo). Ambiguous/nonexistent local times raise rather than guessing.
    Active field, type, output dtype, metadata and explicit bounds are retained.
    The time axis may appear anywhere, but spatial dimensions must be x, y, z
    in that order to match the static payload layout.
    """
    if not isinstance(data, StructuredData):
        raise TypeError("data must be StructuredData")
    if "time" not in data.active_data_array.dims or "time" not in data.data.coords:
        raise ValueError("volume data must have a time dimension and coordinate")

    def frames():
        times = data.data.indexes["time"]
        for i, value in enumerate(times):
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("time coordinate contains NaT")
            if stamp.tzinfo is None:
                if source_timezone is None:
                    raise ValueError("naive time coordinates require source_timezone")
                stamp = stamp.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
            snapshot = replace(data, data=data.data.isel(time=i, drop=True))
            yield stamp, snapshot

    return export_time_series(frames(), output_directory, time_series_id=time_series_id, kind="volume",
                              format_version=format_version)
