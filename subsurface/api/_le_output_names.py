"""Resolve split filenames before any outputs are written."""

from collections.abc import Mapping


def split_destinations(directory, keys, output_names, prefix):
    keys = list(keys)
    if output_names is None:
        names = {key: f"{prefix}_{ordinal:06d}.le" for ordinal, key in enumerate(keys)}
    else:
        if not isinstance(output_names, Mapping):
            raise TypeError("output_names must be a mapping")
        if set(output_names) != set(keys):
            raise ValueError("output_names must contain exactly the output object IDs or window labels")
        names = dict(output_names)
        for name in names.values():
            if (not isinstance(name, str) or not name.endswith(".le") or name == ".le"
                    or any(char in name for char in '/\\:')
                    or any(ord(char) < 32 for char in name)):
                raise ValueError("output_names values must be .le filenames without directory components")
        if len(set(names.values())) != len(names):
            raise ValueError("output_names filenames must be unique")
    return {key: directory / names[key] for key in keys}
