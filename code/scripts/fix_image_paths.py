"""Remap stale absolute image paths in GeneFlow's cell_image_paths.json to the local copy.

The processed Xenium data ships JSON files whose values point at the original
authors' cluster path (``/depot/natallah/.../processed_data/...``). The actual
``.tif`` files exist locally under ``code/data/processed_data/...``; this script
rewrites every value so it points at the local file and writes a new JSON next to
the original (``*_local.json``).

Usage:
    python scripts/fix_image_paths.py \
        --json data/processed_data/<dataset>/cell_patch_256_aux/input/cell_image_paths.json \
        --local_root data/processed_data

It splits each value at the ``processed_data/`` marker and re-roots the suffix at
``--local_root`` (whose basename must be ``processed_data``). Paths that are
already valid are left untouched. Reports how many remapped paths actually exist.
"""
import os
import json
import argparse


MARKER = "processed_data/"


def remap_path(original: str, local_root_abs: str) -> str:
    """Re-root a single path at ``local_root_abs``.

    ``local_root_abs`` is the absolute path to the local ``processed_data``
    directory. The portion of ``original`` after the last ``processed_data/``
    marker is appended to it.
    """
    if os.path.exists(original):
        return original  # already valid, nothing to do
    idx = original.rfind(MARKER)
    if idx == -1:
        return original  # no marker -> cannot remap, leave as-is
    suffix = original[idx + len(MARKER):]  # e.g. "<dataset>/cell_patch_256_aux/.../x.tif"
    return os.path.join(local_root_abs, suffix)


def main():
    parser = argparse.ArgumentParser(description="Remap stale image paths to local copy.")
    parser.add_argument("--json", required=True, help="Path to cell_image_paths.json (or patch_image_paths.json).")
    parser.add_argument("--local_root", default="data/processed_data",
                        help="Local processed_data directory (its basename must be 'processed_data').")
    parser.add_argument("--out", default=None,
                        help="Output JSON path. Default: <input>_local.json next to the input.")
    args = parser.parse_args()

    local_root_abs = os.path.abspath(args.local_root)
    if os.path.basename(local_root_abs.rstrip("/")) != "processed_data":
        raise ValueError(f"--local_root basename must be 'processed_data', got: {local_root_abs}")

    with open(args.json, "r") as f:
        paths = json.load(f)

    # The suffix already starts with the dataset folder; local_root_abs ends with
    # processed_data, so os.path.join gives processed_data/<dataset>/...  We must
    # therefore strip the trailing 'processed_data' from local_root before joining.
    parent_of_processed = os.path.dirname(local_root_abs)

    remapped = {k: remap_path(v, os.path.join(parent_of_processed, "processed_data")) for k, v in paths.items()}

    out_path = args.out or args.json.replace(".json", "_local.json")
    with open(out_path, "w") as f:
        json.dump(remapped, f)

    n_exist = sum(os.path.exists(v) for v in remapped.values())
    n_total = len(remapped)
    print(f"Wrote {out_path}")
    print(f"Paths existing after remap: {n_exist}/{n_total}")
    if n_exist < n_total:
        # show a couple of misses to aid debugging
        misses = [v for v in remapped.values() if not os.path.exists(v)][:3]
        print("WARNING: some paths still missing, examples:")
        for m in misses:
            print("  ", m)
    return out_path


if __name__ == "__main__":
    main()
