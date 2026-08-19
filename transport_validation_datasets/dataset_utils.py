"""Assemble per-episode datasets into one tensorized Zarr store, one episode at a time.

Based on popsim/data/dataset_utils.py
"""

import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr
from loguru import logger


def build_tensorized_dataset(
    process_fn: Callable[[Any], xr.Dataset | None],
    identifiers: Iterable[Any],
    zarr_path: Path | str,
    time_dim: str,
    episode_dim: str,
    extend_existing: bool = False,
    episodes_per_chunk: int | None = None,
    mb_per_chunk: int | None = 10,
    dim_sizes: dict[str, int] | None = None,
) -> xr.Dataset:
    """Build a tensorized multi-episode dataset from one-episode datasets.

    process_fn is called on one identifier at a time and returns that episode's
    dataset (or None to skip it). Each returned episode is appended to the Zarr
    store immediately, so peak memory is one episode, not the whole dataset.
    An episode that raises is logged and skipped, not fatal: a single unreadable
    shot does not lose the rest of the run.

    Args:
        process_fn: Turns one identifier into that episode's dataset, or None
            to skip the episode.
        identifiers: Episode identifiers, passed to process_fn one by one.
        zarr_path: Path of the Zarr store to write, must end in .zarr.
        time_dim: Name of the per-episode time dimension.
        episode_dim: Name of the dimension episodes are stacked along.
        extend_existing: Append to the store at zarr_path if it already exists,
            instead of refusing to touch it.
        episodes_per_chunk: Episodes per storage chunk. Mutually exclusive with
            mb_per_chunk. None with mb_per_chunk None leaves the chunking alone.
        mb_per_chunk: Target chunk size, used to pick episodes_per_chunk from
            the mean episode size. Mutually exclusive with episodes_per_chunk.
        dim_sizes: Upper bound per non-episode dimension. Every episode is
            padded to these sizes before being written, so the store never has
            to be extended (which rewrites the chunks of every episode already
            written). Slack is trimmed during the rechunking pass.

    Returns:
        The assembled dataset, opened from the finished store.

    Raises:
        ValueError: If zarr_path does not end in .zarr, if it exists and
            extend_existing is False, or if both chunking options were given.
    """
    zarr_path = Path(zarr_path)
    if zarr_path.suffix != ".zarr":
        raise ValueError(f"zarr_path must end with .zarr, got {zarr_path}")
    if zarr_path.exists() and not extend_existing:
        raise ValueError(
            f"Zarr store at {zarr_path} already exists and extend_existing is False. "
            "Remove it or pass extend_existing=True."
        )
    if mb_per_chunk is not None and episodes_per_chunk is not None:
        raise ValueError("Pass either mb_per_chunk or episodes_per_chunk, not both.")

    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    # Size of time_dim in the store, tracked here so every episode does not
    # have to reopen the store to find it.
    store_time_dim_size = None
    for identifier in identifiers:
        try:
            ds = process_fn(identifier)
        except Exception as exception:
            logger.warning(f"Error processing {identifier}: {exception}")
            continue
        if ds is None:
            continue

        add_to_zarr_store(
            ds,
            zarr_path,
            time_dim,
            episode_dim,
            store_time_dim_size=store_time_dim_size,
            dim_sizes=dim_sizes,
        )
        if store_time_dim_size is None:
            store_time_dim_size = xr.open_zarr(zarr_path, consolidated=True).sizes[
                time_dim
            ]
        else:
            store_time_dim_size = max(store_time_dim_size, ds.sizes[time_dim])
        n_written += 1
        del ds

    if n_written == 0:
        logger.warning(f"No episodes were written to {zarr_path}")
        if extend_existing and zarr_path.exists():
            return xr.open_zarr(zarr_path, consolidated=None)
        return xr.Dataset()

    logger.info(f"Wrote {n_written} episodes to {zarr_path}, now chunking it")
    ds = xr.open_zarr(zarr_path, consolidated=True)

    # Drop the padding slack left by dim_sizes. Only worth doing when a
    # rechunking pass is going to rewrite the store anyway.
    if dim_sizes is not None and (
        mb_per_chunk is not None or episodes_per_chunk is not None
    ):
        for dim in dim_sizes:
            if dim in ds.dims and dim != episode_dim:
                ds = trim_trailing_nan_slices(ds, dim)

    if mb_per_chunk is not None:
        n_episodes = ds.sizes[episode_dim]
        mb_per_episode = ds.isel({episode_dim: 0}).nbytes / (1024 * 1024)
        episodes_per_chunk = min(max(1, int(mb_per_chunk / mb_per_episode)), n_episodes)
        logger.info(
            f"Chunking {episodes_per_chunk} episodes per chunk "
            f"(mb_per_chunk={mb_per_chunk}, {mb_per_episode:.2f} MB per episode)"
        )

    if episodes_per_chunk is not None:
        # Chunk across episodes only, every other dimension stays in one chunk.
        chunk_spec = {episode_dim: episodes_per_chunk} | {
            dim: ds.sizes[dim] for dim in ds.dims if dim != episode_dim
        }
        ds = zarr_chunk(ds, chunk_spec)

        # Write the rechunked store beside the old one, then swap, so a crash
        # mid-rewrite leaves the original store intact.
        tmp_path = zarr_path.with_suffix(".zarr.tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        ds.to_zarr(tmp_path, mode="w", consolidated=True)
        ds.close()
        shutil.rmtree(zarr_path)
        os.rename(tmp_path, zarr_path)

    zarr.consolidate_metadata(zarr_path)
    return xr.open_zarr(zarr_path, consolidated=True)


def trim_trailing_nan_slices(ds: xr.Dataset, dim: str) -> xr.Dataset:
    """Drop trailing slices along dim that are NaN in every variable.

    Removes the slack left over when episodes were padded up to an upper bound.

    Args:
        ds: Dataset to trim.
        dim: Dimension to trim along.

    Returns:
        The trimmed dataset, unchanged if there was nothing to drop.
    """
    slice_has_data = None
    for var in ds.data_vars:
        if dim not in ds[var].dims:
            continue
        other_dims = [d for d in ds[var].dims if d != dim]
        var_has_data = ds[var].notnull().any(other_dims)
        slice_has_data = (
            var_has_data if slice_has_data is None else (slice_has_data | var_has_data)
        )

    if slice_has_data is None:
        return ds

    valid = slice_has_data.compute().values
    if not valid.any():
        return ds

    last_valid = int(np.nonzero(valid)[0][-1]) + 1
    if last_valid < ds.sizes[dim]:
        logger.info(
            f"Trimming {ds.sizes[dim] - last_valid} trailing all-NaN slices along {dim}"
        )
        ds = ds.isel({dim: slice(0, last_valid)})
    return ds


def extend_zarr_along_dim(zarr_path: Path | str, dim: str, n_extend: int):
    """Extend an existing Zarr store along one dimension, NaN padding the new slots.

    Args:
        zarr_path: Path of the Zarr store.
        dim: Dimension to extend.
        n_extend: Number of slots to add.
    """
    ds = xr.open_zarr(zarr_path, consolidated=True)
    ds = ds.pad({dim: (0, n_extend)})
    ds_padding = ds.isel({dim: slice(-n_extend, None)})
    # Pulled into memory to keep dask out of the append path
    ds_padding = ds_padding.compute()
    ds_padding.to_zarr(
        zarr_path, mode="a-", append_dim=dim, consolidated=True, align_chunks=True
    )


def add_to_zarr_store(
    ds: xr.Dataset,
    zarr_path: Path | str,
    time_dim: str,
    episode_dim: str,
    store_time_dim_size: int | None = None,
    dim_sizes: dict[str, int] | None = None,
):
    """Append one episode to a Zarr store, creating the store if it is not there.

    The episode is NaN padded (or the store extended) so that every non-episode
    dimension lines up with what is already in the store.

    Args:
        ds: One episode's dataset.
        zarr_path: Path of the Zarr store.
        time_dim: Name of the per-episode time dimension.
        episode_dim: Name of the dimension episodes are stacked along.
        store_time_dim_size: Size of time_dim in the store, if the caller is
            already tracking it. None reads it from the store instead.
        dim_sizes: Upper bound per non-episode dimension, see
            build_tensorized_dataset.

    Raises:
        ValueError: If ds holds more than one episode, exceeds a dim_sizes
            bound, carries a dimension the store does not have, or does not
            carry exactly the store's variables.
    """
    zarr_path = Path(zarr_path)

    # Non-index coordinates have to become variables, otherwise they are stored
    # once for the whole store instead of varying per episode.
    ds = ds.reset_coords()
    if time_dim in ds.coords:
        ds = ds.drop_vars(time_dim)

    # Encoding carried over from the episode's source file describes that file,
    # not this store. Its "coordinates" entry is the one that bites: xarray
    # re-emits it, and reading the store then turns the variables it names back
    # into coordinates, so the next episode no longer matches the store.
    for variable in ds.variables.values():
        variable.encoding = {}
    ds = _set_integer_fill_values(ds)

    if episode_dim not in ds.dims:
        ds = ds.set_coords(episode_dim).expand_dims(episode_dim)
    if ds.sizes[episode_dim] > 1:
        raise ValueError(
            f"Dataset holds {ds.sizes[episode_dim]} episodes, expected exactly one."
        )
    # Variables without the episode dimension would not be appended to
    for var in ds.data_vars:
        if episode_dim not in ds[var].dims:
            ds[var] = ds[var].expand_dims(episode_dim)

    # Pad up to the caller's upper bounds so the store never needs extending
    if dim_sizes:
        pad_to_bound = {}
        for dim, target_size in dim_sizes.items():
            if dim not in ds.dims or dim == episode_dim:
                continue
            if ds.sizes[dim] > target_size:
                raise ValueError(
                    f"Dimension {dim} has size {ds.sizes[dim]}, over the dim_sizes "
                    f"bound of {target_size}. dim_sizes must cover the largest episode."
                )
            if ds.sizes[dim] < target_size:
                pad_to_bound[dim] = (0, target_size - ds.sizes[dim])
        ds = _pad(ds, pad_to_bound)

    if not zarr_path.exists():
        logger.info(f"Creating Zarr store at {zarr_path}")
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
        return

    ds_store = xr.open_zarr(zarr_path, consolidated=True)

    # Appending a variable the store does not have creates it with a single
    # episode, leaving conflicting episode-dim sizes that break every later open
    store_vars = set(ds_store.data_vars)
    ds_vars = set(ds.data_vars)
    if ds_vars != store_vars:
        raise ValueError(
            f"Variable mismatch with the store at {zarr_path}: "
            f"missing from the episode {sorted(store_vars - ds_vars)}, "
            f"new in the episode {sorted(ds_vars - store_vars)}. "
            "Every episode must carry the same variables."
        )

    pad_dims = {}
    extend_dims = {}
    for dim in set(ds.dims) - {episode_dim}:
        if dim not in ds_store.dims:
            raise ValueError(f"Dimension {dim} is not in the store at {zarr_path}.")
        if dim == time_dim and store_time_dim_size is not None:
            store_dim_size = store_time_dim_size
        else:
            store_dim_size = ds_store.sizes[dim]
        if ds.sizes[dim] < store_dim_size:
            pad_dims[dim] = (0, store_dim_size - ds.sizes[dim])
        elif ds.sizes[dim] > store_dim_size:
            extend_dims[dim] = ds.sizes[dim] - store_dim_size

    for dim, n_extend in extend_dims.items():
        extend_zarr_along_dim(zarr_path, dim, n_extend)
    ds = _pad(ds, pad_dims)

    # a- appends only to the variables that carry episode_dim
    ds.to_zarr(zarr_path, mode="a-", append_dim=episode_dim, consolidated=True)
    # Stale consolidated metadata breaks later opens, so re-consolidate every time
    zarr.consolidate_metadata(zarr_path)


def _set_integer_fill_values(ds: xr.Dataset) -> xr.Dataset:
    """Give integer variables a _FillValue so NaN padding survives the round trip.

    Padding promotes an integer variable to float64 with NaN in the new slots.
    Appending that to a store whose variable is still an integer type re-encodes
    the NaN back to that type, and without a _FillValue to encode it as, the
    padding lands as a garbage integer (plus a serialization warning): an int8
    status pads as 0, which reads as a clean fit. The sentinel is the type's
    extreme value, which decodes back to NaN on read, so a real value must
    never be one (no flag or count of ours is anywhere near).

    Args:
        ds: One episode's dataset.

    Returns:
        The dataset, with a _FillValue in the encoding of every integer variable.
    """
    for name in ds.data_vars:
        dtype = ds[name].dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            fill = info.min if np.issubdtype(dtype, np.signedinteger) else info.max
            ds[name].encoding.setdefault("_FillValue", fill)
    return ds


def _pad(ds: xr.Dataset, pad_widths: dict[str, tuple[int, int]]) -> xr.Dataset:
    """NaN pad a dataset, keeping whole-number index coordinates whole.

    Padding an index coordinate leaves NaN in the new slots, which turns an
    integer index (channel number, boundary point number) into floats with a
    NaN tail. Those are ordinals, so they are simply extended instead. Index
    coordinates that carry a physical position (an R grid) keep the NaN, which
    is what "this episode has no point here" should look like.

    Args:
        ds: Dataset to pad.
        pad_widths: Slots to add per dimension, as (before, after).

    Returns:
        The padded dataset, unchanged if pad_widths is empty.
    """
    if not pad_widths:
        return ds
    ordinal_dtypes = {
        dim: ds[dim].dtype
        for dim in pad_widths
        if dim in ds.coords and np.issubdtype(ds[dim].dtype, np.integer)
    }
    ds = ds.pad(pad_widths)
    for dim, dtype in ordinal_dtypes.items():
        ds = ds.assign_coords({dim: np.arange(ds.sizes[dim], dtype=dtype)})
    return ds


def zarr_chunk(ds: xr.Dataset, chunk_spec: dict[str, int]) -> xr.Dataset:
    """Chunk a dataset, clearing the encoding that would override the new chunks.

    See https://stackoverflow.com/questions/67476513, zarr keeps the chunk sizes
    from the encoding unless they are deleted first.

    Args:
        ds: Dataset to chunk.
        chunk_spec: Chunk size per dimension.

    Returns:
        The chunked dataset.
    """
    ds = ds.chunk(chunk_spec)
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding.pop("chunks", None)
    return ds
