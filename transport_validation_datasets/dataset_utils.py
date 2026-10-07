"""Assemble per-episode datasets into one tensorized Zarr store, one episode at a time.

Based on popsim/data/dataset_utils.py
"""

import os
import shutil
from collections.abc import Callable, Hashable, Iterable
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
    dim_sizes: dict[str, int],
    mb_per_chunk: int | None = 10,
) -> xr.Dataset:
    """Build a tensorized multi-episode dataset from one-episode datasets.

    process_fn turns one identifier at a time into that episode's dataset, or None to skip it.
    Each episode is appended to the Zarr store as it comes, so peak memory is one episode.
    An episode whose process_fn raises is logged with its traceback and skipped,
    so a single unreadable shot does not lose the rest of the run.

    Args:
        process_fn: Turns one identifier into that episode's dataset, or None to skip the episode.
        identifiers: Episode identifiers, passed to process_fn one by one.
        zarr_path: Path of the Zarr store to write, must end in .zarr and not exist.
        time_dim: Name of the per-episode time dimension.
        episode_dim: Name of the dimension episodes are stacked along.
        dim_sizes: Upper bound per non-episode dimension.
            Every episode is padded to these sizes before it is written,
            so the store is never extended, which would rewrite every chunk already in it.
            The rechunking pass trims the slack.
        mb_per_chunk: Target size of each variable's chunks.
            Each variable gets its own episodes per chunk from its own size per episode,
            so a 0D signal packs many more episodes into a chunk than a 2D map.
            None leaves the chunking alone.

    Returns:
        The assembled dataset, opened from the finished store.

    Raises:
        ValueError: If zarr_path does not end in .zarr or already exists.
        RuntimeError: If no episode was written.
    """
    zarr_path = Path(zarr_path)
    if zarr_path.suffix != ".zarr":
        raise ValueError(f"zarr_path must end with .zarr, got {zarr_path}")
    if zarr_path.exists():
        raise ValueError(f"Zarr store at {zarr_path} already exists, remove it first.")
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    for identifier in identifiers:
        try:
            ds_episode = process_fn(identifier)
        except Exception:
            logger.opt(exception=True).warning(
                f"Error processing {identifier}, skipping it"
            )
            n_skipped += 1
            continue
        if ds_episode is None:
            continue
        add_to_zarr_store(ds_episode, zarr_path, time_dim, episode_dim, dim_sizes)
        n_written += 1
        # Released before the next episode is built
        del ds_episode

    if n_skipped:
        logger.warning(f"Skipped {n_skipped} episodes that raised while processing")
    if n_written == 0:
        raise RuntimeError(f"No episodes were written to {zarr_path}")

    logger.info(f"Wrote {n_written} episodes to {zarr_path}, now chunking it")
    ds_store = xr.open_zarr(zarr_path, consolidated=True)
    if mb_per_chunk is None:
        return ds_store

    # The rechunking pass rewrites the store anyway, so it drops the padding slack of dim_sizes
    for dim in dim_sizes:
        if dim in ds_store.dims and dim != episode_dim:
            ds_store = trim_trailing_nan_slices(ds_store, dim)
    chunk_specs = episode_chunk_specs(ds_store, episode_dim, mb_per_chunk)

    # Written beside the old store and swapped in, so a crash mid-rewrite leaves the original intact
    tmp_path = zarr_path.with_suffix(".zarr.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    write_rechunked_store(ds_store, tmp_path, episode_dim, chunk_specs)
    ds_store.close()
    shutil.rmtree(zarr_path)
    os.rename(tmp_path, zarr_path)
    return xr.open_zarr(zarr_path, consolidated=True)


def trim_trailing_nan_slices(ds: xr.Dataset, dim: str) -> xr.Dataset:
    """Drop the trailing slices along dim that are NaN in every variable.

    Removes the slack left when episodes were padded up to an upper bound.

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

    mask_slice_has_data = slice_has_data.compute().values
    idx_slices_with_data = np.flatnonzero(mask_slice_has_data)
    if idx_slices_with_data.size == 0:
        return ds
    n_kept = int(idx_slices_with_data[-1]) + 1
    if n_kept < ds.sizes[dim]:
        logger.info(
            f"Trimming {ds.sizes[dim] - n_kept} trailing all-NaN slices along {dim}"
        )
        ds = ds.isel({dim: slice(0, n_kept)})
    return ds


def add_to_zarr_store(
    ds: xr.Dataset,
    zarr_path: Path | str,
    time_dim: str,
    episode_dim: str,
    dim_sizes: dict[str, int],
):
    """Append one episode to a Zarr store, creating the store if it is not there.

    The episode is NaN padded so every non-episode dimension lines up with the store.
    The store itself is never extended.

    Args:
        ds: One episode's dataset.
        zarr_path: Path of the Zarr store.
        time_dim: Name of the per-episode time dimension.
        episode_dim: Name of the dimension episodes are stacked along.
        dim_sizes: Upper bound per non-episode dimension, see build_tensorized_dataset.

    Raises:
        ValueError: If ds holds more than one episode,
            exceeds a dim_sizes bound or the store's size of a dimension,
            carries a dimension the store does not have,
            does not carry exactly the store's variables,
            or gives a non-episode coordinate other values than the store holds.
    """
    zarr_path = Path(zarr_path)

    # Non-index coordinates become variables, else they are stored once for the whole store
    ds = ds.reset_coords()
    if time_dim in ds.coords:
        ds = ds.drop_vars(time_dim)

    # Encoding carried over from the episode's source file describes that file, not this store.
    # Its "coordinates" entry would turn the variables it names back into coordinates on read,
    # so the next episode would no longer match the store.
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

    pad_to_bound = {}
    for dim, size_bound in dim_sizes.items():
        if dim not in ds.dims or dim == episode_dim:
            continue
        if ds.sizes[dim] > size_bound:
            raise ValueError(
                f"Dimension {dim} has size {ds.sizes[dim]}, over the dim_sizes "
                f"bound of {size_bound}. dim_sizes must cover the largest episode."
            )
        if ds.sizes[dim] < size_bound:
            pad_to_bound[dim] = (0, size_bound - ds.sizes[dim])
    ds = _pad(ds, pad_to_bound)

    if not zarr_path.exists():
        logger.info(f"Creating Zarr store at {zarr_path}")
        # consolidated=True consolidates the metadata on write, every append below does the same
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
        return

    with xr.open_zarr(zarr_path, consolidated=True) as ds_store:
        # A variable the store does not have would be created with a single episode,
        # and its episode-dim size would then break every later open
        store_vars = set(ds_store.data_vars)
        episode_vars = set(ds.data_vars)
        if episode_vars != store_vars:
            raise ValueError(
                f"Variable mismatch with the store at {zarr_path}: "
                f"missing from the episode {sorted(store_vars - episode_vars)}, "
                f"new in the episode {sorted(episode_vars - store_vars)}. "
                "Every episode must carry the same variables."
            )

        pad_to_store = {}
        for dim in set(ds.dims) - {episode_dim}:
            if dim not in ds_store.dims:
                raise ValueError(f"Dimension {dim} is not in the store at {zarr_path}.")
            store_size = ds_store.sizes[dim]
            if ds.sizes[dim] > store_size:
                raise ValueError(
                    f"Dimension {dim} has size {ds.sizes[dim]}, over the store's "
                    f"{store_size} at {zarr_path}. dim_sizes must cover the largest episode."
                )
            if ds.sizes[dim] < store_size:
                pad_to_store[dim] = (0, store_size - ds.sizes[dim])
        ds = _pad(ds, pad_to_store)
        _check_coordinates_match(ds, ds_store, episode_dim, zarr_path)

    # a- appends only to the variables that carry episode_dim
    ds.to_zarr(zarr_path, mode="a-", append_dim=episode_dim, consolidated=True)


def _check_coordinates_match(
    ds: xr.Dataset, ds_store: xr.Dataset, episode_dim: str, zarr_path: Path
):
    """Refuse an episode whose non-episode coordinates differ from the store's.

    An append never rewrites a coordinate without the episode dimension,
    so an episode on another grid (an R grid, a rho_tor_norm grid)
    would be stored under the first episode's values, silently misplaced.
    Compared after padding, so the episode's grid must match the store's slot for slot.

    Args:
        ds: The padded episode.
        ds_store: The open store.
        episode_dim: Name of the dimension episodes are stacked along.
        zarr_path: Path of the store, for the message.

    Raises:
        ValueError: If a coordinate differs, NaN padding counting as equal to NaN.
    """
    for name in ds.coords:
        if name == episode_dim or name not in ds_store.coords:
            continue
        episode_values = np.asarray(ds[name].values)
        store_values = np.asarray(ds_store[name].values)
        if episode_values.dtype.kind in "fc" and store_values.dtype.kind in "fc":
            same = np.array_equal(episode_values, store_values, equal_nan=True)
        else:
            same = np.array_equal(episode_values, store_values)
        if not same:
            raise ValueError(
                f"Coordinate {name} of the episode differs from the store's at {zarr_path}. "
                "Every episode must sit on the store's grids."
            )


def _set_integer_fill_values(ds: xr.Dataset) -> xr.Dataset:
    """Give integer variables a _FillValue so NaN padding survives the round trip.

    Padding promotes an integer variable to float64 with NaN in the new slots.
    Appended to a store that holds it as an integer, the NaN is encoded back to that integer type,
    and without a _FillValue it lands as garbage: an int8 status pads as 0, which reads as a clean fit.
    The fill value is the type's extreme value, which decodes back to NaN on read,
    so no real value may take it.

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
    """NaN pad a dataset, extending its integer index coordinates instead.

    Padding leaves NaN in the new slots of an index coordinate.
    An integer one is an ordinal (a channel or boundary point number), so it is extended.
    A physical one (an R grid) keeps the NaN, for "this episode has no point here".

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
        ordinals = np.arange(ds.sizes[dim], dtype=dtype)
        ds = ds.assign_coords({dim: ordinals})
    return ds


def episode_chunk_specs(
    ds: xr.Dataset,
    episode_dim: str,
    mb_per_chunk: float,
) -> dict[Hashable, dict[str, int]]:
    """Chunk sizes for every variable, chunking across episodes only.

    Zarr chunks every variable separately, so each one is sized on its own.
    Every other dimension stays in one chunk.

    Args:
        ds: Dataset to chunk, every variable carrying episode_dim.
        episode_dim: Name of the dimension episodes are stacked along.
        mb_per_chunk: Target size of each variable's chunks,
            divided by the variable's own size per episode.

    Returns:
        Chunk size per dimension, per variable, for zarr_chunk.
    """
    n_episodes = ds.sizes[episode_dim]
    chunk_specs = {}
    var_names_by_episodes_per_chunk = {}
    for name in ds.data_vars:
        mb_per_episode = ds[name].nbytes / n_episodes / (1024 * 1024)
        episodes_per_chunk = int(mb_per_chunk / mb_per_episode)
        episodes_per_chunk = min(max(1, episodes_per_chunk), n_episodes)
        chunk_specs[name] = dict(ds[name].sizes) | {episode_dim: episodes_per_chunk}
        var_names_by_episodes_per_chunk.setdefault(episodes_per_chunk, []).append(name)
    for episodes_per_chunk, var_names in sorted(
        var_names_by_episodes_per_chunk.items()
    ):
        logger.info(
            f"Chunking {episodes_per_chunk} episodes per chunk "
            f"(mb_per_chunk={mb_per_chunk}): {', '.join(var_names)}"
        )
    return chunk_specs


def write_rechunked_store(
    ds: xr.Dataset,
    zarr_path: Path | str,
    episode_dim: str,
    chunk_specs: dict[Hashable, dict[str, int]],
):
    """Write a dataset to a new Zarr store with new chunks, one chunk at a time.

    One dask compute over the whole dataset reads source chunks far ahead of the writes,
    which can run a workstation out of memory on thousands of shots.
    Filling each variable one block of episodes at a time holds one chunk in memory.

    Args:
        ds: Dataset to write, its data variables dask backed and chunked across episodes only,
            its coordinates in memory.
        zarr_path: Path of the new Zarr store, must not exist.
        episode_dim: Name of the dimension episodes are stacked along.
        chunk_specs: Chunk size per dimension, per data variable, from episode_chunk_specs.
    """
    ds = zarr_chunk(ds, chunk_specs)
    # Writes the metadata and the in-memory coordinates, the dask-backed variables are filled below
    ds.to_zarr(zarr_path, mode="w-", consolidated=True, compute=False)
    n_episodes = ds.sizes[episode_dim]
    for name, chunk_spec in chunk_specs.items():
        chunk_episodes = chunk_spec[episode_dim]
        for start in range(0, n_episodes, chunk_episodes):
            stop = min(start + chunk_episodes, n_episodes)
            region = slice(start, stop)
            variable_block = ds[name].variable.isel({episode_dim: region})
            variable_block = variable_block.compute()
            ds_block = xr.Dataset({name: variable_block})
            ds_block.to_zarr(zarr_path, mode="r+", region={episode_dim: region})
    zarr.consolidate_metadata(zarr_path)


def zarr_chunk(
    ds: xr.Dataset, chunk_specs: dict[Hashable, dict[str, int]]
) -> xr.Dataset:
    """Chunk each variable of a dataset on its own.

    Clears the chunk encoding, which zarr would otherwise keep over the new chunks
    (https://stackoverflow.com/questions/67476513).

    Args:
        ds: Dataset to chunk.
        chunk_specs: Chunk size per dimension, per variable.
            Variables left out keep their chunks.

    Returns:
        The chunked dataset.
    """
    ds = ds.assign(
        {name: ds[name].chunk(chunk_spec) for name, chunk_spec in chunk_specs.items()}
    )
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding.pop("chunks", None)
    return ds
