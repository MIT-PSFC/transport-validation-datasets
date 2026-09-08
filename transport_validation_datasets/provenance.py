"""Provenance attributes: what built a file, and from what.

Two things are recorded. build_provenance is this package as it runs: its
version, the git commit of the checkout it runs from (with a dirty flag), and
the versions of the packages the numbers depend on. source_provenance is what
the source package stamped on a shot when it was pulled, rewritten into
source_* attributes with the parts that cannot be trusted replaced (see
SOURCE_ATTR_KEYS). Every value is a string or a JSON string, so the same
attributes land in netCDF and Zarr alike, and merge_shot_attrs combines the
attributes of many shots into one set that says where they disagree.
"""

import dataclasses
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

PACKAGE = "transport_validation_datasets"
REPO_URL = "https://github.com/MIT-PSFC/transport-validation-datasets"
REPO_DIR = Path(__file__).resolve().parent.parent

# Paths whose uncommitted changes make a checkout dirty: what the code does
# and what it runs on. Scratch scripts and notes outside them do not count.
DIRTY_PATHS = (PACKAGE, "pyproject.toml", "uv.lock")

# Packages whose versions go into the dependency_versions attribute,
# by their distribution names. A package that is not installed is left out.
DEPENDENCIES = ("disruption-py", "mkgp")

# The attributes disruption-py's runner stamps on every dataset it returns
# (disruption_py.core.utils.misc.get_metadata). commit and source are not
# trusted: get_commit_hash runs `git rev-parse HEAD` in the working directory,
# so a build launched from a checkout of this repository records this
# repository's commit as disruption-py's, and a source URL that does not
# exist. user and host are dropped.
SOURCE_ATTR_KEYS = ("package", "version", "commit", "source", "time", "user", "host")

# Per-shot source attributes that differ between shots by nature and stay
# out of the stacked store.
SOURCE_VOLATILE_KEYS = ("source_retrieval_time",)


def _json_default(obj):
    """Serialize what json cannot: dataclasses, numpy, paths, sets.

    Args:
        obj: The object json.dumps could not handle.

    Returns:
        A json-serializable stand-in, str(obj) as the last resort.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (Path, set, frozenset)):
        return str(obj) if isinstance(obj, Path) else sorted(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def to_json(obj) -> str:
    """JSON string of obj for an attribute, sorted keys so it is byte-stable.

    Args:
        obj: Anything json handles, plus dataclasses, numpy, paths and sets.

    Returns:
        The JSON string.
    """
    return json.dumps(obj, default=_json_default, sort_keys=True)


def _git(*args: str) -> str | None:
    """Run git in the package's repository.

    Args:
        *args: The git subcommand and its arguments.

    Returns:
        Stripped stdout, or None when git is missing or the command fails.
    """
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_DIR, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@lru_cache
def git_state() -> dict[str, str]:
    """The git checkout this package runs from.

    Returns:
        commit, dirty ("true" or "false", uncommitted changes under
        DIRTY_PATHS, untracked files included), and branch. Empty when the
        package is not running from its own checkout (installed from a wheel,
        or git is missing): a git repository that merely encloses the install
        directory is not this package's.
    """
    toplevel = _git("rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() != REPO_DIR:
        return {}
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return {}
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *DIRTY_PATHS)
    state = {"commit": commit, "dirty": "true" if status else "false"}
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        state["branch"] = branch
    return state


def installed_package(name: str) -> dict[str, str]:
    """What is installed under a distribution name.

    Args:
        name: Distribution name (disruption-py, not disruption_py; both are
            accepted by importlib).

    Returns:
        version, plus commit and url when it was installed from a git URL
        (recorded by the installer in direct_url.json). Empty when it is not
        installed.
    """
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {}
    installed = {"version": dist.version}
    text = dist.read_text("direct_url.json")
    if text:
        direct = json.loads(text)
        commit = direct.get("vcs_info", {}).get("commit_id")
        if commit:
            installed["commit"] = commit
            installed["url"] = direct.get("url", "")
    return installed


@lru_cache
def build_provenance() -> dict[str, str]:
    """This package as it runs: version, git state, dependency versions.

    Byte-stable for a fixed checkout and environment, so it can go into the
    per-shot unprocessed files without making them differ between runs.

    Returns:
        <package>_version, <package>_commit / _dirty / _branch / _url (only
        from a git checkout), and dependency_versions, a JSON object of
        distribution name to version for DEPENDENCIES.
    """
    attrs = {}
    version = installed_package(PACKAGE).get("version")
    if version is not None:
        attrs[f"{PACKAGE}_version"] = version
    git = git_state()
    for key, value in git.items():
        attrs[f"{PACKAGE}_{key}"] = value
    if "commit" in git:
        attrs[f"{PACKAGE}_url"] = f"{REPO_URL}/tree/{git['commit']}"
    else:
        attrs[f"{PACKAGE}_url"] = REPO_URL
    versions = {
        name: installed_package(name)["version"]
        for name in DEPENDENCIES
        if installed_package(name)
    }
    attrs["dependency_versions"] = to_json(versions)
    return attrs


def build_stamp() -> dict[str, str]:
    """When and where a build happened. Volatile: for the stores, not the per-shot files.

    Returns:
        build_time (local time with UTC offset, to the second) and build_host.
    """
    return {
        "build_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "build_host": platform.node(),
    }


def _repo_root(source_url: str | None) -> str | None:
    """The repository URL a disruption-py source URL points into.

    Args:
        source_url: The stamped source attribute, a /tree/<commit> or
            /releases/tag/<tag> URL, or None.

    Returns:
        The URL up to that marker, or None when there is no URL.
    """
    if not source_url:
        return None
    for marker in ("/tree/", "/releases/"):
        if marker in source_url:
            return source_url.split(marker)[0]
    return source_url.rstrip("/")


def source_provenance(attrs: Mapping) -> dict:
    """Rewrite what the source package stamped on a shot into source_* attributes.

    Keeps the package name, the version it ran as, and when it pulled the
    shot. The commit and URL are rebuilt from what is installed: the recorded
    git commit when the package was installed from git and is the version
    that pulled the shot, otherwise the release tag of that version, because
    the stamped commit is the working directory's (see SOURCE_ATTR_KEYS).
    Anything else passes through unchanged, so attributes that are already
    source_* or this package's are left alone: applying this twice is a no-op.

    Args:
        attrs: Root attributes of a shot's dataset.

    Returns:
        The attributes with the SOURCE_ATTR_KEYS replaced by source_package,
        source_version, source_retrieval_time, source_commit and source_url,
        each present when there is something to say.
    """
    attrs = dict(attrs)
    out = {k: v for k, v in attrs.items() if k not in SOURCE_ATTR_KEYS}
    package = attrs.get("package")
    if package is None:
        return out
    out["source_package"] = str(package)
    version = attrs.get("version")
    if version is not None:
        out["source_version"] = str(version)
    if attrs.get("time") is not None:
        out["source_retrieval_time"] = str(attrs["time"])
    installed = installed_package(str(package))
    repo = _repo_root(attrs.get("source"))
    if "commit" in installed and installed.get("version") == version:
        out["source_commit"] = installed["commit"]
        repo = repo or installed.get("url", "").removesuffix(".git") or None
        if repo:
            out["source_url"] = f"{repo}/tree/{installed['commit']}"
    elif repo and version is not None:
        # Mirrors disruption-py's own URL when it has no commit
        tag = "v" + ".".join(str(version).split(".")[:2])
        out["source_url"] = f"{repo}/releases/tag/{tag}"
    return out


def merge_shot_attrs(
    per_shot: Iterable[Mapping], exclude: Iterable[str] = SOURCE_VOLATILE_KEYS
) -> dict:
    """Combine the root attributes of many shots into one set.

    A key every shot agrees on keeps its value. A key whose value differs
    between shots, or that only some shots carry, becomes a JSON list of the
    distinct values and is logged, so a store stacked from files pulled under
    two versions says so instead of reporting the first shot's.

    Args:
        per_shot: One attribute mapping per shot.
        exclude: Keys left out altogether, per-shot by nature.

    Returns:
        The merged attributes.
    """
    per_shot = [dict(attrs) for attrs in per_shot]
    exclude = set(exclude)
    # First-seen order of the keys, without the excluded ones
    keys = dict.fromkeys(
        key for attrs in per_shot for key in attrs if key not in exclude
    )
    missing = object()
    merged = {}
    for key in keys:
        distinct: list = []
        for attrs in per_shot:
            value = attrs.get(key, missing)
            if value not in distinct:
                distinct.append(value)
        if len(distinct) == 1 and distinct[0] is not missing:
            merged[key] = distinct[0]
        else:
            # None stands for the shots that do not carry the key at all
            values = [None if value is missing else value for value in distinct]
            logger.warning(
                f"Attribute {key!r} differs between the shots: {values}, "
                "recording every value"
            )
            merged[key] = to_json(values)
    return merged
