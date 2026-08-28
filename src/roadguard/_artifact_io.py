"""Private filesystem and byte-level primitives for Phase 13 artifact bundles."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import joblib  # type: ignore[import-untyped]

BUFFER_SIZE: Final[int] = 65_536
MAX_MANIFEST_BYTES: Final[int] = 1_048_576
PATH_ERROR: Final[str] = "Phase 13 artifact path validation failed."
SERIALIZATION_ERROR: Final[str] = "Phase 13 artifact serialization failed."
WRITE_ERROR: Final[str] = "Phase 13 artifact write failed."
VERIFICATION_ERROR: Final[str] = "Phase 13 artifact verification failed."


class ArtifactPersistenceError(RuntimeError):
    """Raised when a Phase 13 artifact bundle cannot be safely published."""


def dump_joblib(value: Any, path: Path) -> None:
    try:
        with path.open("xb") as handle:
            joblib.dump(value, handle, compress=0, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, ValueError, TypeError):
        raise ArtifactPersistenceError(SERIALIZATION_ERROR) from None


def write_bytes(path: Path, value: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        raise ArtifactPersistenceError(WRITE_ERROR) from None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_root(value: Path) -> Path:
    if str(value).startswith("\\\\") or value == Path(value.anchor):
        raise ArtifactPersistenceError(PATH_ERROR)
    try:
        if not value.exists():
            require_safe_components(value)
            if not value.parent.is_dir():
                raise OSError
            value.mkdir()
        require_safe_components(value)
        require_safe_path(value, value.parent)
        if not value.is_dir():
            raise OSError
        return value.resolve(strict=True)
    except OSError:
        raise ArtifactPersistenceError(PATH_ERROR) from None


def mkdir_checked(path: Path) -> None:
    try:
        path.mkdir(exist_ok=True)
        require_safe_path(path, path.parent)
    except OSError:
        raise ArtifactPersistenceError(PATH_ERROR) from None


def require_safe_path(path: Path, parent: Path) -> None:
    try:
        details = path.lstat()
        attributes = int(getattr(details, "st_file_attributes", 0))
        if stat.S_ISLNK(details.st_mode) or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise OSError
        resolved_parent = parent.resolve(strict=True)
        if not path.resolve(strict=True).is_relative_to(resolved_parent):
            raise OSError
    except (AttributeError, OSError, RuntimeError):
        raise ArtifactPersistenceError(PATH_ERROR) from None


def require_safe_components(path: Path) -> None:
    """Reject observable links, reparse points, and mounts below the anchor."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            if not current.exists():
                continue
            details = current.lstat()
            attributes = int(getattr(details, "st_file_attributes", 0))
            if stat.S_ISLNK(details.st_mode) or bool(
                attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise OSError
            if current != Path(absolute.anchor) and os.path.ismount(current):
                raise OSError
            if current != path and not stat.S_ISDIR(details.st_mode):
                raise OSError
    except (OSError, RuntimeError):
        raise ArtifactPersistenceError(PATH_ERROR) from None


def rename_without_replacement(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a collision target."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        raise OSError(ctypes.get_errno(), "atomic no-replace rename failed")


def sync_directory(directory: Path) -> None:
    """Synchronize a directory when the active platform exposes directory FDs."""
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    descriptor = os.open(directory, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_bundle(
    directory: Path,
    digest: str,
    expected_manifest_bytes: bytes,
    expected_artifacts: Iterable[tuple[str, int, str]],
    artifact_filenames: tuple[str, ...],
) -> None:
    try:
        require_safe_path(directory, directory.parent)
        files = {path.name: path for path in directory.iterdir()}
        if set(files) != set(artifact_filenames) or any(
            not is_regular_file(path) for path in files.values()
        ):
            raise OSError
        manifest_path = files["manifest.json"]
        require_safe_path(manifest_path, directory)
        manifest_bytes = read_limited(manifest_path, MAX_MANIFEST_BYTES)
        if hashlib.sha256(manifest_bytes).hexdigest() != digest:
            raise OSError
        if manifest_bytes != expected_manifest_bytes:
            raise OSError
        for filename, size_bytes, file_digest in expected_artifacts:
            path = files[filename]
            require_safe_path(path, directory)
            if path.stat().st_size != size_bytes or sha256(path) != file_digest:
                raise OSError
    except ArtifactPersistenceError as exc:
        if str(exc) == VERIFICATION_ERROR:
            raise
        raise ArtifactPersistenceError(VERIFICATION_ERROR) from None
    except (OSError, TypeError, ValueError):
        raise ArtifactPersistenceError(VERIFICATION_ERROR) from None


def read_limited(path: Path, maximum_bytes: int) -> bytes:
    size = path.stat().st_size
    if size < 0 or size > maximum_bytes:
        raise OSError
    chunks: list[bytes] = []
    remaining = size
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(BUFFER_SIZE, remaining))
            if not chunk:
                raise OSError
            chunks.append(chunk)
            remaining -= len(chunk)
        if handle.read(1):
            raise OSError
    return b"".join(chunks)


def is_regular_file(path: Path) -> bool:
    details = path.lstat()
    attributes = int(getattr(details, "st_file_attributes", 0))
    return stat.S_ISREG(details.st_mode) and not bool(
        attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def file_identity(path: Path) -> tuple[int, int]:
    details = path.stat(follow_symlinks=False)
    return details.st_dev, details.st_ino


def cleanup_stage(stage: Path, parent: Path, expected_identity: tuple[int, int]) -> None:
    try:
        if stage.exists() and stage.parent.resolve(strict=True) == parent.resolve(strict=True):
            require_safe_path(stage, parent)
            if file_identity(stage) == expected_identity:
                shutil.rmtree(stage)
    except (ArtifactPersistenceError, OSError):
        return
