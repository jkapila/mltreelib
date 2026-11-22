"""Shared JSON interface for unsupervised models with native snapshots."""

# Python 3.10 lacks typing.Self. Keep TypeVar without a runtime dependency.
# ruff: noqa: PYI019

import json
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeVar

_SnapshotT = TypeVar("_SnapshotT", bound="_UnsupervisedSnapshot")


class _UnsupervisedSnapshot(ABC):
    _snapshot_format: str
    _snapshot_engines: tuple[str, ...]
    _fitted: bool
    _inner: Any
    kind: str

    @abstractmethod
    def _snapshot_config(self) -> dict[str, Any]: ...

    @abstractmethod
    def _snapshot_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def _restore_snapshot_state(self, state: dict[str, Any]) -> None: ...

    @classmethod
    def _check_snapshot_engine(cls, kind: str) -> None:
        if kind not in cls._snapshot_engines:
            supported = ", ".join(cls._snapshot_engines)
            raise ValueError(
                f"JSON snapshots are available for {supported}, got {kind!r}."
            )

    def to_json(self) -> str:
        """Return the fitted model as a versioned JSON string."""
        if not self._fitted:
            raise ValueError(f"{type(self).__name__} is not fitted yet.")
        self._check_snapshot_engine(self.kind)
        native, category_levels = self._inner._snapshot()
        return json.dumps(
            {
                "format": self._snapshot_format,
                "format_version": 1,
                "configuration": self._snapshot_config(),
                "state": self._snapshot_state(),
                "category_levels": category_levels,
                "native": json.loads(native),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )

    def snapshot(self) -> str:
        """Return the fitted model as a JSON snapshot string."""
        return self.to_json()

    @classmethod
    def from_snapshot(cls: type[_SnapshotT], data: str) -> _SnapshotT:
        """Restore a fitted model from a JSON snapshot string."""
        return cls.from_json(data)

    @classmethod
    def from_json(cls: type[_SnapshotT], data: str) -> _SnapshotT:
        """Restore a fitted model from a versioned JSON string."""
        snapshot = json.loads(data)
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("format") != cls._snapshot_format
        ):
            raise ValueError(f"Expected a {cls.__name__} snapshot.")
        version = snapshot.get("format_version")
        if type(version) is not int or version != 1:
            raise ValueError(f"Unsupported {cls.__name__} format version {version}.")
        try:
            configuration = snapshot["configuration"]
            cls._check_snapshot_engine(configuration["kind"])
            model = cls(**configuration)
            model._inner = model._inner._from_snapshot(
                json.dumps(snapshot["native"], allow_nan=False),
                snapshot["category_levels"],
            )
            model._restore_snapshot_state(snapshot["state"])
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(f"Invalid {cls.__name__} snapshot.") from error
        model._fitted = True
        return model

    def to_bytes(self) -> bytes:
        """Return the fitted model as UTF-8 JSON bytes."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_bytes(cls: type[_SnapshotT], data: bytes) -> _SnapshotT:
        """Restore a fitted model from UTF-8 JSON bytes."""
        return cls.from_json(data.decode("utf-8"))

    def save(self, path: str | Path) -> None:
        """Write the JSON snapshot, replacing the file after a complete write."""
        destination = Path(path)
        data = self.to_bytes()
        temp = None
        try:
            with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temp = Path(stream.name)
                stream.write(data)
            temp.replace(destination)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    @classmethod
    def load(cls: type[_SnapshotT], path: str | Path) -> _SnapshotT:
        """Restore a fitted model from a JSON file."""
        return cls.from_bytes(Path(path).read_bytes())

    def __getstate__(self) -> bytes:
        return self.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        restored = type(self).from_bytes(state)
        self.__dict__.update(restored.__dict__)
