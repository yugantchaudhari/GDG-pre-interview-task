"""
app/services/dataset_service.py
=================================
DatasetService — responsible for all raw dataset I/O operations.

Responsibilities
----------------
* Resolving the dataset file path from ``.env`` or a caller-supplied override.
* Validating that the file exists and is non-empty.
* Confirming the 7 required columns are present.
* Optionally deduplicating the DataFrame and reporting how many rows were removed.

No value is hardcoded.  The required column set is defined as a module-level
constant (not a magic string) and all path references come from ``Settings``.
No ``print()`` calls are used anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.settings import get_settings
from app.exceptions.custom_exceptions import (
    DatasetNotFoundException,
    InvalidDatasetSchemaException,
)
from app.utils.logger import LoggerFactory

log = LoggerFactory.get_logger(__name__)

# The canonical set of columns expected in every insurance dataset.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
)


class DatasetService:
    """
    Service layer for all raw dataset file operations.

    Uses ``get_settings()`` exclusively for file-path resolution so the caller
    never needs to know where the data lives on disk.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_dataset_availability(
        self, custom_path: str | None = None
    ) -> bool:
        """
        Return ``True`` if the dataset file exists on disk and is non-empty.

        Parameters
        ----------
        custom_path:
            Optional override path. Uses ``DATASET_FILE_PATH`` from ``.env``
            when ``None``.

        Returns
        -------
        bool
            ``True`` if the file exists and has content; ``False`` otherwise.
        """
        path = self._resolve_path(custom_path)
        exists = path.exists() and path.stat().st_size > 0
        log.debug("Dataset availability check for '%s': %s", path, exists)
        return exists

    def load_raw_dataset(
        self, custom_path: str | None = None
    ) -> pd.DataFrame:
        """
        Load and schema-validate the raw insurance CSV dataset.

        Parameters
        ----------
        custom_path:
            Optional override path.

        Returns
        -------
        pd.DataFrame
            The validated raw DataFrame (including duplicates, before cleaning).

        Raises
        ------
        DatasetNotFoundException
            When the resolved path does not exist or is empty.
        InvalidDatasetSchemaException
            When required columns are missing or the DataFrame is empty after load.
        """
        path = self._resolve_path(custom_path)

        if not path.exists():
            log.error(
                "Dataset file not found at resolved path: '%s'", path
            )
            raise DatasetNotFoundException(
                message=f"Dataset file not found: '{path}'",
                details={"resolved_path": str(path)},
            )

        if path.stat().st_size == 0:
            log.error("Dataset file at '%s' is empty (0 bytes).", path)
            raise DatasetNotFoundException(
                message=f"Dataset file is empty: '{path}'",
                details={"resolved_path": str(path)},
            )

        df = pd.read_csv(path)

        if df.empty:
            log.error("Dataset at '%s' loaded successfully but contains 0 rows.", path)
            raise InvalidDatasetSchemaException(
                message="Dataset loaded but contains no rows.",
                details={"resolved_path": str(path)},
            )

        missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            log.error(
                "Dataset schema validation failed — missing columns: %s. "
                "Present columns: %s",
                missing_cols,
                list(df.columns),
            )
            raise InvalidDatasetSchemaException(
                message=f"Dataset is missing required columns: {missing_cols}",
                details={
                    "missing_columns": missing_cols,
                    "present_columns": list(df.columns),
                    "resolved_path": str(path),
                },
            )

        log.info(
            "Dataset loaded successfully from '%s': %d rows × %d columns.",
            path,
            len(df),
            len(df.columns),
        )
        return df

    def get_clean_dataset(
        self,
        drop_duplicates: bool = True,
        custom_path: str | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """
        Load the raw dataset and optionally remove fully duplicate rows.

        Parameters
        ----------
        drop_duplicates:
            When ``True``, identical rows are removed and the count reported.
        custom_path:
            Optional override path.

        Returns
        -------
        tuple[pd.DataFrame, int]
            ``(cleaned_df, duplicates_removed)`` where ``duplicates_removed``
            is 0 when ``drop_duplicates=False``.
        """
        df = self.load_raw_dataset(custom_path)
        duplicates_removed = int(df.duplicated().sum())

        if drop_duplicates and duplicates_removed > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            log.info(
                "Removed %d duplicate row(s) — dataset now has %d rows.",
                duplicates_removed,
                len(df),
            )
        elif duplicates_removed > 0:
            log.info(
                "Detected %d duplicate row(s) — kept (drop_duplicates=False).",
                duplicates_removed,
            )
        else:
            log.debug("No duplicate rows detected in dataset.")

        return df, duplicates_removed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, custom_path: str | None) -> Path:
        """Return the resolved dataset path from override or settings."""
        if custom_path:
            return Path(custom_path)
        return self.settings.DATASET_FILE_PATH
