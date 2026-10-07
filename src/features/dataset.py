"""Temporal split, time series columns, and TimeSeriesDataSet builder.

Provides the data preparation layer between feature engineering and
model training: chronological train/test split, PyTorch Forecasting
compatibility columns (time_idx, group_id), and a TimeSeriesDataSet
constructor for TFT models.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import numpy as np
import pandas as pd


def temporal_train_test_split(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each pair chronologically: first N% of bars for train, rest for test.

    Per-pair split (not global cutoff) because pairs have different time ranges.
    Asserts max(train.timestamp) < min(test.timestamp) within each pair_id.

    Args:
        df: Feature DataFrame sorted by [pair_id, timestamp].
        split_ratio: Fraction of each pair's bars for training (default 0.8).

    Returns:
        (train_df, test_df) tuple.
    """
    df = df.sort_values(["pair_id", "timestamp"]).reset_index(drop=True)

    train_parts = []
    test_parts = []

    for pid, group in df.groupby("pair_id"):
        n = len(group)
        split_idx = int(n * split_ratio)
        # Ensure at least 1 row in test if there are enough rows
        if split_idx == n and n > 1:
            split_idx = n - 1

        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Assert per-pair no-leakage invariant
    for pid in train_df["pair_id"].unique():
        t_max = train_df[train_df["pair_id"] == pid]["timestamp"].max()
        if pid in test_df["pair_id"].unique():
            t_min = test_df[test_df["pair_id"] == pid]["timestamp"].min()
            assert t_max < t_min, (
                f"Temporal leak in {pid}: train max {t_max} >= test min {t_min}"
            )

    return train_df, test_df


def add_timeseries_columns(
    df: pd.DataFrame,
    pair_id_mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add time_idx (int, 0-based per pair) and group_id (int encoding of pair_id).

    time_idx: Within each pair_id, rows sorted by timestamp get indices 0, 1, 2, ...
    group_id: Each unique pair_id gets a stable integer ID.

    Args:
        df: Feature DataFrame with pair_id and timestamp columns.
        pair_id_mapping: Optional dict mapping pair_id -> int. If None,
            creates mapping from sorted unique pair_ids in df.

    Returns:
        DataFrame with time_idx and group_id columns added.
    """
    result = df.sort_values(["pair_id", "timestamp"]).reset_index(drop=True)

    # time_idx: 0-based contiguous per pair_id
    result["time_idx"] = result.groupby("pair_id").cumcount().astype(int)

    # group_id: stable int encoding
    if pair_id_mapping is None:
        unique_pairs = sorted(result["pair_id"].unique())
        pair_id_mapping = {pid: i for i, pid in enumerate(unique_pairs)}

    result["group_id"] = result["pair_id"].map(pair_id_mapping).astype(int)

    return result


def build_timeseries_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 12,
    max_prediction_length: int = 1,
):
    """Build a PyTorch Forecasting TimeSeriesDataSet from the feature DataFrame.

    Args:
        df: DataFrame with time_idx, group_id, and feature columns.
            Must already have add_timeseries_columns applied.
        max_encoder_length: Number of historical bars for encoder (default 12 = 48 hours).
        max_prediction_length: Number of bars to predict (default 1 = 4 hours).

    Returns:
        pytorch_forecasting.TimeSeriesDataSet configured for spread prediction.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    # Select numeric feature columns (exclude non-feature columns)
    exclude_cols = {"time_idx", "group_id", "timestamp", "spread", "pair_id"}
    time_varying_known = ["time_idx"]
    time_varying_unknown = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    # Ensure group_id is a string column for PyTorch Forecasting
    df = df.copy()
    df["group_id"] = df["group_id"].astype(str)

    # Fill NaN values in numeric feature columns with 0.
    # TimeSeriesDataSet does not accept NaN values in feature columns;
    # allow_missing_timesteps only handles missing rows, not missing values.
    for col in time_varying_unknown:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="spread",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        allow_missing_timesteps=True,
    )
