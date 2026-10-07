"""CLI pipeline: load aligned_pairs.parquet, compute features, split, save.

Reads aligned_pairs.parquet from Phase 2.1, computes derived features,
performs temporal train/test split, adds PyTorch Forecasting columns,
and saves train.parquet and test.parquet.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import argparse
import logging
import os

import pandas as pd

from src.features.dataset import add_timeseries_columns, temporal_train_test_split
from src.features.engineering import compute_derived_features

logger = logging.getLogger(__name__)


def build_feature_pipeline(
    input_path: str = "data/processed/aligned_pairs.parquet",
    output_dir: str = "data/processed",
    split_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned pairs, compute features, split, add timeseries cols, save.

    Steps:
    1. pd.read_parquet(input_path)
    2. compute_derived_features(df)
    3. temporal_train_test_split(df, split_ratio)
    4. add_timeseries_columns to both train and test
       (fit group_id mapping on train, apply same mapping to test)
    5. Save train.parquet and test.parquet to output_dir
    6. Print summary stats

    Args:
        input_path: Path to aligned_pairs.parquet.
        output_dir: Directory to save train.parquet and test.parquet.
        split_ratio: Fraction of each pair's bars for training (default 0.8).

    Returns:
        (train_df, test_df) tuple with all features and timeseries columns.
    """
    # Step 1: Load aligned pairs
    logger.info("Loading aligned pairs from %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows, %d pairs, %d columns", len(df), df["pair_id"].nunique(), len(df.columns))

    # Step 2: Compute derived features
    logger.info("Computing derived features...")
    df = compute_derived_features(df)
    logger.info("Feature DataFrame: %d rows, %d columns", len(df), len(df.columns))

    # Step 3: Temporal train/test split
    logger.info("Splitting with ratio %.2f...", split_ratio)
    train_df, test_df = temporal_train_test_split(df, split_ratio=split_ratio)
    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Step 4: Add timeseries columns
    # Fit group_id mapping on train, apply same mapping to test
    unique_pairs = sorted(train_df["pair_id"].unique())
    pair_id_mapping = {pid: i for i, pid in enumerate(unique_pairs)}

    # Handle any pair_id that appears only in test (safety)
    for pid in test_df["pair_id"].unique():
        if pid not in pair_id_mapping:
            pair_id_mapping[pid] = len(pair_id_mapping)

    train_df = add_timeseries_columns(train_df, pair_id_mapping=pair_id_mapping)
    test_df = add_timeseries_columns(test_df, pair_id_mapping=pair_id_mapping)

    # Step 5: Save outputs
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info("Saved train.parquet (%d rows) and test.parquet (%d rows)", len(train_df), len(test_df))

    # Step 6: Print summary stats
    print(f"Input: {input_path}")
    print(f"Train: {len(train_df)} rows, {train_df['pair_id'].nunique()} pairs, {len(train_df.columns)} columns")
    print(f"Test:  {len(test_df)} rows, {test_df['pair_id'].nunique()} pairs, {len(test_df.columns)} columns")
    print(f"Columns: {list(train_df.columns)}")
    print(f"Derived features: price_velocity, volume_ratio, spread_momentum, spread_volatility, kalshi_order_flow_imbalance, polymarket_order_flow_imbalance")
    print(f"Saved to: {train_path}, {test_path}")

    return train_df, test_df


def main():
    """CLI entry point for feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Build feature matrix from aligned_pairs.parquet."
    )
    parser.add_argument(
        "--input",
        default="data/processed/aligned_pairs.parquet",
        help="Path to aligned_pairs.parquet (default: data/processed/aligned_pairs.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for train.parquet and test.parquet (default: data/processed)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    build_feature_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
