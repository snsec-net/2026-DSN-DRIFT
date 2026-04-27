import polars as pl
pl.Config.set_engine_affinity(engine="streaming")
from utility.path import path_period_data

def get_train_set():
    files = [
        path_period_data.joinpath("T17_benign_test.parquet"),
        path_period_data.joinpath("T17_dga_test.parquet"),
        path_period_data.joinpath("T18_benign_test.parquet"),
        path_period_data.joinpath("T18_dga_test.parquet"),
        path_period_data.joinpath("T19_benign_test.parquet"),
        path_period_data.joinpath("T19_dga_test.parquet"),

        path_period_data.joinpath("T17_benign_train.parquet"),
        path_period_data.joinpath("T17_dga_train.parquet"),
        path_period_data.joinpath("T18_benign_train.parquet"),
        path_period_data.joinpath("T18_dga_train.parquet"),
        path_period_data.joinpath("T19_benign_train.parquet"),
        path_period_data.joinpath("T19_dga_train.parquet"),
        ]

    return pl.read_parquet(files).unique(), files

def get_val_set():
    files = [
        path_period_data.joinpath("T17_benign_val.parquet"),
        path_period_data.joinpath("T17_dga_val.parquet"),
        path_period_data.joinpath("T18_benign_val.parquet"),
        path_period_data.joinpath("T18_dga_val.parquet"),
        path_period_data.joinpath("T19_benign_val.parquet"),
        path_period_data.joinpath("T19_dga_val.parquet"),
        ]

    return pl.read_parquet(files).unique()

def get_test_set_20():
    files = [
        path_period_data.joinpath("T20_benign.parquet"),
        path_period_data.joinpath("T20_dga.parquet"),
        ]

    return pl.read_parquet(files).unique()

def get_test_set_21():
    files = [
        path_period_data.joinpath("T21_benign.parquet"),
        path_period_data.joinpath("T21_dga.parquet"),
        ]

    return pl.read_parquet(files).unique()

def get_test_set_22():
    files = [
        path_period_data.joinpath("T22_benign.parquet"),
        path_period_data.joinpath("T22_dga.parquet"),
        ]

    return pl.read_parquet(files).unique()

def get_test_set_23():
    files = [
        path_period_data.joinpath("T23_benign.parquet"),
        path_period_data.joinpath("T23_dga.parquet"),
        ]

    return pl.read_parquet(files).unique()

def get_test_set_24():
    files = [
        path_period_data.joinpath("T24_benign.parquet"),
        path_period_data.joinpath("T24_dga.parquet"),
        ]

    return pl.read_parquet(files).unique()

def get_test_set_25():
    files = [
        path_period_data.joinpath("T25_benign.parquet"),
        path_period_data.joinpath("T25_dga.parquet"),
        ]

    return pl.read_parquet(files).unique()