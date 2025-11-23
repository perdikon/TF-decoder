from pathlib import Path

import pytest
import torch

import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.data.sepsis_datamodule import SepsisDataModule, SepsisDataset
from src.data.mnist_datamodule import MNISTDataModule
import os


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    
    
# Fixture that creates a configuration for the SepsisDataModule.
@pytest.fixture
def config() -> OmegaConf:
    # Define a YAML configuration string.
    config_yaml = """
    _target_: src.data.sepsis_datamodule.SepsisDataModule
    data_dir: "tests/data"
    batch_size: 128
    train_val_test_split: [0.7, 0.15, 0.15]
    num_workers: 0
    pin_memory: False
    """
    return OmegaConf.create(config_yaml)


@pytest.fixture
def datamodule(config) -> SepsisDataModule:
    dm = instantiate(config)
    return dm


def test_sepsis_initialization(datamodule: SepsisDataModule, config: OmegaConf) -> None:
    """
    Tests the initialization of SepsisDataModule by verifying that:
      - Hyperparameters are correctly loaded.
      - The computed file paths (raw_dir, zip_path, gz_file, xes_file) are set as expected.
      - The dataset splits and full_dataset are not initialized.
    """
    # Verify hyperparameters.
    assert datamodule.hparams.data_dir == "tests/data", "data_dir should be 'tests/data'"
    assert datamodule.hparams.batch_size == 128, "batch_size should be 128"
    assert datamodule.hparams.train_val_test_split == [629, 135, 135], "train_val_test_split should be [629, 135, 135]"
    assert datamodule.hparams.num_workers == 0, "num_workers should be 0"
    assert datamodule.hparams.pin_memory is False, "pin_memory should be False"

    # Compute expected file paths.
    expected_raw_dir = os.path.join("tests/data", "raw")
    expected_zip_path = os.path.join(expected_raw_dir, "Sepsis Cases - Event Log_1_all.zip")
    expected_gz_file = os.path.join(expected_raw_dir, "Sepsis Cases - Event Log.xes.gz")
    expected_xes_file = os.path.join(expected_raw_dir, "Sepsis Cases - Event Log.xes")

    # Verify computed file paths.
    assert datamodule.raw_dir == expected_raw_dir, f"raw_dir should be {expected_raw_dir}"
    assert datamodule.zip_path == expected_zip_path, f"zip_path should be {expected_zip_path}"
    assert datamodule.gz_file == expected_gz_file, f"gz_file should be {expected_gz_file}"
    assert datamodule.xes_file == expected_xes_file, f"xes_file should be {expected_xes_file}"

    # Verify that dataset splits and full_dataset are not yet initialized.
    assert datamodule.full_dataset is None, "full_dataset should be None on initialization"
    assert datamodule.data_train is None, "data_train should be None on initialization"
    assert datamodule.data_val is None, "data_val should be None on initialization"
    assert datamodule.data_test is None, "data_test should be None on initialization"
    
    
def test_sepsis_prepare_data(datamodule, config) -> None:
    """
    Integration test for SepsisDataModule.prepare_data.
    
    This test uses the original data URL to download the zip file, extract the
    gzipped XES file, decompress it, and load the full dataset. It then verifies
    that the expected files (zip, gz, decompressed XES) exist and that full_dataset
    is loaded.
    """
    # Create a temporary data directory.
    data_dir = Path(datamodule.hparams.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    
    # Ensure the raw data directory is empty (if it exists) so that prepare_data will run.
    raw_dir = Path(datamodule.raw_dir)
    if raw_dir.exists():
        for f in raw_dir.iterdir():
            if f.is_file():
                f.unlink()
    else:
        raw_dir.mkdir(parents=True)
    
    # Call prepare_data(), which should download, extract, decompress and load the dataset.
    datamodule.prepare_data()
    
    # Verify that the raw directory exists.
    assert raw_dir.exists(), "Raw directory should exist after prepare_data()"
    
    # Verify that the final decompressed XES file exists.
    xes_file = Path(datamodule.xes_file)
    assert xes_file.exists(), "Decompressed XES file should exist after decompression"
    
    # Check that the final XES file is non-empty.
    file_size = os.path.getsize(xes_file)
    assert file_size > 1000, f"Decompressed XES file size ({file_size} bytes) is unexpectedly small."
    
    # Verify that the full dataset is loaded.
    assert datamodule.full_dataset is not None, "full_dataset should be loaded after prepare_data()"
    
    
def test_getitem_dimensions(datamodule: SepsisDataModule):
    """
    This test ensures that after prepare_data() is called and the full dataset is loaded,
    retrieving a sample returns a dictionary with the expected keys and tensor shapes,
    specifically that "activity_seq" is one-hot encoded with shape (max_seq_len, vocab_size)
    and each row sums to 1.
    """
    # Trigger dataset loading.
    datamodule.prepare_data()
    dataset = datamodule.full_dataset
    sample = dataset[0]

    # Ensure that the sample contains the expected keys.
    expected_keys = {"activity_seq", "timestamp_diff", "time_interval_matrix", "duration", "mask"}
    assert expected_keys.issubset(sample.keys()), f"Sample keys {sample.keys()} do not contain expected keys {expected_keys}"

    # Retrieve maximum sequence length and vocabulary size.
    max_seq_len = dataset.max_seq_len
    vocab_size = len(dataset.activity2idx)

    # Check that "activity_seq" is one-hot encoded.
    act_seq = sample["activity_seq"]
    assert isinstance(act_seq, torch.Tensor), "activity_seq should be a torch.Tensor"
    # Expect a 2D tensor: (max_seq_len, vocab_size)
    assert act_seq.dim() == 2, f"activity_seq should be 2D, got {act_seq.dim()} dimensions"
    assert act_seq.shape[0] == max_seq_len, f"Expected activity_seq first dimension {max_seq_len}, got {act_seq.shape[0]}"
    assert act_seq.shape[1] == vocab_size, f"Expected activity_seq second dimension {vocab_size}, got {act_seq.shape[1]}"
    
    # Verify that each row of the one-hot encoded tensor sums to 1.
    row_sums = act_seq.sum(dim=1)
    expected_sums = torch.ones(max_seq_len, dtype=act_seq.dtype)
    assert torch.allclose(row_sums, expected_sums, atol=1e-6), f"Each row should sum to 1, got {row_sums}"

    # Check dimensions for timestamp_diff: should be a 1D tensor of length max_seq_len.
    ts_diff = sample["timestamp_diff"]
    assert isinstance(ts_diff, torch.Tensor), "timestamp_diff should be a torch.Tensor"
    assert ts_diff.dim() == 1, f"timestamp_diff should be 1D, got {ts_diff.dim()} dimensions"
    assert ts_diff.shape[0] == max_seq_len, f"Expected timestamp_diff length {max_seq_len}, got {ts_diff.shape[0]}"

    # Check that time_interval_matrix is a 2D tensor of shape (max_seq_len, max_seq_len).
    time_interval_matrix = sample["time_interval_matrix"]
    assert isinstance(time_interval_matrix, torch.Tensor), "time_interval_matrix should be a torch.Tensor"
    assert time_interval_matrix.dim() == 2, f"time_interval_matrix should be 2D, got {time_interval_matrix.dim()} dimensions"
    assert time_interval_matrix.shape == (max_seq_len, max_seq_len), f"Expected time_interval_matrix shape {(max_seq_len, max_seq_len)}, got {time_interval_matrix.shape}"

    # Check that duration is a scalar tensor.
    duration = sample["duration"]
    assert isinstance(duration, torch.Tensor), "duration should be a torch.Tensor"
    assert duration.dim() == 0, f"Expected duration to be a scalar tensor, got dimension {duration.dim()}"

    # Check that mask is a 1D tensor of length max_seq_len.
    mask = sample["mask"]
    assert isinstance(mask, torch.Tensor), "mask should be a torch.Tensor"
    assert mask.dim() == 1, f"mask should be 1D, got {mask.dim()} dimensions"
    assert mask.shape[0] == max_seq_len, f"Expected mask length {max_seq_len}, got {mask.shape[0]}"
    
    
    
def test_sepsis_setup(datamodule: SepsisDataModule, config: OmegaConf) -> None:
    """
    Test that SepsisDataModule.setup() correctly sets global attributes and
    splits the full dataset into train/val/test according to the configured split.
    """
    # Load the full dataset.
    datamodule.prepare_data()
    assert datamodule.full_dataset is not None, "full_dataset should be loaded after prepare_data()"
    
    # Before calling setup, the data splits should be None.
    assert datamodule.data_train is None, "data_train should be None before setup()"
    assert datamodule.data_val is None, "data_val should be None before setup()"
    assert datamodule.data_test is None, "data_test should be None before setup()"


    datamodule.setup()

    # Verify that global properties are stored from the full_dataset.
    dataset = datamodule.full_dataset
    assert hasattr(datamodule, 'activity2idx'), "activity2idx should be set after setup()"
    assert datamodule.activity2idx == dataset.activity2idx, "activity2idx does not match full_dataset"
    assert datamodule.max_seq_len == dataset.max_seq_len, "max_seq_len does not match full_dataset"
    assert datamodule.min_duration == dataset.min_duration, "min_duration does not match full_dataset"
    assert datamodule.max_duration == dataset.max_duration, "max_duration does not match full_dataset"

    # Check that the data splits are now created.
    assert datamodule.data_train is not None, "data_train should be initialized after setup()"
    assert datamodule.data_val is not None, "data_val should be initialized after setup()"
    assert datamodule.data_test is not None, "data_test should be initialized after setup()"
    
    # Verify that the total number of samples in the splits equals the length of full_dataset.
    total_split_length = len(datamodule.data_train) + len(datamodule.data_val) + len(datamodule.data_test)
    full_length = len(dataset)
    assert total_split_length == full_length, (
        f"Sum of splits ({total_split_length}) should equal full_dataset length ({full_length})"
    )
    
    # Optionally, check that the split lengths match the hyperparameters.
    expected_lengths = [int(ratio * len(datamodule.full_dataset)) for ratio in config.train_val_test_split]
    expected_lengths[2] =  len(datamodule.full_dataset) - expected_lengths[0] - expected_lengths[1]
    assert len(datamodule.data_train) == expected_lengths[0], (
        f"Expected data_train length {expected_lengths[0]}, got {len(datamodule.data_train)}"
    )
    assert len(datamodule.data_val) == expected_lengths[1], (
        f"Expected data_val length {expected_lengths[1]}, got {len(datamodule.data_val)}"
    )
    assert len(datamodule.data_test) == expected_lengths[2], (
        f"Expected data_test length {expected_lengths[2]}, got {len(datamodule.data_test)}"
    )

    # Verify that calling setup again does not re-split the dataset.
    train_len, val_len, test_len = len(datamodule.data_train), len(datamodule.data_val), len(datamodule.data_test)
    datamodule.setup()
    assert len(datamodule.data_train) == train_len, "data_train length changed after calling setup() again"
    assert len(datamodule.data_val) == val_len, "data_val length changed after calling setup() again"
    assert len(datamodule.data_test) == test_len, "data_test length changed after calling setup() again"