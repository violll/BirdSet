from . import BirdSetDataModule
from birdset.configs import DataS1DatasetConfig, LoadersConfig
from typing import Literal

from birdset.utils import pylogger
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.event_mapping import XCEventMapping

from datasets import (
    load_dataset,
    Audio,
    DatasetDict,
    Dataset,
)

log = pylogger.get_pylogger(__name__)

class DataS1DataModule(BirdSetDataModule):
    """A BirdSetDataModule for the seabird (DataS1) dataset."""

    def __init__(
        self,
        dataset: DataS1DatasetConfig = None,
        loaders: LoadersConfig = None,
        transforms: BirdSetTransformsWrapper = None,
        mapper: BirdSetTransformsWrapper = None
    ):
        """Initializes the DataS1DataModule.

        Args:
            n_workers (int, optional): The number of worker processes used for data loading. Defaults to 3.
            val_split (float, optional): The proportion of the dataset reserved for validation. Defaults to 0.2.
            task (str, optional): Defines the type of task (e.g., 'multilabel' or 'multiclass'). Defaults to "multilabel".
            sample_rate (int, optional): The sample rate for audio data processing. Defaults to 32000.
            classlimit (int, optional): The maximum number of samples per class. If None, all samples are used. Defaults to 500.
            eventlimit (int, optional): Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed. Defaults to 5.
        """

        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper=mapper
        )
    
    def _load_data(self, decode: bool = False):
        """
        Loads the data.
        This method loads the data by calling the superclass's _load_data method.
        Args:
            decode (bool, optional): Whether to decode the data. Defaults to False.
        Returns:
            The loaded data.
        """
    
        log.info("> Loading data set.")

        train_dataset = load_dataset("parquet", data_files = self.dataset_config.train_parquet_path)["train"]
        test_5s_dataset = load_dataset("parquet", data_files = self.dataset_config.test_5s_parquet_path)["train"]

        dataset = DatasetDict({
            "train": train_dataset,
            "test_5s": test_5s_dataset
        })

        assert isinstance(dataset, DatasetDict | Dataset)

        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sample_rate,
                mono=True,
                decode=decode,
            ),
        )
        return dataset
    
    @property
    def num_classes(self):
        return 31
    