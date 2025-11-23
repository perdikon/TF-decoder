from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import  DataLoader, Dataset
import os

import torch.nn.functional as F
import copy

from src.utils.utils_cvae import read_log


from pm4py.objects.log.importer.xes import importer as xes_importer

import torch
import copy

from utils_cvae import get_dataset_attributes_info, read_log

from dataclasses import dataclass



@dataclass
class DatasetInfo:
    train_path: str
    val_path: str
    test_path: str
    full_path: str
    csv_separator: str
    activity_key: str
    timestamp_key: str
    resource_key: str
    label_key: str
    trace_key: str
    trace_attributes: list
    max_trace_length: int
    activities: list
    resources : list
    labels: list
    num_activities: int
    num_resources: int


class GenericDataset(Dataset):
    """
    dataset_path: path to the xes log
    max_trace_length: length of the longest trace in the log + 1 (for EOT activity)
    num_activities: number of different activities in the log + 1 (for EOT activity)
    num_labels: number of different labels in the log
    labels: list of possible labels
    trace_key: name of the column of the log containing traces id
    activity_key: name of the column of the log containing activity names
    timestamp_key: name of the column of the log containing the timestamp
    resource_key: name of the column of the log containing the resource
    label_key: name of the column of the log containing the label
    activities: (optional) list of activities to consider (used to impose a specific list of activities instead of using the ones found in the provided log)
    resources: (optional) list of resources to consider (used to impose a specific list of resources instead of using the ones found in the provided log)
    """

    def __init__(
        self,
        dataset_path="",
        max_trace_length=100,
        num_activities=10,
        num_labels=2,
        labels=None,
        trace_key="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
        resource_key="org:resource",
        label_key="case:label",
        trace_attributes=[],
        activities=None,
        resources=None,
        highest_ts=None,
    ):
        self.log = read_log(dataset_path, verbose=False)

        self.max_trace_length = max_trace_length
        self.num_activities = num_activities
        self.num_labels = num_labels
        self.trace_attributes = trace_attributes

        # Special activities
        self.EOT_ACTIVITY = "EOT"
        self.PADDING_ACTIVITY = "PAD"

        # Special resources
        self.EOT_RESOURCE = "EOT-RES"
        self.PADDING_RESOURCE = "PAD-RES"

        # Get activity names
        if activities is None:
            activities = self.log[activity_key].unique().tolist()
        else:
            activities = copy.deepcopy(activities)
        activities.sort()
        activities.append(self.EOT_ACTIVITY)
        activities.append(self.PADDING_ACTIVITY)

        assert (
            len(activities) - 1 == self.num_activities
        )  # Check whether num_activities and found number activities coincide

        # Mapping from activity name to index
        self.activity2n = {a: n for n, a in enumerate(activities)}
        self.n2activity = {n: a for n, a in enumerate(activities)}

        # Get resources
        if resources is None:
            resources = self.log[resource_key].unique().tolist()
        else:
            resources = copy.deepcopy(resources)

        # cast every item of resources to str
        resources = [str(r) for r in resources]

        resources.sort()
        resources.append(self.EOT_RESOURCE)
        resources.append(self.PADDING_RESOURCE)

        # Mapping from resource name to index
        self.resource2n = {r: n for n, r in enumerate(resources)}
        self.n2resource = {n: r for n, r in enumerate(resources)}

        # Get highest ts value
        self.highest_ts = (
            self.log[timestamp_key].quantile(q=0.95)
            if highest_ts is None
            else highest_ts
        )

        # Get labels
        if labels is None:
            labels = self.log[label_key].unique().tolist()
        else:
            labels = copy.deepcopy(labels)

        labels.sort()

        assert len(labels) == self.num_labels

        # Mapping from label name to one hot encoding
        self.label2onehot = {
            label: F.one_hot(torch.tensor(i), num_classes=self.num_labels)
            for i, label in enumerate(labels)
        }
        
        self.onehot2label = {
            tuple(onehot.tolist()): label 
            for label, onehot in self.label2onehot.items()
        }

        # Build trace attributes mapping
        self.s2i, self.i2s = {}, {}
        for trace_attr in trace_attributes:
            if trace_attr["type"] == "categorical":
                self.s2i[trace_attr["name"]] = {
                    a: n for n, a in enumerate(trace_attr["possible_values"])
                }
                self.i2s[trace_attr["name"]] = {
                    n: a for n, a in enumerate(trace_attr["possible_values"])
                }

        # Build dataset
        traces = list(self.log.groupby(trace_key).groups.values())
        self.x, self.y = [], []

        # trace attributes
        for trace in traces:
            x_attr = {}
            trace_idx = trace[0]
            for trace_attr in trace_attributes:
                attr = self.log.iloc[trace_idx][
                    trace_attr["name"]
                ]  # get attribute value

                if trace_attr["type"] == "categorical":
                    attr = self.s2i[trace_attr["name"]][
                        attr
                    ]  # convert attribute to index
                    attr = torch.tensor(attr, dtype=torch.int64)
                elif trace_attr["type"] == "numerical":
                    attr = (attr - trace_attr["min_value"]) / (
                        trace_attr["max_value"] - trace_attr["min_value"]
                    )  # min-max normalization
                    attr = torch.tensor(attr, dtype=torch.float32)
                else:
                    raise Exception(
                        f'Unknown trace attribute type: {trace_attr["type"]}'
                    )

                x_attr[trace_attr["name"]] = attr

            # activities
            x_trace = self.log.iloc[trace][activity_key].tolist()[
                : self.max_trace_length - 1
            ]  # get trace activities
            x_trace += [self.EOT_ACTIVITY]  # append End Of Trace token
            x_trace += [self.PADDING_ACTIVITY] * (
                self.max_trace_length - len(trace) - 1
            )  # append padding if needed
            x_trace = torch.tensor([self.activity2n[a] for a in x_trace]).to(
                torch.int
            )  # convert to tensor

            # timestamps
            x_ts = self.log.iloc[trace][timestamp_key].tolist()[
                : self.max_trace_length - 1
            ]  # get trace timestamps
            x_ts += [0]  # append timestamp for fictional "End Of Trace" activity
            x_ts += [0] * (
                self.max_trace_length - len(trace) - 1
            )  # append padding if needed
            x_ts = [ts / self.highest_ts for ts in x_ts]  # normalize
            x_ts = torch.tensor(x_ts).to(torch.float32)  # convert to tensor

            # resources
            x_res = self.log.iloc[trace][resource_key].tolist()[
                : self.max_trace_length - 1
            ]
            x_res += [self.EOT_RESOURCE]
            x_res += [self.PADDING_RESOURCE] * (self.max_trace_length - len(trace) - 1)
            x_res = [str(x) for x in x_res]
            x_res = torch.tensor([self.resource2n[r] for r in x_res]).to(torch.int)

            # label
            y = self.log.iloc[trace][label_key].tolist()[0]  # get label from log
            y = self.label2onehot[y].to(torch.float32)  # convert to one-hot tensor

            self.x.append((x_attr, x_trace, x_ts, x_res))
            self.y.append(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class SepsisDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_name: str = "sepsis",
        max_trace_length=50,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
                
        self.generator =  torch.Generator().manual_seed(42)
        csv_file = os.path.abspath(
                os.path.join(self.hparams.data_dir, dataset_name, "sepsis.csv")
            )
        
        
        dataset_attr_info = get_dataset_attributes_info(
            csv_file,
            activity_key="Activity",
            trace_key="Case ID",
            resource_key="Resource",
            trace_attributes=["relative_timestamp_from_start", "Diagnose", "Age"],
        )
        
        self.dataset_name = dataset_name


        

        self.dataset_info = DatasetInfo(
            train_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "sepsis_TRAIN.csv")
            ),
            val_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "sepsis_VAL.csv")
            ), 
            test_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "sepsis_TEST.csv")
            ),
            full_path=csv_file,
            csv_separator= ";",
            activity_key="Activity",
            timestamp_key="relative_timestamp_from_previous_activity",
            resource_key="Resource",
            label_key="label",
            trace_key = 'Case ID',
            trace_attributes=dataset_attr_info["trace_attributes"],
            max_trace_length=max_trace_length + 1,
            activities= dataset_attr_info['activities'],
            resources= dataset_attr_info['resources'],
            labels = dataset_attr_info['labels'],
            num_activities = len(dataset_attr_info['activities']) + 1,
            num_resources = len(dataset_attr_info["resources"]) + 1
        )
        


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        test = 1

    def prepare_data(self):
        pass
        # # Ensure the raw data directory exists.
        # os.makedirs(self.raw_dir, exist_ok=True)

        # # DOWNLOAD
        # if not os.path.exists(self.xes_file):
        #     self._download_sepsis_data()

        # # PREPROCESS
        # if not os.path.exists(self.pp_csv_file):
        #     self._preprocess_sepsis_data()

        # # SPLIT
        # splits_exist = all(
        #     os.path.exists(os.path.join(self.raw_dir, f"sepsis_pp_{split}.csv"))
        #     for split in ["TRAIN", "VAL", "TEST"]
        # )
        # if not splits_exist:
        #     self._split_temporal(split_perc=self.hparams.train_val_test_split)

        # # LOAD DATASETS
        # if self.full_dataset is None:
        #     # Read the preprocessed data saved in sepsis_pp.csv
        #     # self.full_dataset = SepsisDataset(
        #     #     pp_file=self.pp_csv_file, trace_attributes=self.trace_attributes
        #     # )
        #     pass

        # # WORK HERE
        # # fix categorical trace attributes to have possible values, and then pass trace attributes to models

        # for trace_attr in self.trace_attributes:
        #     if trace_attr["type"] == "categorical":
        #         # Get the unique values from the log
        #         unique_values = (
        #             self.full_dataset.log[trace_attr["name"]].dropna().unique().tolist()
        #         )
        #         # Sort the unique values
        #         unique_values.sort()
        #         # Update the trace attribute with possible values
        #         trace_attr["possible_values"] = unique_values

    def setup(self, stage=None):
        
        if stage in (None, "fit"):
            # TRAIN
            self.data_train  = GenericDataset(
            dataset_path=self.dataset_info.train_path,
            max_trace_length=self.dataset_info.max_trace_length,
            num_activities=self.dataset_info.num_activities,
            labels=self.dataset_info.labels,
            num_labels=len(self.dataset_info.labels),
            trace_attributes=self.dataset_info.trace_attributes,
            activities = self.dataset_info.activities,
            resources = self.dataset_info.resources,
            activity_key=self.dataset_info.activity_key,
            timestamp_key=self.dataset_info.timestamp_key,
            resource_key=self.dataset_info.resource_key,
            trace_key = self.dataset_info.trace_key,
            label_key=self.dataset_info.label_key,
        )

            # VALIDATION
            self.data_val  = GenericDataset(
            dataset_path=self.dataset_info.val_path,
            max_trace_length=self.dataset_info.max_trace_length,
            num_activities=self.dataset_info.num_activities,
            labels=self.dataset_info.labels,
            num_labels=len(self.dataset_info.labels),
            trace_attributes=self.dataset_info.trace_attributes,
            activities = self.dataset_info.activities,
            resources = self.dataset_info.resources,
            activity_key=self.dataset_info.activity_key,
            timestamp_key=self.dataset_info.timestamp_key,
            resource_key=self.dataset_info.resource_key,
            trace_key = self.dataset_info.trace_key,
            label_key=self.dataset_info.label_key,
        )

        if stage in (None, "test"):
            
            # TEST
            self.data_test  = GenericDataset(
            dataset_path=self.dataset_info.test_path,
            max_trace_length=self.dataset_info.max_trace_length,
            num_activities=self.dataset_info.num_activities,
            labels=self.dataset_info.labels,
            num_labels=len(self.dataset_info.labels),
            trace_attributes=self.dataset_info.trace_attributes,
            activities = self.dataset_info.activities,
            resources = self.dataset_info.resources,
            activity_key=self.dataset_info.activity_key,
            timestamp_key=self.dataset_info.timestamp_key,
            resource_key=self.dataset_info.resource_key,
            trace_key = self.dataset_info.trace_key,
            label_key=self.dataset_info.label_key,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            generator=self.generator
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            generator= self.generator
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
        )


class BPICDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_name: str = "bpic2012_a",
        max_trace_length: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.generator = torch.Generator().manual_seed(42)
        self.dataset_name = dataset_name
        
        csv_file = os.path.abspath(
            os.path.join(self.hparams.data_dir, self.dataset_name, "bpic2012_a.csv")
        )
        
        # Specific trace attributes for bpic2012_a
        trace_attrs = ["relative_timestamp_from_start", "AMOUNT_REQ"]
        
        dataset_attr_info = get_dataset_attributes_info(
            csv_file,
            activity_key="Activity",
            trace_key="Case ID",
            resource_key="Resource",
            trace_attributes=trace_attrs,
        )
        
        self.dataset_info = DatasetInfo(
            train_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "bpic2012_a_TRAIN.csv")
            ),
            val_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "bpic2012_a_VAL.csv")
            ), 
            test_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "bpic2012_a_TEST.csv")
            ),
            full_path=csv_file,
            csv_separator=";",
            activity_key="Activity",
            timestamp_key="relative_timestamp_from_previous_activity",
            resource_key="Resource",
            label_key="label",
            trace_key="Case ID",
            trace_attributes=dataset_attr_info["trace_attributes"],
            max_trace_length=max_trace_length + 1,
            activities=dataset_attr_info['activities'],
            resources=dataset_attr_info['resources'],
            labels=dataset_attr_info['labels'],
            num_activities=len(dataset_attr_info['activities']) + 1,
            num_resources=len(dataset_attr_info["resources"]) + 1
        )
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass
        # BPIC2012_a-specific preprocessing could be added here

    def setup(self, stage=None):
        if stage in (None, "fit"):
            # TRAIN
            self.data_train = GenericDataset(
                dataset_path=self.dataset_info.train_path,
                max_trace_length=self.dataset_info.max_trace_length,
                num_activities=self.dataset_info.num_activities,
                labels=self.dataset_info.labels,
                num_labels=len(self.dataset_info.labels),
                trace_attributes=self.dataset_info.trace_attributes,
                activities=self.dataset_info.activities,
                resources=self.dataset_info.resources,
                activity_key=self.dataset_info.activity_key,
                timestamp_key=self.dataset_info.timestamp_key,
                resource_key=self.dataset_info.resource_key,
                trace_key=self.dataset_info.trace_key,
                label_key=self.dataset_info.label_key,
            )

            # VALIDATION
            self.data_val = GenericDataset(
                dataset_path=self.dataset_info.val_path,
                max_trace_length=self.dataset_info.max_trace_length,
                num_activities=self.dataset_info.num_activities,
                labels=self.dataset_info.labels,
                num_labels=len(self.dataset_info.labels),
                trace_attributes=self.dataset_info.trace_attributes,
                activities=self.dataset_info.activities,
                resources=self.dataset_info.resources,
                activity_key=self.dataset_info.activity_key,
                timestamp_key=self.dataset_info.timestamp_key,
                resource_key=self.dataset_info.resource_key,
                trace_key=self.dataset_info.trace_key,
                label_key=self.dataset_info.label_key,
            )

        if stage in (None, "test"):
            # TEST
            self.data_test = GenericDataset(
                dataset_path=self.dataset_info.test_path,
                max_trace_length=self.dataset_info.max_trace_length,
                num_activities=self.dataset_info.num_activities,
                labels=self.dataset_info.labels,
                num_labels=len(self.dataset_info.labels),
                trace_attributes=self.dataset_info.trace_attributes,
                activities=self.dataset_info.activities,
                resources=self.dataset_info.resources,
                activity_key=self.dataset_info.activity_key,
                timestamp_key=self.dataset_info.timestamp_key,
                resource_key=self.dataset_info.resource_key,
                trace_key=self.dataset_info.trace_key,
                label_key=self.dataset_info.label_key,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            generator=self.generator
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            generator=self.generator
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
        )


class EmergencyORTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_name: str = "emergency_ORT",
        max_trace_length: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.generator = torch.Generator().manual_seed(42)
        self.dataset_name = dataset_name
        
        csv_file = os.path.abspath(
            os.path.join(self.hparams.data_dir, self.dataset_name, "emergency_ORT.csv")
        )
        
        
        dataset_attr_info = get_dataset_attributes_info(
            csv_file,
            activity_key="Activity",
            trace_key="Case ID",
            resource_key="Resource",
            trace_attributes=["relative_timestamp_from_start", "Gender"],
        )
        
        self.dataset_info = DatasetInfo(
            train_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "emergency_ORT_TRAIN.csv")
            ),
            val_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "emergency_ORT_VAL.csv")
            ), 
            test_path=os.path.abspath(
                os.path.join(self.hparams.data_dir, self.dataset_name, "emergency_ORT_TEST.csv")
            ),
            full_path=csv_file,
            csv_separator=";",
            activity_key="Activity",
            # Expect a numeric relative timestamp column for modeling; adjust if different
            timestamp_key="relative_timestamp_from_previous_activity",
            resource_key="Resource",
            label_key="label",
            trace_key="Case ID",
            trace_attributes=dataset_attr_info["trace_attributes"],
            max_trace_length=max_trace_length + 1,
            activities=dataset_attr_info['activities'],
            resources=dataset_attr_info['resources'],
            labels=dataset_attr_info['labels'],
            num_activities=len(dataset_attr_info['activities']) + 1,
            num_resources=len(dataset_attr_info["resources"]) + 1
        )
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass
        # Emergency ORT specific preprocessing could be added here

    def setup(self, stage=None):
        if stage in (None, "fit"):
            # TRAIN
            self.data_train = GenericDataset(
                dataset_path=self.dataset_info.train_path,
                max_trace_length=self.dataset_info.max_trace_length,
                num_activities=self.dataset_info.num_activities,
                labels=self.dataset_info.labels,
                num_labels=len(self.dataset_info.labels),
                trace_attributes=self.dataset_info.trace_attributes,
                activities=self.dataset_info.activities,
                resources=self.dataset_info.resources,
                activity_key=self.dataset_info.activity_key,
                timestamp_key=self.dataset_info.timestamp_key,
                resource_key=self.dataset_info.resource_key,
                trace_key=self.dataset_info.trace_key,
                label_key=self.dataset_info.label_key,
            )

            # VALIDATION
            self.data_val = GenericDataset(
                dataset_path=self.dataset_info.val_path,
                max_trace_length=self.dataset_info.max_trace_length,
                num_activities=self.dataset_info.num_activities,
                labels=self.dataset_info.labels,
                num_labels=len(self.dataset_info.labels),
                trace_attributes=self.dataset_info.trace_attributes,
                activities=self.dataset_info.activities,
                resources=self.dataset_info.resources,
                activity_key=self.dataset_info.activity_key,
                timestamp_key=self.dataset_info.timestamp_key,
                resource_key=self.dataset_info.resource_key,
                trace_key=self.dataset_info.trace_key,
                label_key=self.dataset_info.label_key,
            )

        if stage in (None, "test"):
            # TEST
            self.data_test = GenericDataset(
                dataset_path=self.dataset_info.test_path,
                max_trace_length=self.dataset_info.max_trace_length,
                num_activities=self.dataset_info.num_activities,
                labels=self.dataset_info.labels,
                num_labels=len(self.dataset_info.labels),
                trace_attributes=self.dataset_info.trace_attributes,
                activities=self.dataset_info.activities,
                resources=self.dataset_info.resources,
                activity_key=self.dataset_info.activity_key,
                timestamp_key=self.dataset_info.timestamp_key,
                resource_key=self.dataset_info.resource_key,
                trace_key=self.dataset_info.trace_key,
                label_key=self.dataset_info.label_key,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            generator=self.generator
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            generator=self.generator
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
        )
