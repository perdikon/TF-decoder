import os
import json
import pm4py
import torch
import pandas as pd
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from os import devnull

def read_log(dataset_path, separator=';', timestamp_key='time:timestamp', verbose=True):
  """Read xes or csv logs"""
  with suppress_stdout_stderr() if verbose is False else nullcontext():
    if dataset_path.endswith('.xes'):
      log = pm4py.read_xes(dataset_path)
    elif dataset_path.endswith('.csv'):
      log = pd.read_csv(dataset_path, sep=separator)
      log[timestamp_key] = pd.to_datetime(log[timestamp_key], format='mixed')
    else:
      raise ValueError("Unsupported file extension")
    
  return log


def get_dataset_attributes_info(
  dataset_path,
  activity_key='concept:name',
  trace_key='case:concept:name',
  resource_key='org:group',
  trace_attributes=[],
):
  dataset_attributes_info = {}

  log = read_log(dataset_path, verbose=False)

  # Compute list of activities
  dataset_attributes_info['activities'] = log[activity_key].unique().tolist()

  # Compute list of resources
  resources = log[resource_key].unique().tolist()
  resources = [str(r) for r in resources]
  resources = list(set(resources))
  dataset_attributes_info['resources'] = resources

  # Compute max trace length
  traces = list(log.groupby(trace_key).groups.values())
  traces_lengths = [len(trace) for trace in traces]
  dataset_attributes_info['max_trace_length'] = max(traces_lengths)

  # Get info about each trace attribute
  dataset_attributes_info['trace_attributes'] = []
  for trace_attr in trace_attributes:
    possible_values = log[trace_attr].unique().tolist()
    possible_values.sort()
    is_numerical = all([isinstance(v, (int, float, complex)) for v in possible_values])

    trace_attribute_info = {
      'name': trace_attr,
      'type': 'numerical' if is_numerical else 'categorical',
    }

    if is_numerical:
      trace_attribute_info['min_value'] = min(possible_values)
      trace_attribute_info['max_value'] = max(possible_values)
    else:
      trace_attribute_info['possible_values'] = possible_values

    dataset_attributes_info['trace_attributes'].append(trace_attribute_info)

  return dataset_attributes_info


@contextmanager
def suppress_stdout_stderr():
  """A context manager that redirects stdout and stderr to devnull"""
  with open(devnull, 'w') as fnull:
    with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
      yield (err, out)


def move_to_device(data, device):
  """Move to specified device a list or dictionary of tensors"""
  if isinstance(data, list):
    return [move_to_device(item, device) for item in data]
  elif isinstance(data, dict):
    return {key: move_to_device(value, device) for key, value in data.items()}
  elif isinstance(data, torch.Tensor):
    return data.to(device)
  else:
    return data


def save_dict_to_json(d, filepath, indent=2):
  with open(filepath, mode='w') as f:
    json.dump(d, f, indent=indent)

def load_dict_from_json(filepath):
  with open(filepath, mode='r') as f:
    return json.load(f)
  
  
def split_temporal(log_path, log_name, output_path, split_perc=[0.7, 0.1, 0.12], csv_sep=';', case_id_key='case:concept:name', timestamp_key='time:timestamp'):
  assert len(split_perc) == 3 # train, val, test
  assert 0.9999999999 <= sum(split_perc) <= 1.0000000001

  log = read_log(log_path, separator=csv_sep)
  train_perc, val_perc, test_perc = split_perc

  # sort cases by timestamp of first activity
  cases = list(log[case_id_key].unique())
  cases_by_start_time = { case: log[log[case_id_key] == case][timestamp_key].min() for case in cases }
  cases_by_start_time = sorted(cases_by_start_time.items(), key=lambda item: item[1])

  # split train-val-test
  train_cases = cases_by_start_time[:int(len(cases)*train_perc)]
  val_cases = cases_by_start_time[int(len(cases)*train_perc):int(len(cases)*(train_perc+val_perc))]
  test_cases = cases_by_start_time[int(len(cases)*(train_perc+val_perc)):]

  assert len(train_cases) + len(val_cases) + len(test_cases) == len(cases)

  train_cases = [train_case[0] for train_case in train_cases]
  train = log[log[case_id_key].isin(train_cases)]

  val_cases = [val_case[0] for val_case in val_cases]
  val = log[log[case_id_key].isin(val_cases)]

  test_cases = [test_case[0] for test_case in test_cases]
  test = log[log[case_id_key].isin(test_cases)]

  print(f'Train cases: {len(train_cases)}, Val cases: {len(val_cases)}, Test cases: {len(test_cases)}')

  if log_path.endswith('.xes'):
    pm4py.write_xes(train, os.path.join(output_path, f'{log_name}_TRAIN.xes'))
    pm4py.write_xes(val, os.path.join(output_path, f'{log_name}_VAL.xes'))
    pm4py.write_xes(test, os.path.join(output_path, f'{log_name}_TEST.xes'))
  else:
    train.to_csv(os.path.join(output_path, f'{log_name}_TRAIN.csv'), sep=csv_sep, index=False)
    val.to_csv(os.path.join(output_path, f'{log_name}_VAL.csv'), sep=csv_sep, index=False)
    test.to_csv(os.path.join(output_path, f'{log_name}_TEST.csv'), sep=csv_sep, index=False)