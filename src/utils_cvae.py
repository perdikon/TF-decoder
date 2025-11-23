import os
import json
import pm4py
import torch
import numpy as np
import pandas as pd
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from os import devnull
from torch import Tensor
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



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
  label_key='label',
  trace_key='case:concept:name',
  resource_key='org:resource',
  trace_attributes=[],
):
  dataset_attributes_info = {}

  log = read_log(dataset_path, verbose=False)

  # Compute list of activities
  dataset_attributes_info['activities'] = log[activity_key].unique().tolist()

  # Compute list of labels
  dataset_attributes_info['labels'] = log[label_key].unique().tolist()

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
  
 
 
@torch.jit.script
def _ed_pair_trim(a: Tensor, b: Tensor) -> Tensor:
    """
    Levenshtein distance between two 1-D padded int64 tensors.
    Trailing zeros are ignored (zeros are **only** padding).
    Returns a 0-d float tensor.
    """
    # -------- find true lengths (first zero or full length) --------
    L1 = a.size(0)
    lenA = L1
    for i in range(L1):
        if a[i] == 0:
            lenA = i
            break

    L2 = b.size(0)
    lenB = L2
    for j in range(L2):
        if b[j] == 0:
            lenB = j
            break

    # -------- DP table (lenA+1) × (lenB+1) ------------------------
    dp = torch.zeros(lenA + 1, lenB + 1,
                     dtype=torch.int32, device=a.device)

    # initialise first row / column
    for i in range(1, lenA + 1):
        dp[i, 0] = i
    for j in range(1, lenB + 1):
        dp[0, j] = j

    # classic Levenshtein DP
    for i in range(1, lenA + 1):
        ai = a[i - 1]
        for j in range(1, lenB + 1):
            cost = (ai != b[j - 1]).to(torch.int32)
            delete  = dp[i - 1, j]   + 1
            insert  = dp[i,     j-1] + 1
            subst   = dp[i - 1, j-1] + cost
            dp[i, j] = torch.min(torch.min(delete, insert), subst)

    return dp[lenA, lenB].float()     # scalar result

  


def seq_to_bytes(t: torch.Tensor) -> bytes:
    """
    Trim right-padding 0s from a 1-D int64 tensor and pack the real tokens
    (1 … 255) into a Python bytes object.
    """
    if t.ndim != 1:
        raise ValueError("expect 1-D tensor")
    # keep only the prefix before the first 0
    if (t == 0).any():
        t = t[: int((t != 0).float().argmin().item())]
    return bytes(t.tolist())              # each token → one byte
  
  
  
from rapidfuzz.distance import Levenshtein    # Levenshtein scorer
from rapidfuzz.process  import cdist 

def ed_gram(A: torch.Tensor, B: torch.Tensor,
                   workers: int = -1) -> torch.Tensor:
    """
    A, B: (B, L) 0-padded int64 tensors on any device.
    Returns: (B, B) float32 tensor *on CPU* with Levenshtein distances.
    """
    A_list = [seq_to_bytes(x) for x in A.to('cpu', non_blocking=True)]
    B_list = [seq_to_bytes(x) for x in B.to('cpu', non_blocking=True)]

    # fast C++ bulk distance; returns a Python list of lists
    D_python = cdist(A_list, B_list,
                     scorer=Levenshtein.distance,
                     dtype=np.int32,
                     workers=workers)          # maximise CPU utilisation

    # convert to a torch Tensor (still on CPU)
    return torch.tensor(D_python, dtype=torch.float32)
  
  
def ed_laplacian(ta: torch.Tensor,
                        tb: torch.Tensor,
                        min_lambda: float = 1e-3) -> torch.Tensor:
    """
    Compute a Laplacian-kernel matrix from edit distances, using the
    median heuristic on the 'ta' batch to choose the decay rate λ.

    Args:
        ta: Tensor of shape (B, L)  — batch of B sequences
        tb: Tensor of shape (B, L)  — another batch of B sequences
        min_lambda: minimum value for λ to avoid zero or tiny rates

    Returns:
        K: Tensor of shape (B, B) with
           K[i,j] = exp(-λ * D_cross[i,j]),
        where λ = 1/median(D_aa[i<j]).
    """
    B = ta.size(0)

    # 1) compute within‐batch edit distances
    D_aa = ed_gram(ta, ta)                       # shape (B, B)

    # 2) extract the strict upper‐triangle and take median
    idx = torch.triu_indices(B, B, offset=1)
    pairwise = D_aa[idx[0], idx[1]]              # (B*(B-1)/2,)
    nonzero = pairwise[pairwise > 0]
    if nonzero.numel() == 0:
        lam = 1.0                                # fallback if no variation
    else:
        med = nonzero.median().item()
        lam = 1.0 / med if med > 0 else 1.0
    lam = max(lam, min_lambda)

    # 3) compute cross‐batch distances
    D_ab = ed_gram(ta, tb)                       # shape (B, B)

    # 4) apply Laplacian kernel
    K = torch.exp(- lam * D_ab)
    return K
  
from dtaidistance import dtw

def seq_to_list(t: torch.Tensor) -> np.ndarray:
    """
    Trim right-padding zeros from a 1D int/float tensor
    and return as a NumPy 1D array.
    """
    arr = t.cpu().numpy()
    # find first zero (padding) if present
    zero_idx = np.where(arr == 0)[0]
    if zero_idx.size > 0:
        arr = arr[:zero_idx[0]]
    return arr
  
  
def seq_to_array(t: torch.Tensor) -> np.ndarray:
    """
    Trim *only* trailing zeros from a 1D tensor and return a
    C‐contiguous NumPy array of dtype float64.

    This uses `np.trim_zeros(..., 'b')` which drops zeros on the right
    but keeps any zeros that occur in the “real” data.
    """
    arr = t.cpu().numpy().astype(np.double)
    # trim only the trailing zeros (padding)
    trimmed = np.trim_zeros(arr, 'b')
    # ensure C‐contiguous
    if not trimmed.flags.c_contiguous:
        trimmed = np.ascontiguousarray(trimmed)
    return trimmed
  
def dtw_gram(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-DTW distances via dtaidistance,
    passing a list of lists directly.
    """
    # 1) build Python lists of int‐lists
    A_list = [seq_to_array(row) for row in A]
    B_list = [seq_to_array(row) for row in B]

    # 2) concatenate into one big list-of-lists
    combined = A_list + B_list

    # 3) let dtaidistance compute the full 2B×2B matrix
    D_all = dtw.distance_matrix_fast(
        combined,
        parallel=True,
        use_c=True,
        max_dist=None,
        use_pruning=False,
        compact=False,
    )
    # 4) slice out rows 0:B, cols B:2B
    Bsz = len(A_list)
    D_ab = D_all[:Bsz, Bsz:]

    # 5) back to torch
    return torch.from_numpy(D_ab).float()


  
def dtw_rbf(A: torch.Tensor, B: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Convert DTW distances into an RBF kernel matrix.
    Returns similarities in (0,1].
    """
    D = dtw_gram(A, B)  # (B, B) distances on CPU
    K = torch.exp(- sigma * D)
    return K
  
  
def gaussian_rbf(A: torch.Tensor,
                 B: torch.Tensor,
                 sigma: float = 1.0) -> torch.Tensor:
    """
    Compute the Gaussian RBF kernel matrix between two sets of attribute vectors.

    Args:
        A: Tensor of shape (n, d), where each row is a d-dim attribute vector.
        B: Tensor of shape (m, d), likewise.
        sigma: bandwidth parameter (float).

    Returns:
        K: Tensor of shape (n, m) with
           K[i, j] = exp( - ||A[i] - B[j]||^2 / (2 * sigma^2) ).
    """
     # 1) Estimate sigma via the median heuristic on A
    with torch.no_grad():
        # pairwise distances within A
        D_AA = torch.cdist(A, A, p=2)             # (n, n)
        # take upper‐triangular off‐diagonals
        idx = torch.triu_indices(D_AA.size(0), D_AA.size(1), offset=1)
        pairwise = D_AA[idx[0], idx[1]]
        sigma = pairwise.median().item()
        # guard against zero sigma
        if sigma == 0.0:
            sigma = 1.0

    # 2) compute cross‐distances and RBF kernel
    D = torch.cdist(A, B, p=2)                     # (n, m)
    K = torch.exp(- D.pow(2) / (2 * sigma * sigma))
    return K.to("cpu")
  
  
  
def plot_boxplots(metrics, output_path='output', output_filename='results.png', max_categories_per_plot: int = 40):
  """Plot boxplots for metrics, splitting very wide category sets into chunks.

  Args:
    metrics (dict): mapping metric_name -> {category_label: [values...]}
    output_path (str): directory to save plots
    output_filename (str): base filename (used for first figure or when unchunked)
    max_categories_per_plot (int): maximum categories shown in one subplot row for a metric.

  Behavior:
    - If a metric has <= max_categories_per_plot categories, behaves like before (one axis).
    - If more, splits that metric's categories into multiple separate FIGURES, each saved with
      suffix _{metric}_part{i}.png (i starting at 1). This prevents huge single-line figures
      that exceed backend size limits.
  """
  os.makedirs(output_path, exist_ok=True)
  metrics = {k: dict(v) for k, v in dict(metrics).items()}
  default_figsize = plt.rcParams.get('figure.figsize')

  # Consistent color mapping
  color_mapping = {
    'TRAIN': '#1f77b4',  # Blue
    'GEN': '#ff7f0e',    # Orange
    'TEST': '#2ca02c',   # Green
    'VAL': '#d62728',    # Red
    'TEST1': '#9467bd',  # Purple
    'TEST2': '#8c564b',  # Brown
    'TEST3': '#e377c2',  # Pink,
  }
  fallback_colors = ['#17becf', '#bcbd22', '#7f7f7f', '#ff9896', '#c5b0d5', '#c49c94']

  # Helper to plot a single metric chunk set
  def _plot_metric(metric_name: str, metric_dict: dict, file_suffix: str = None):
    labels = list(metric_dict.keys())
    data = [metric_dict[l] for l in labels]
    fig, ax = plt.subplots(figsize=(min(3 + 0.25 * len(labels), 0.5 * max_categories_per_plot + 6), default_figsize[1] * 1.2))
    boxplot = ax.boxplot(data, patch_artist=True, medianprops={'color': 'black', 'linewidth': 1})
    colors = []
    fallback_idx = 0
    for label in labels:
      if label in color_mapping:
        colors.append(color_mapping[label])
      else:
        colors.append(fallback_colors[fallback_idx % len(fallback_colors)])
        fallback_idx += 1
    for patch, color in zip(boxplot['boxes'], colors):
      patch.set_facecolor(color)
      patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    title = metric_name.upper()
    if file_suffix:
      title += f" ({file_suffix})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # Decide filename
    if file_suffix is None and len(metrics) == 1 and len(labels) <= max_categories_per_plot:
      # Use provided output filename only if single unchunked metric
      fname = output_filename
    else:
      base, ext = os.path.splitext(output_filename)
      safe_metric = metric_name.replace(' ', '_')
      suffix = file_suffix if file_suffix else 'part1'
      fname = f"{base}_{safe_metric}_{suffix}{ext}"
    plt.savefig(os.path.join(output_path, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)

  # Iterate metrics and chunk if necessary
  for metric_name, metric_values in metrics.items():
    labels = list(metric_values.keys())
    if len(labels) == 0:
      continue
    if len(labels) <= max_categories_per_plot:
      _plot_metric(metric_name, metric_values)
    else:
      # Chunk into parts
      for i in range(0, len(labels), max_categories_per_plot):
        part_labels = labels[i:i + max_categories_per_plot]
        part_dict = {lbl: metric_values[lbl] for lbl in part_labels}
        part_idx = i // max_categories_per_plot + 1
        total_parts = (len(labels) + max_categories_per_plot - 1) // max_categories_per_plot
        file_suffix = f"part{part_idx}of{total_parts}"
        _plot_metric(metric_name, part_dict, file_suffix)