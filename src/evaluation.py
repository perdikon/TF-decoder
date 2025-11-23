import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
import itertools
import csv
from collections import defaultdict, OrderedDict
from pathlib import Path

from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
from pathlib import Path
import editdistance
import seaborn as sns

from typing import Sequence, Tuple, Dict, Any, Mapping, List, Optional
from datetime import datetime  

from data.sepsis_datamodule import SepsisDataModule, SepsisDataset


from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import json
import math
import statistics as stats



# ---------------------------------------------------------------------------
# Low‑level helpers
# ---------------------------------------------------------------------------


def _length_stats(seqs: Sequence[Sequence]) -> Tuple[float, float, int]:
    """Return *mean*, *std* and *max* of the sequence lengths."""
    lengths = [len(s) for s in seqs]
    return (
        stats.fmean(lengths) if lengths else math.nan,
        stats.pstdev(lengths) if len(lengths) > 1 else 0.0,
        max(lengths, default=0),
    )


def _freq_dict(seqs: Sequence[Sequence]) -> Counter[int]:
    """Frequency distribution of lengths → Counter({length: count})."""
    return Counter(len(s) for s in seqs)


def _act_freq(seqs: Sequence[Sequence]) -> Counter[int]:
    """Counter mapping *activity id* → *occurrence count* (flattened)."""
    return Counter(a for trace in seqs for a in trace)


def _write_json(data: Dict[str, Any], file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=2))


def _show_or_close(show: bool) -> None:
    if show:
        plt.show()
    plt.close()
    
    
def _norm_edit(seq_a: Sequence, seq_b: Sequence) -> float:
    """Normalized Levenshtein distance ∈ [0, 1]."""
    denom = len(seq_a) + len(seq_b) or 1  # avoid zero‑div when both empty
    return editdistance.eval(seq_a, seq_b) / denom


def _sequence_variance(seqs: Sequence[Sequence]) -> float:
    """Average pair‑wise normalized edit‑distance.

    The formula matches the legacy implementation:

        sum_{i,j} d(i,j) / (2 · N²)

    where *d* is the normalized distance and the sum iterates over
    *ordered* pairs (hence the 2). Self‑distances are zero anyway.
    """
    n = len(seqs)
    if n == 0:
        return 0.0

    total = 0.0
    for i, j in itertools.product(range(n), repeat=2):
        total += _norm_edit(seqs[i], seqs[j])
    return total / (2 * n * n)



def write(item, save_path):
    with open(save_path + 'dif_log.txt', 'a') as filehandle:
        filehandle.write('%s\n' % item)

    
def save_activity_report(
    gen_seqs: Sequence[Sequence[int]],
    aut_seqs: Sequence[Sequence[int]],
    vocab_size: int,
    output_dir:  Path,
    activity2idx: Dict[str, int] ,
    *,
    show: bool = False,
) -> Dict[str, Any]:
    """Compare *activity-type* distributions between two datasets.

    Parameters
    ----------
    vocab_size
        Total number of distinct activity IDs (``0 … vocab_size‑1``). This
        ensures the report includes not‑appearing activities with 0%.
    """

    out = Path(output_dir)

    # Raw counts ------------------------------------------------------------
    gen_counts = _act_freq(gen_seqs)
    aut_counts = _act_freq(aut_seqs)

    # Relative frequencies for every possible activity ID -------------------
    n_gen = max(sum(gen_counts.values()), 1)
    n_aut = max(sum(aut_counts.values()), 1)

    rel_gen = np.zeros(vocab_size)
    rel_aut = np.zeros(vocab_size)
    for a, c in gen_counts.items():
        if activity2idx[a] < vocab_size:
            rel_gen[activity2idx[a]] = c / n_gen
    for a, c in aut_counts.items():
        if activity2idx[a] < vocab_size:
            rel_aut[activity2idx[a]] = c / n_aut

    # L1 distance (sum of absolute diffs) -----------------------------------
    distance = float(np.abs(rel_aut - rel_gen).sum())

    summary = {
        "difference": distance,
        "authentic": rel_aut.tolist(),
        "synthetic": rel_gen.tolist(),
    }

    _write_json(summary, out / "activity_stats.json")
    print("Activity stats →", out / "activity_stats.json")

    # Bar chart -------------------------------------------------------------
    bar_width = 0.3
    rel_aut = rel_aut[1:]
    rel_gen = rel_gen[1:]
    x = np.arange(vocab_size-1)

    plt.figure(figsize=(max(10, vocab_size * 0.5), 6))
    plt.rcParams.update({"font.size": 14})

    plt.bar(x - bar_width / 2, rel_aut, width=bar_width, label="Authentic", alpha=0.6)
    plt.bar(x + bar_width / 2, rel_gen, width=bar_width, label="Synthetic", alpha=0.6, color="tab:red")

    plt.title("Activity Frequency")
    plt.xlabel("Activity Type")
    plt.ylabel("Relative frequency")
    idx2activity: dict[int, str] = {v: k for k, v in activity2idx.items()}
    x_list = x.tolist()
    x_ticks = [idx2activity[i+ 1] for i in x_list]
    plt.xticks(x, x_ticks, rotation=90)
    plt.ylim(0, max(rel_aut.max(), rel_gen.max()) * 1.1 + 1e-6)
    plt.legend()
    plt.tight_layout()

    fig_path = out / "activity_distribution.png"
    plt.savefig(fig_path, dpi=300)
    _show_or_close(show)

    print("Activity histogram →", fig_path)
    return summary



def save_length_report(
    gen_seqs: Sequence[Sequence],
    aut_seqs: Sequence[Sequence],
    output_dir: Path,
    *,
    show: bool = False,
) -> dict:
    """Compare *generated* and *authentic* sequence collections.

    Parameters
    ----------
    gen_seqs, aut_seqs
        Iterables whose elements support ``len()`` - typically *lists*
        of traces.
    output_dir
        Where the report (``.json``) and *PNG* will be stored. The
        directory is created if necessary.
    show
        If *True* call :pyfunc:`plt.show()` after saving the figure.

    Returns
    -------
    dict
        Raw numbers so the caller can keep working with them.
    """
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    
    # ------------------------------------------------------------------
    # 1) Summary statistics
    # ------------------------------------------------------------------
    
    gen_mean, gen_std, gen_max = _length_stats(gen_seqs)
    aut_mean, aut_std, aut_max = _length_stats(aut_seqs)
    seq_diff = abs(aut_mean - gen_mean) + abs(aut_std - gen_std)
    
    summary = {
        "authentic": {"mean": aut_mean, "std": aut_std, "max": aut_max},
        "synthetic": {"mean": gen_mean, "std": gen_std, "max": gen_max},
        "difference": seq_diff,
    }
    
        # Write tiny *JSON* file instead of ad‑hoc txt lines – easier to reuse
    (out / "length_stats.json").write_text(json.dumps(summary, indent=2))

    # Pretty print to console ------------------------------------------------
    print("Sequence length summary →", out / "length_stats.json")
    print(
        f"synthetic mean/std = {gen_mean:.2f}/{gen_std:.2f} | "
        f"authentic mean/std = {aut_mean:.2f}/{aut_std:.2f} | "
        f"abs‑diff = {seq_diff:.2f}"
    )
    
    
    # ------------------------------------------------------------------
    # 2) Histogram figure (inlined – no helper)
    # ------------------------------------------------------------------
    bar_width = 0.9

    gen_freq = _freq_dict(gen_seqs)
    aut_freq = _freq_dict(aut_seqs)
    max_len = max(gen_freq.keys() | aut_freq.keys(), default=0)

    lengths = list(range(max_len + 1))
    df = pd.DataFrame(
        {
            "length": lengths,
            "authentic": [aut_freq.get(l, 0) / max(len(aut_seqs), 1) for l in lengths],
            "synthetic": [gen_freq.get(l, 0) / max(len(gen_seqs), 1) for l in lengths],
        }
    )

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 14})

    x = np.arange(len(df))
    plt.bar(x - bar_width / 2, df["authentic"], width=bar_width, label="Authentic", alpha=0.6)
    plt.bar(x + bar_width / 2, df["synthetic"], width=bar_width, label="Synthetic", alpha=0.6, color="tab:red")

    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence length (# events)")
    plt.ylabel("Relative frequency")
    plt.xticks(ticks=x[:: max(1, len(x)//20)], rotation=45)
    plt.ylim(0, df[["authentic", "synthetic"]].to_numpy().max() * 1.1)
    plt.legend()
    plt.tight_layout()

    fig_path = out / "length_distribution.png"
    plt.savefig(fig_path, dpi=300)
    if show:
        plt.show()
    plt.close()

    print("Histogram saved →", fig_path)


def save_variance_report(
    gen_seqs: Sequence[Sequence],
    aut_seqs: Sequence[Sequence],
    output_dir: Path,
) -> Dict[str, Any]:
    """Compare the *sequence variance* (average normalized edit‑distance)."""

    out = Path(output_dir)

    gen_var = _sequence_variance(gen_seqs)
    aut_var = _sequence_variance(aut_seqs)
    diff = abs(aut_var - gen_var)

    summary = {
        "authentic": aut_var,
        "synthetic": gen_var,
        "difference": diff,
    }

    _write_json(summary, out / "variance_stats.json")
    print("Variance stats →", out / "variance_stats.json")
    print(
        f"authentic variance = {aut_var:.4f} | "
        f"synthetic variance = {gen_var:.4f} | "
        f"abs‑diff = {diff:.4f}"
    )
    return summary
    

def _activity_time_arrays(
    seqs: Sequence[Sequence[int]],
    deltas: Sequence[Sequence[float]],
    vocab_size: int,
    activity2idx: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and 90‑percentile arrays of size *vocab_size*."""
    data = {}
    for labels, inc in zip(seqs, deltas):
        pos = np.cumsum(inc)
        for a, t in zip(labels, pos):
            data.setdefault(a, []).append(float(t))
    
    means = np.full(vocab_size, -1.0)
    q90   = np.full(vocab_size, -1.0)
    for a, positions in data.items():
        if a < vocab_size and positions:
            arr = np.asarray(positions)
            means[a] = arr.mean()
            q90[a]   = np.quantile(arr, 0.90)
    return means, q90




def _plot_violin(
    seqs: Sequence[Sequence[int]],
    deltas: Sequence[Sequence[float]],
    vocab: Mapping[int, str],
    filename: Path,
) -> None:
    """Horizontal violin plot with one colour per activity row."""

    # ── 1. Build a normalised (activity, position) table ────────────────────
    recs = []
    for labels, inc in zip(seqs, deltas):
        pos = np.cumsum(inc)
        if pos[-1] > 0:
            pos = pos / pos[-1]          # scale each trace to [0, 1]
        recs.extend(zip(labels, pos))

    df = pd.DataFrame(recs, columns=["Activity", "Position"])
    df = df[df["Activity"] != 0]         # drop the PAD/“0” activity

    # order of rows on the y-axis (and colours)
    order = [k for k in vocab.keys() if k != 0]

    # one distinct colour per activity code
    palette = sns.color_palette("husl", len(order))      # change if you prefer
    palette = dict(zip(order, palette))                  # map code → colour

    # ── 2. Main violin plot ─────────────────────────────────────────────────
    plt.figure(figsize=(15, max(25, 0.4 * len(order))))
    plt.rcParams.update({"font.size": 18})

    ax = sns.violinplot(
        data=df,
        x="Position",
        y="Activity",
        hue="Activity",          
        hue_order=order,         # keep colours aligned with rows
        palette=palette,
        density_norm="width",
        cut=0,
        orient="h",
        inner="box",
        dodge=False,             # no side-by-side split
        legend=False,            # silence legend creation
    )

    # x-axis
    ax.set_xlabel("Position", fontsize=23)
    ax.set_xlim(0, 1)

    # y-axis: lock ticks, then relabel with vocab names
    yticks = np.arange(len(order))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.set_ylabel("Activity", fontsize=20)
    ax.set_yticklabels([vocab[idx] for idx in order], fontsize=18)

    # ── 3. Secondary y-axis with counts ─────────────────────────────────────
    counts = df["Activity"].value_counts()
    ax2 = ax.twinx()
    ax2.yaxis.set_major_locator(FixedLocator(yticks))
    ax2.set_ylim(ax.get_ylim())
    ax2.set_ylabel("Count", rotation=270, labelpad=25, fontsize=23)
    ax2.set_yticklabels([str(counts.get(code, 0)) for code in order])
    ax2.tick_params(axis="y", length=0)

    # ── 4. Save & close ─
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Violin plot saved →", filename)




def _cumtime_mse(
    aut_deltas: Sequence[Sequence[float]],
    gen_deltas: Sequence[Sequence[float]],
) -> float:
    """Mean‑squared error between cumulative timestamps per trace."""

    def _pad(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = max(len(a), len(b))
        return np.pad(a, (0, m - len(a))), np.pad(b, (0, m - len(b)))

    mse = []
    for a, b in zip(aut_deltas, gen_deltas):
        a_cum, b_cum = np.cumsum(a), np.cumsum(b)
        a_p, b_p = _pad(a_cum, b_cum)
        mse.append(np.mean((a_p - b_p) ** 2))
    return float(np.mean(mse)) if mse else 0.0




def save_temporal_report(
    aut_seqs: Sequence[Sequence[int]],
    aut_deltas: Sequence[Sequence[float]],
    gen_seqs: Sequence[Sequence[int]],
    gen_deltas: Sequence[Sequence[float]],
    vocab: Mapping[int, str],
    output_dir: Path,
    *,
    show: bool = False,
) -> Dict[str, Any]:
    """Compare per‑activity timing + overall cumulative timestamp MSE."""

    out = Path(output_dir)
    vocab_size = len(vocab)
    

    aut_mean, aut_q90 = _activity_time_arrays(aut_seqs, aut_deltas, vocab_size)
    gen_mean, gen_q90 = _activity_time_arrays(gen_seqs, gen_deltas, vocab_size)

    mean_diff = float(np.abs(aut_mean - gen_mean)[aut_mean >= 0].sum())
    q90_diff  = float(np.abs(aut_q90 - gen_q90)[aut_q90 >= 0].sum())
    mse       = _cumtime_mse(aut_deltas, gen_deltas)

    summary = {
        "mean_difference": mean_diff,
        "q90_difference": q90_diff,
        "cumtime_mse": mse,
    }
    _write_json(summary, out / "temporal_stats.json")

    # Plots ---------------------------------------------------------------
    _plot_violin(aut_seqs, aut_deltas, vocab, out / "timestamps_violin_authentic.png")
    _plot_violin(gen_seqs, gen_deltas, vocab, out / "timestamps_violin_synthetic.png")

    return summary



if __name__ == '__main__':
    
            
    
    xes_file = "data/raw/Sepsis Cases - Event Log.xes"
    dataset_real = SepsisDataset(file = xes_file, extension=".xes")
    sequences_real = []
    timestamps_real = []
    for i in range(len(dataset_real)):
        sample = dataset_real[i]
        activities = sample['activities']

        sequences_real.append(activities)
        timestamps_real.append(sample['timestamp_diff'].tolist())
        
    csv_file = "output_1999.csv"
    
    save_dir = Path("./")     # must exist
    
    activity2idx = dataset_real.activity2idx
    
    dataset_fake = SepsisDataset(file = csv_file, extension=".csv", real_dataset=dataset_real)
    sequences_fake = []
    timestamps_fake = []
    for i in range(len(dataset_fake)):
        sample = dataset_fake[i]
        activities = sample['activities']
        activities = activities[:activities.index("<PAD>")]
        sequences_fake.append(activities)
        
        timestamps_fake.append(sample['timestamp_diff'].tolist())
        
    test = 1
    
   
    

    # ------------------------------------------------------------------
    # 2) DataFrame ➜ integer traces
    # ------------------------------------------------------------------


    # Parameters you already used
    
    # save_length_report(
    #         gen_seqs = sequences_fake,          
    #         aut_seqs = sequences_real,          
    #         output_dir = str(save_dir) + "/", 
    # )
    
    # save_activity_report(
    #         gen_seqs = sequences_fake,          
    #         aut_seqs = sequences_real,          
    #         vocab_size = 17,
    #         output_dir = str(save_dir) + "/", 
    #         activity2idx = dataset_real.activity2idx
    # )
    # save_variance_report(
    #         gen_seqs = sequences_fake,          
    #         aut_seqs = sequences_real,          
    #         output_dir = str(save_dir) + "/", 
    # )
    
    
    aut_seqs2 = [
        [activity2idx[a] for a in seq] for seq in sequences_real
    ]
    gen_seqs2 = [
        [activity2idx[a] for a in seq] for seq in sequences_fake if len(seq) > 0
    ]
    
    idx2activity: dict[int, str] = {v: k for k, v in activity2idx.items()}

    
    save_temporal_report(
        aut_seqs= aut_seqs2,
        aut_deltas= timestamps_real,
        gen_seqs= gen_seqs2,   
        gen_deltas= timestamps_fake,
        vocab= idx2activity,
        output_dir= str(save_dir) + "/",
    )
    
    