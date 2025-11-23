import sys
import os
import time
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils_cvae import read_log, get_dataset_attributes_info

def add_trace_attr_relative_timestamp_to_first_activity(
    log, trace_key='case:concept:name', timestamp_key='time:timestamp',
    custom_timestamp_key='relative_timestamp_from_start'):
  
  traces = list(log.groupby(trace_key).groups.values())
  log[custom_timestamp_key] = 0.0
  
  # get ts of first activity in the log
  lowest_timestamp = log[timestamp_key].min()

  for t in traces:
    # get ts of first activity in trace
    lowest_timestamp_trace = log.iloc[t][timestamp_key].min()
    # compute diff between first activity in trace and first activity in log
    custom_timestamp = (lowest_timestamp_trace - lowest_timestamp).total_seconds() / 60.0

    log.loc[t, custom_timestamp_key] = custom_timestamp

  return log


def add_relative_timestamp_between_activities(
    log, trace_key='case:concept:name', timestamp_key='time:timestamp',
    custom_timestamp_key='relative_timestamp_from_previous_activity'):
  
  traces = list(log.groupby(trace_key).groups.values())
  log[custom_timestamp_key] = 0.0

  for t in traces:
    for n, a in enumerate(t):
      if n == 0:
        log.loc[a, custom_timestamp_key] = 0.0
        continue

      log.loc[a, custom_timestamp_key] = (log.iloc[t[n]][timestamp_key] - log.iloc[t[n-1]][timestamp_key]).total_seconds() / 60.0

  return log



if __name__ == '__main__':
  DATASET = 'Sepsis Cases - Event Log.xes'
  DATASET_PATH = f'./data/raw/'
  DATASET_NAME = f'{DATASET}'
  DATASET_TRACE_KEY = 'Case ID'
  DATASET_TIMESTAMP_KEY = 'time:timestamp'
  TIMESTAMP_FROM_START_KEY = 'relative_timestamp_from_start'
  TIMESTAMP_FROM_PREV_KEY = 'relative_timestamp_from_previous_activity'

  start_time = time.time()
  
  info = get_dataset_attributes_info(os.path.join(DATASET_PATH, DATASET_NAME))

  df = read_log(os.path.join(DATASET_PATH, DATASET_NAME), verbose=False)
  
  # ── 0. Load raw CSV ──────────────────────────────────────────────────────
  
  df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

  # ── 1. Basic renaming ────────────────────────────────────────────────────
  df = df.rename(columns={
      "case:concept:name": "Case ID",
      "concept:name":       "Activity",
      "org:group":          "Resource"
  })

  # ── 2. Sort for reproducible feature generation ─────────────────────────
  df = df.sort_values(["Case ID", "time:timestamp"]).reset_index(drop=True)

  # ── 3. Variant & variant index ───────────────────────────────────────────
  variant_series = (
      df.groupby("Case ID")["Activity"]
        .transform(lambda s: "▹".join(s))
  )
  df["Variant"] = variant_series
  df["Variant index"] = (
      df.groupby("Variant").cumcount()
  )
  
  df["Variant"] = "Variant " + df["Variant index"].astype(str)

  # ── 4. Temporal features ────────────────────────────────────────────────
  ts = df["time:timestamp"]

  df["timesincemidnight"]   = (ts - ts.dt.normalize()).dt.total_seconds()
  df["month"]               = ts.dt.month
  df["weekday"]             = ts.dt.weekday
  df["hour"]                = ts.dt.hour
  df["timesincecasestart"]  = (
      ts - ts.groupby(df["Case ID"]).transform("first")
  ).dt.total_seconds()
  df["timesincelastevent"]  = (
      ts - ts.groupby(df["Case ID"]).shift(1, fill_value=ts.min())
  ).dt.total_seconds()
  df.loc[df.groupby("Case ID").head(1).index, "timesincelastevent"] = 0
  df["event_nr"] = df.groupby("Case ID").cumcount() + 1

  # ── 5. Open cases (true concurrency) ──────────────────────────────────────
  # 5-a.  First and last timestamp of every trace
  case_start = df.groupby("Case ID")["time:timestamp"].first().sort_values()
  case_end   = df.groupby("Case ID")["time:timestamp"].last() .sort_values()

  starts = case_start.values        # numpy arrays, already sorted
  ends   = case_end.values

  # 5-b.  Walk through the log in chronological order
  df = df.sort_values("time:timestamp").reset_index(drop=True)
  event_ts = df["time:timestamp"].values

  # open cases = cases started ≤ t  −  cases ended  < t
  open_counts = (
      np.searchsorted(starts, event_ts, side="right")   # starts ≤ t
    - np.searchsorted(ends,   event_ts, side="left")    # ends   < t
  )

  df["open_cases"] = open_counts


  # ── 6. Label (example definition) ───────────────────────────────────────
  is_release = df["Activity"].str.startswith("Release")
  is_return  = df["Activity"] == "Return ER"

  # get the last discharge timestamp per case
  last_release_ts = (
      df.loc[is_release]
        .groupby("Case ID")["time:timestamp"]
        .max()
        .rename("discharge_ts")
  )

  # get the first return-ER timestamp *after* that discharge
  first_return_ts = (
      df.loc[is_return]
        .groupby("Case ID")["time:timestamp"]
        .min()
        .rename("return_ts")
  )

  # bring the two together
  readmission = (
      pd.concat([last_release_ts, first_return_ts], axis=1)
        .dropna()                                     # need both timestamps
        .assign(delta=lambda x: (x["return_ts"] - x["discharge_ts"]).dt.days)
  )

  # cases with delta ≤ 28 days → label 1
  cases_with_relapse = set(readmission.loc[readmission["delta"] <= 28].index)

  df["label"] = np.where(df["Case ID"].isin(cases_with_relapse), "relapse", "regular")
  # ── 7. Final column order — exactly as requested ────────────────────────
  wanted_order = [
      "Case ID", "Activity", "Resource", "time:timestamp",
      "Variant", "Variant index",
      "Diagnose", "DiagnosticArtAstrup", "DiagnosticBlood", "DiagnosticECG",
      "DiagnosticIC", "DiagnosticLacticAcid", "DiagnosticLiquor",
      "DiagnosticOther", "DiagnosticSputum", "DiagnosticUrinaryCulture",
      "DiagnosticUrinarySediment", "DiagnosticXthorax",
      "DisfuncOrg", "Hypotensie", "Hypoxie", "InfectionSuspected",
      "Infusion", "Oligurie", "SIRSCritHeartRate", "SIRSCritLeucos",
      "SIRSCritTachypnea", "SIRSCritTemperature", "SIRSCriteria2OrMore",
      "Age", "CRP", "LacticAcid", "Leucocytes",
      "timesincemidnight", "month", "weekday", "hour",
      "timesincelastevent", "timesincecasestart", "event_nr", "open_cases",
      "label"
  ]
  df_final = df[wanted_order]

  # ── 8. Save ─────────────────────────────────────────────────────────────
  df_final.to_csv("sepsis_transformed.csv", index=False)
  print("✓ sepsis_transformed.csv written")

  log = add_trace_attr_relative_timestamp_to_first_activity(
    df_final,
    trace_key="Case ID",
    timestamp_key="time:timestamp",

  )
  log = add_relative_timestamp_between_activities(
    log,
    trace_key="Case ID",
    timestamp_key="time:timestamp",

  )
  
  log = log.sort_values(["Case ID", "time:timestamp"])

  log.to_csv(os.path.join(DATASET_PATH,  f'{DATASET}_pp.csv'), sep=';', index=False)

  end_time = time.time()

  print(f'Execution time: {end_time - start_time} seconds')

  import matplotlib.pyplot as plt
  # Plot histogram of new timestamp
  plt.hist(log[TIMESTAMP_FROM_START_KEY], bins=50)
  plt.xlabel(TIMESTAMP_FROM_START_KEY)
  plt.ylabel('Frequency')
  plt.title(f'Histogram of {TIMESTAMP_FROM_START_KEY}')
  plt.show()

  plt.hist(log[TIMESTAMP_FROM_PREV_KEY], bins=50)
  plt.xlabel(TIMESTAMP_FROM_PREV_KEY)
  plt.ylabel('Frequency')
  plt.title(f'Histogram of {TIMESTAMP_FROM_PREV_KEY}')
  plt.show()