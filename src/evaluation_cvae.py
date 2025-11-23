from conformance import discover_declare_model, conformance_checking
import os
import pandas as pd
from collections import defaultdict

from utils_cvae import save_dict_to_json, read_log
from log_distance_measures.config import (
    EventLogIDs,
    AbsoluteTimestampType,
    discretize_to_hour,
)
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import (
    absolute_event_distribution_distance,
)
from log_distance_measures.case_arrival_distribution import (
    case_arrival_distribution_distance,
)
from log_distance_measures.circadian_event_distribution import (
    circadian_event_distribution_distance,
)
from log_distance_measures.relative_event_distribution import (
    relative_event_distribution_distance,
)
from log_distance_measures.work_in_progress import work_in_progress_distance
from log_distance_measures.cycle_time_distribution import (
    cycle_time_distribution_distance,
)
from log_distance_measures.circadian_workforce_distribution import (
    circadian_workforce_distribution_distance,
)


def split_training_into_n_temporal_chunks(
    train_log_path,
    log_name,
    output_path,
    n=3,
    csv_sep=";",
    case_id_key="Case ID",
    timestamp_key="time:timestamp",
):
    import os
    from math import ceil

    assert n > 0, "Number of splits must be greater than 0"

    log = read_log(train_log_path, separator=csv_sep)
    cases = list(log[case_id_key].unique())

    cases = list(log[case_id_key].unique())
    cases_by_start_time = {
        case: log[log[case_id_key] == case][timestamp_key].min() for case in cases
    }
    cases_by_start_time = sorted(cases_by_start_time.items(), key=lambda item: item[1])

    chunk_size = ceil(len(cases_by_start_time) / n)
    paths = []
    for i in range(n):
        split_cases = cases_by_start_time[i * chunk_size : (i + 1) * chunk_size]
        split_case_ids = [case[0] for case in split_cases]
        split_df = log[log[case_id_key].isin(split_case_ids)]

        split_name = f"{log_name}_SPLIT_{i+1}"
        split_path = os.path.abspath(os.path.join(output_path, f"{split_name}.csv"))
        paths.append(split_path)
        split_df.to_csv(split_path, sep=csv_sep, index=False)
        print(f"[Split {i+1}] {len(split_cases)} cases → {split_path}")

    return paths


def evaluate_model(
    output_path, dataset_info, metrics=["cfld", "2gram", "red", "ctd", "cwd"]
):
    gen_log_dir = os.path.join(output_path, "generated")
    gen_xes_files = [f for f in os.listdir(gen_log_dir) if f.endswith(".xes") and f.startswith("gen")]

    train_paths = split_training_into_n_temporal_chunks(
        train_log_path=dataset_info.train_path,
        log_name="train",
        output_path=gen_log_dir,
        n=3,
        csv_sep=dataset_info.csv_separator,
    )
    
   
      
    if "conformance" in metrics:
      declare_model_path = discover_declare_model(
          log_path=dataset_info.full_path,
          log_csv_separator=dataset_info.csv_separator,
          case_key=dataset_info.trace_key,
          activity_key=dataset_info.activity_key,
          timestamp_key="time:timestamp",
          output_path=os.path.join(output_path, "conformance"),
          consider_vacuity=False,
          min_support=0.9,
          itemsets_support=0.9,
          max_declare_cardinality=2,
          filter_log_by=None,
      )
    
    
    
    metrics_save = defaultdict(dict)
    
    ### Compute metrics for training logs
    for train_path in train_paths:
        print(f"Evaluating {train_path}")
        train_name = os.path.basename(train_path).replace(".csv", "")
        # compute conformance
        if "conformance" in metrics:
          conformance = compute_conformance(
              log_path=train_path,
              declare_model_path=declare_model_path,
              output_path=os.path.join(output_path, "conformance"),
              output_name=os.path.basename(train_path).replace(".csv", ""),
              filter_log_by=None,
              consider_vacuity=False,
              activity_key=dataset_info.activity_key,
              case_key=dataset_info.trace_key,
          )

          metrics_save[train_name]["conformance"] = conformance
        
        #compute metrics
        for measure in metrics:
          if measure != "conformance":
              value = compute_log_distance_measure(
                  dataset_info.test_path,
                  train_path,
                  measure=measure,
                  method=None,
                  filter_log_by=None,
                  cwd_convert_resources_to_roles=False,  # this enables comparison between resources and not roles!
                  gen_log_trace_key=dataset_info.trace_key,
                  gen_log_activity_key=dataset_info.activity_key,
                  gen_resource_key=dataset_info.resource_key,
              )
              metrics_save[train_name][measure] = value
        
        
    ### Compute metrics for generated logs
    for gen_csv in gen_xes_files:
        gen_log_path = os.path.join(gen_log_dir, gen_csv)
        print(f"Evaluating {gen_csv}")
        if "conformance" in metrics:
          # compute conformance
          conformance = compute_conformance(
              log_path=gen_log_path,
              declare_model_path=declare_model_path,
              output_path=os.path.join(output_path, "conformance"),
              output_name=gen_csv.replace(".csv", ""),
              filter_log_by=None,
              consider_vacuity=False,
          )

          metrics_save[gen_csv]["conformance"] = conformance

        
        for measure in metrics:
            if measure != "conformance":
              value = compute_log_distance_measure(
                  dataset_info.test_path,
                  gen_log_path,
                  measure=measure,
                  method=None,
                  filter_log_by=None,
                  cwd_convert_resources_to_roles=False,  # this enables comparison between resources and not roles!
                  gen_log_trace_key="case:concept:name",
                  gen_log_activity_key="concept:name",
                  gen_resource_key="org:resource",
              )
              metrics_save[gen_csv][measure] = value

    save_dict_to_json(metrics_save, filepath=os.path.join(output_path, f"metrics.json"))
    return metrics_save


def compute_conformance(
    log_path,
    declare_model_path,
    output_path,
    output_name,
    filter_log_by=None,
    consider_vacuity=False,
    case_key="case:concept:name",
    activity_key="concept:name",
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    conformance_checking_result = conformance_checking(
        log_path,
        declare_model_path,
        consider_vacuity=consider_vacuity,
        filter_log_by=filter_log_by,
        log_csv_separator=";",
        case_key=case_key,
        activity_key=activity_key,
    )
    conformance_checking_result.to_csv(
        os.path.join(output_path, f"conformance_checking_result_{output_name}.csv")
    )
    conformance_per_trace = conformance_checking_result.mean(axis=1)
    conformance_per_trace.to_csv(
        os.path.join(
            output_path, f"conformance_checking_per_trace_result_{output_name}.csv"
        )
    )

    return conformance_per_trace.mean()


def compute_log_distance_measure(
    original_log_path,
    generated_log_path,
    measure="cfld",
    method=None,
    filter_log_by=None,
    cwd_resource_to_role_mapping_file=None,
    cwd_convert_resources_to_roles=True,
    gen_log_trace_key="case:concept:name",
    gen_log_activity_key="concept:name",
    gen_log_timestamp_key="time:timestamp",
    gen_resource_key="org:resource",
):
    original_log = read_log(original_log_path, separator=";", verbose=False)
    generated_log = read_log(generated_log_path, separator=";", verbose=False)

    # if provided, filter log by provided label
    if filter_log_by:
        original_log = original_log[original_log["label"] == filter_log_by]
        generated_log = generated_log[generated_log["label"] == filter_log_by]

    # if provided, load the resource-->role mapping csv file
    if cwd_resource_to_role_mapping_file:
        resource_role_mapping = pd.read_csv(cwd_resource_to_role_mapping_file, sep=",")

    original_log_ids = EventLogIDs(
        case="Case ID",
        activity="Activity",
        start_time="time:timestamp",
        end_time="time:timestamp",
        resource="Resource",
    )
    generated_log_ids = EventLogIDs(
        case=gen_log_trace_key,
        activity=gen_log_activity_key,
        start_time=gen_log_timestamp_key,
        end_time=gen_log_timestamp_key,
        resource=gen_resource_key,
    )

    generated_log[gen_log_timestamp_key] = generated_log[
        gen_log_timestamp_key
    ].dt.tz_localize(None)

    if measure == "cfld":
        original_log_traces = original_log["Case ID"].unique().tolist()
        generated_log_traces = generated_log[gen_log_trace_key].unique().tolist()
        # assert len(original_log_traces) == len(generated_log_traces)

        # if logs are not of the same size, drop cases to get same size
        if len(original_log_traces) > len(generated_log_traces):
            print(
                f"WARNING: Dropping cases from original log ({len(original_log_traces)}) to match the size of the generated log ({len(generated_log_traces)})"
            )
            num_cases_to_drop = len(original_log_traces) - len(generated_log_traces)
            for _ in range(num_cases_to_drop):
                original_log_traces.pop()
            original_log = original_log[
                original_log["Case ID"].isin(original_log_traces)
            ]
        elif len(generated_log_traces) > len(original_log_traces):
            print(
                f"WARNING: Dropping cases from generated log ({len(generated_log_traces)}) to match the size of the original log ({len(original_log_traces)})"
            )
            num_cases_to_drop = len(generated_log_traces) - len(original_log_traces)
            for _ in range(num_cases_to_drop):
                generated_log_traces.pop()
            generated_log = generated_log[
                generated_log[gen_log_trace_key].isin(generated_log_traces)
            ]

        return control_flow_log_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            parallel=False,
        )

    if "ngram" in measure:
        ngram_n = int(measure.split("_")[-1])

        return n_gram_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            n=ngram_n,
            normalize=True,
        )
        
    if measure == "2gram":

        return n_gram_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            n=2,
            normalize=True,
        )

    if measure == "aed":
        return absolute_event_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            discretize_type=AbsoluteTimestampType.START,
            discretize_event=discretize_to_hour,
        )

    if measure == "cad":
        return case_arrival_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            discretize_event=discretize_to_hour,
        )

    if measure == "ced":
        return circadian_event_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            discretize_type=AbsoluteTimestampType.START,
        )

    if measure == "red":
        return relative_event_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            discretize_type=AbsoluteTimestampType.START,
            discretize_event=discretize_to_hour,
        )

    if measure == "wip":
        return work_in_progress_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            window_size=pd.Timedelta(hours=1),
        )

    if measure == "ctd":
        return cycle_time_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
            bin_size=pd.Timedelta(hours=1),
        )

    if measure == "cwd":
        # ensure resource column are of the same type
        original_log["Resource"] = original_log["Resource"].astype(str)
        if method in ["LSTM_1", "LSTM_2"]:
            generated_log["role"] = generated_log["role"].astype(str)
        else:
            generated_log[gen_resource_key] = generated_log[gen_resource_key].astype(
                str
            )

        # convert original_log resources to roles, if needed
        if cwd_convert_resources_to_roles:
            original_log = original_log.merge(
                resource_role_mapping, how="left", left_on="Resource", right_on="user"
            )
            original_log = original_log.drop(columns=["Resource", "user"])
            original_log = original_log.rename(columns={"role": "Resource"})

            if method in ["LSTM_1", "LSTM_2"]:
                # if log generated by LSTM, just rename column
                generated_log = generated_log.rename(columns={"role": gen_resource_key})
            else:
                # otherwise map resources to roles
                generated_log = generated_log.merge(
                    resource_role_mapping,
                    how="left",
                    left_on=gen_resource_key,
                    right_on="user",
                )
                generated_log = generated_log.drop(columns=[gen_resource_key, "user"])
                generated_log = generated_log.rename(columns={"role": gen_resource_key})

        print(
            "[WARNING] Computing custom CWD metric. Change code if you want to use the default one."
        )
        return circadian_workforce_distribution_distance(
            original_log,
            original_log_ids,
            generated_log,
            generated_log_ids,
        )
