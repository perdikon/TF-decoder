import os
import shutil
import re
import json
import torch
import pandas as pd
import datetime
import random
from collections import Counter
import pm4py
from src.evaluation_cvae import evaluate_model
from src.preprocess_log import add_relative_timestamp_between_activities, add_trace_attr_relative_timestamp_to_first_activity
from src.utils_cvae import plot_boxplots, save_dict_to_json

import numpy as np
from pathlib import Path
import glob
import seaborn as sns
import matplotlib.pyplot as plt



def reshape_for_boxplot(per_file_metrics):
    """
    Convert {file: {metric: value, ...}, ...}
    into {metric: {category: [value]}, ...}
    """
    from collections import defaultdict, OrderedDict
    
    by_metric = defaultdict(lambda: defaultdict(list))

    for file_name, metric_dict in per_file_metrics.items():
        for metric, val in metric_dict.items():
            # Only plot scalar numeric metrics; skip None and non-numeric (e.g., dicts)
            if val is None:
                continue
            if not isinstance(val, (int, float, np.floating, np.integer)):
                continue
            
            # Determine category based on filename
            if file_name.startswith("gen"):
                category = "GEN"
            elif file_name.startswith("train"):
                category = "TRAIN"
            else:
                category = file_name
            
            by_metric[metric][category].append(val)

    # Convert to ordered dict with TRAIN first, then GEN
    ordered_result = {}
    for metric, categories in by_metric.items():
        ordered_categories = OrderedDict()
        # Add TRAIN first if it exists
        if "TRAIN" in categories:
            ordered_categories["TRAIN"] = categories["TRAIN"]
        # Add GEN second if it exists
        if "GEN" in categories:
            ordered_categories["GEN"] = categories["GEN"]
        # Add any other categories
        for cat, vals in categories.items():
            if cat not in ["TRAIN", "GEN"]:
                ordered_categories[cat] = vals
        
        ordered_result[metric] = ordered_categories

    return ordered_result



class CVAEEvaluationPipeline:
    def __init__(self, model, datamodule, config):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.dataset_info = datamodule.dataset_info
        
    def run_full_evaluation(self, output_path: str, generate = True):
        """Complete evaluation pipeline"""
        results = {}
        
        # Step 1: Use existing generated logs if a path is provided; otherwise generate
        generation_cfg = self.config.get('generation', {})
        existing_dir = generation_cfg.get('existing_dir')
        generated_dir_override = None
        if existing_dir:
            if not os.path.isdir(existing_dir):
                raise ValueError("Provided 'existing_dir' is not a valid directory: " + str(existing_dir))
            # Copy/normalize existing files into output_path/generated as genN.xes/.csv
            gen_dir = os.path.join(output_path, 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            copy_summary = self._populate_generated_from_existing(existing_dir, gen_dir)
            results['generation'] = {
                'source': 'existing',
                'existing_dir': existing_dir,
                'copied': copy_summary,
            }
        else:
            gen_results = self.generate_synthetic_data(output_path)
            results['generation'] = {
                'source': 'generated',
                'details': gen_results
            }
            generated_dir_override = None  # default 'output_path/generated' will be used
        
        # Step 2: After generation, compute latent embeddings for TRAIN and GEN
        
        latent_meta = self.compute_latent_embeddings(output_path)
        results['latent_embeddings'] = latent_meta
        
        
        # Step 3: Compute metrics
        metrics_results = self.compute_evaluation_metrics(output_path)
        
        # Step 3.1: cHSIC(R, T | A) — conditional dependence the baseline cannot model
        chsic_results = self.evaluate_chsic_rt_given_activity(output_path)
        results['chsic'] = chsic_results

        # Step 3.2: Per-file cHSIC aggregates for boxplots; append to metrics_results
        chsic_flat = self.evaluate_chsic_rt_given_activity_metrics_flat(output_path)
        for key, mdict in chsic_flat.items():
            if key not in metrics_results:
                metrics_results[key] = {}
            metrics_results[key].update(mdict)

        # Persist augmented metrics (including cHSIC) back to metrics.json
        try:
            save_dict_to_json(metrics_results, filepath=os.path.join(output_path, 'metrics.json'))
        except Exception as e:
            print(f"[WARNING] Failed to write augmented metrics with cHSIC to metrics.json: {e}")
        
        # # Step 3: Create visualizations
        viz_results = self.create_visualizations(output_path, metrics_results)
        results['visualizations'] = viz_results
        
        return results

    # ------------------------- Latent encoding utilities -------------------------
    def _build_trace_batch(self, traces, schema: str):
        """Convert a list of per-trace DataFrames into tensors expected by the encoder.
        schema: 'gen' for generated CSVs, 'real' for training CSVs.
        Returns: (attrs_dict, acts, ts, ress, cond, case_ids)
        """
        import pandas as pd
        device = next(self.model.parameters()).device
        dm_train = self.datamodule.data_test
        info = self.dataset_info

        # Column names
        if schema == 'gen':
            case_col = 'case:concept:name'
            act_col = 'concept:name'
            res_col = 'org:resource'
            ts_col = 'time:timestamp'
            label_col = 'case:label'
        else:  # 'real'
            case_col = info.trace_key
            act_col = info.activity_key
            res_col = info.resource_key
            ts_col = 'time:timestamp'  # training CSVs use this name in evaluation code
            label_col = info.label_key

        # Prepare attribute spec maps
        trace_specs = info.trace_attributes
        cat_maps = getattr(dm_train, 's2i', {})

        # Numerical ranges
        num_ranges = {}
        for spec in trace_specs:
            if spec.get('type') == 'numerical':
                num_ranges[spec['name']] = (spec.get('min_value', 0.0), spec.get('max_value', 1.0))

        max_len = dm_train.max_trace_length
        pad_act = dm_train.activity2n[dm_train.PADDING_ACTIVITY]
        pad_res = dm_train.resource2n[dm_train.PADDING_RESOURCE]
        eot_act = dm_train.activity2n[dm_train.EOT_ACTIVITY]
        eot_res = dm_train.resource2n[dm_train.EOT_RESOURCE]
        highest_ts = dm_train.highest_ts

        attrs_batch = {spec['name']: [] for spec in trace_specs}
        acts_list, ts_list, ress_list, cond_list, case_ids = [], [], [], [], []

        for g in traces:
            # Case id
            case_id_val = str(g.iloc[0][case_col])
            case_ids.append(case_id_val)

            # Trace attributes from first row
            first = g.iloc[0]
            for spec in trace_specs:
                name = spec['name']
                if spec['type'] == 'categorical':
                    raw = first.get(name)
                    # Ensure consistent type (str) for mapping
                    raw = str(raw) if raw is not None else ''
                    idx = cat_maps.get(name, {}).get(raw, 0)
                    attrs_batch[name].append(torch.tensor(idx, dtype=torch.int64))
                elif spec['type'] == 'numerical':
                    try:
                        raw_f = float(first.get(name))
                    except Exception:
                        raw_f = 0.0
                    mn, mx = num_ranges.get(name, (0.0, 1.0))
                    norm = 0.0 if mx == mn else (raw_f - mn) / (mx - mn)
                    attrs_batch[name].append(torch.tensor(norm, dtype=torch.float32))
                else:
                    raise ValueError(f"Unknown trace attribute type: {spec['type']}")

            # Sequence columns
            act_seq = g[act_col].astype(str).tolist()
            res_seq = g[res_col].astype(str).tolist()

            # Relative timestamps from start of trace in minutes
            if ts_col in g.columns:
                ts_series = pd.to_datetime(g[ts_col])
                base = ts_series.iloc[0]
                rel_secs = (ts_series - base).dt.total_seconds().tolist()
            else:
                rel_secs = [0.0] * len(act_seq)
            rel_norm = [(s / highest_ts) for s in rel_secs]

            # Truncate + EOT + Pad
            act_seq = act_seq[: max_len - 1]
            res_seq = res_seq[: max_len - 1]
            rel_norm = rel_norm[: max_len - 1]
            act_seq.append(dm_train.EOT_ACTIVITY)
            res_seq.append(dm_train.EOT_RESOURCE)
            rel_norm.append(0.0)
            pad_need = max_len - len(act_seq)
            act_seq += [dm_train.PADDING_ACTIVITY] * pad_need
            res_seq += [dm_train.PADDING_RESOURCE] * pad_need
            rel_norm += [0.0] * pad_need

            acts_idx = torch.tensor([dm_train.activity2n.get(a, pad_act) for a in act_seq], dtype=torch.int64)
            ress_idx = torch.tensor([dm_train.resource2n.get(r, pad_res) for r in res_seq], dtype=torch.int64)
            ts_vals = torch.tensor(rel_norm, dtype=torch.float32)

            acts_list.append(acts_idx)
            ress_list.append(ress_idx)
            ts_list.append(ts_vals)

            # Conditional label c
            lbl_raw = first.get(label_col)
            if isinstance(lbl_raw, (list, tuple)):
                # unexpected, take first
                lbl_raw = lbl_raw[0]
            try:
                onehot = dm_train.label2onehot[lbl_raw]
            except Exception:
                # try cast to str for generated logs
                onehot = dm_train.label2onehot.get(str(lbl_raw), None)
                if onehot is None:
                    # fallback to zeros
                    L = len(next(iter(dm_train.label2onehot.values())))
                    onehot = torch.zeros(L, dtype=torch.float32)
            cond_list.append(onehot.to(torch.float32))

        # Stack tensors
        acts = torch.stack(acts_list).to(device)
        ts = torch.stack(ts_list).to(device)
        ress = torch.stack(ress_list).to(device)
        cond = torch.stack(cond_list).to(device)

        # Attributes dict tensors
        attr_tensors = {}
        for spec in trace_specs:
            name = spec['name']
            vals = attrs_batch[name]
            t = torch.stack(vals)
            attr_tensors[name] = t.to(device)

        return attr_tensors, acts, ts, ress, cond, case_ids

    @torch.no_grad()
    def _encode_dataframe(self, df: pd.DataFrame, schema: str, batch_size: int = 256) -> pd.DataFrame:
        """Encode all traces from df into latent mean/logvar."""
        import pandas as pd
        if schema == 'gen':
            case_col = 'case:concept:name'
        else:
            case_col = self.dataset_info.trace_key

        groups = [g for _, g in df.groupby(case_col)]
        rows = []
        self.model.eval()
        device = next(self.model.parameters()).device
        for i in range(0, len(groups), batch_size):
            batch_traces = groups[i:i+batch_size]
            attrs, acts, ts, ress, c, cids = self._build_trace_batch(batch_traces, schema)
            mean, logvar = self.model.encode((attrs, acts, ts, ress), c)
            mean = mean.detach().cpu()
            logvar = logvar.detach().cpu()
            for cid, m_vec, lv_vec in zip(cids, mean, logvar):
                rec = { 'case_id': cid }
                for j in range(m_vec.shape[0]):
                    rec[f'mean_{j}'] = float(m_vec[j])
                for j in range(lv_vec.shape[0]):
                    rec[f'logvar_{j}'] = float(lv_vec[j])
                rows.append(rec)
            # free
            del attrs, acts, ts, ress, c, mean, logvar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return pd.DataFrame(rows)

    def compute_latent_embeddings(self, output_path: str) -> dict:
        """Compute latent embeddings for the real train data and all generated logs.
        Saves two CSVs under output_path/latent_embeddings and returns metadata.
        """
        from src.utils_cvae import read_log
        import pandas as pd
        out_dir = os.path.join(output_path, 'latent_embeddings')
        os.makedirs(out_dir, exist_ok=True)

        # Real train
        train_df = read_log(self.dataset_info.train_path, separator=self.dataset_info.csv_separator, verbose=False)
        lat_train = self._encode_dataframe(train_df, schema='real', batch_size=int(self.config.get('latent', {}).get('batch_size', 256)))
        train_out = os.path.join(out_dir, 'latents_train.csv')
        lat_train.to_csv(train_out, index=False)

        # Generated
        gen_dir = os.path.join(output_path, 'generated')
        gen_csvs = sorted(glob.glob(os.path.join(gen_dir, 'gen*.csv')))
        if gen_csvs:
            lat_gen_parts = []
            for f in gen_csvs:
                df = pd.read_csv(f, sep=';')
                lat = self._encode_dataframe(df, schema='gen', batch_size=int(self.config.get('latent', {}).get('batch_size', 256)))
                lat['source_file'] = os.path.basename(f)
                lat_gen_parts.append(lat)
            lat_gen = pd.concat(lat_gen_parts, ignore_index=True)
        else:
            lat_gen = pd.DataFrame(columns=['case_id'])
        gen_out = os.path.join(out_dir, 'latents_generated.csv')
        lat_gen.to_csv(gen_out, index=False)

        # new_path = "C://Users//nikol//MT-repo//logs//eval//runs//2025-11-08_20-01-18//baseline_cvae_rerun_test//generated"
        # new_gen_csvs = sorted(glob.glob(os.path.join(new_path, 'gen*.csv')))
        # if new_gen_csvs:
        #     lat_gen_parts = []
        #     for f in new_gen_csvs:
        #         df = pd.read_csv(f, sep=';')
        #         lat = self._encode_dataframe(df, schema='gen', batch_size=int(self.config.get('latent', {}).get('batch_size', 256)))
        #         lat['source_file'] = os.path.basename(f)
        #         lat_gen_parts.append(lat)
        #     new_lat_gen = pd.concat(lat_gen_parts, ignore_index=True)
        # else:
        #     new_lat_gen = pd.DataFrame(columns=['case_id'])
        # new_gen_out = os.path.join(out_dir, 'latents_generated2.csv')
        # new_lat_gen.to_csv(new_gen_out, index=False)
        return {
            'train': {
                'path': train_out,
                'rows': int(len(lat_train)),
                'latent_dim': int(sum(1 for c in lat_train.columns if c.startswith('mean_'))),
            },
            'generated': {
                'path': gen_out,
                'rows': int(len(lat_gen)),
                'latent_dim': int(sum(1 for c in lat_gen.columns if c.startswith('mean_'))) if len(lat_gen) else None,
                'num_files': len(gen_csvs),
            }
        }

    def _populate_generated_from_existing(self, existing_dir: str, gen_dir: str) -> dict:
        """
        Simplest possible copy: recursively copy the entire existing_dir tree into gen_dir
        (creating or merging) and return a summary. We do NOT rename files; downstream
        code that relies on gen*.csv patterns may need adjustment if source files are
        differently named.

        Uses shutil.copytree(..., dirs_exist_ok=True) (Python 3.8+) to preserve structure.
        Filters summary to .csv/.xes files only for brevity.
        """
        # Perform recursive copy (merge if target exists)
        shutil.copytree(existing_dir, gen_dir, dirs_exist_ok=True)

        # Gather copied .csv/.xes file basenames for summary
        collected = []
        for root, _, files in os.walk(gen_dir):
            for f in files:
                if f.lower().endswith(('.csv', '.xes')):
                    rel_path = os.path.relpath(os.path.join(root, f), gen_dir)
                    collected.append(rel_path)

        return {
            'source_dir': existing_dir,
            'dest_dir': gen_dir,
            'num_csv_xes': len(collected),
            'files': sorted(collected),
            'mode': 'copytree'
        }
    
    def generate_synthetic_data(self, output_path: str):
        # Move generate_dataset logic here
        """Generate synthetic process traces using the trained CVAE model"""
        # Get configuration values
        generation_cfg = self.config.get('generation', {})
        logs_to_generate = generation_cfg.get('logs_to_generate', 10)
        output_name = generation_cfg.get('output_name', 'gen')
        sample_proportions = generation_cfg.get('sample_proportions', True)
        
        return self._generate_dataset(
            output_path=output_path,
            output_name=output_name,
            logs_to_generate=logs_to_generate,
            sample_proportions=sample_proportions
        )
    
    def _generate_dataset(self, output_path='output', output_name='gen', logs_to_generate=1, sample_proportions=True):
        """
        Generates synthetic process traces using the trained CVAE model.
        """
        gen_path = os.path.join(output_path, 'generated')
        os.makedirs(gen_path, exist_ok=True)

        test_set = self.datamodule.data_test
        len_test = len(test_set)
        
        if sample_proportions:
            # Count label proportions in test set
            test_labels = test_set.y
            label_values = [tuple(label.tolist()) for label in test_labels]
            
            label_counts = Counter(label_values)
            label_proportions = {test_set.onehot2label[label]: count/len_test for label, count in label_counts.items()}
            
            print(f"Test set label distribution:")
            for label, proportion in label_proportions.items():
                count = label_counts[tuple(test_set.label2onehot[label].tolist())]
                print(f"  {label}: {count} ({proportion:.2%})")
        else:
            # Use equal proportions or single label
            labels = list(test_set.label2onehot.keys())
            label_proportions = {label: 1.0/len(labels) for label in labels}

        generation_results = []

        # BEAM SEARCH CONFIG (NEW): read optional beam settings from self.config['generation']
        beam_cfg = self.config.get('generation', {}).get('beam_search', {})
        use_beam = bool(beam_cfg.get('enabled', True))  # set True to activate beam search
        beam_size = int(beam_cfg.get('beam_size', 4))    # number of beams
        beam_max_len = beam_cfg.get('max_length', None)  # optional override for trace length
        if use_beam:
            print(f"[BeamSearch] Using beam search: size={beam_size}, max_len={beam_max_len or self.datamodule.dataset_info.max_trace_length}")

        for log_id in range(logs_to_generate):
            if sample_proportions:
                # Sample labels according to test set proportions
                sampled_labels = random.choices(
                    list(label_proportions.keys()), 
                    weights=list(label_proportions.values()), 
                    k=len_test
                )
            else:
                # Use first available label for all cases
                first_label = list(test_set.label2onehot.keys())[0]
                sampled_labels = [first_label] * len_test
            
            # Create conditional encodings for each sampled label
            c_list = []
            for label in sampled_labels:
                c_list.append(test_set.label2onehot[label])
            
            # Generation in chunks to limit GPU memory
            device = next(self.model.parameters()).device
            labels_tensor = torch.stack(c_list)  # (len_test, cond_dim) on CPU for now
            gen_cfg = self.config.get('generation', {})
            gen_bs = int(gen_cfg.get('batch_size', 256))  # tune based on GPU memory

            new_data = []
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len_test, gen_bs):
                    end = min(start + gen_bs, len_test)
                    c = labels_tensor[start:end].to(device, non_blocking=True)
                    z = torch.randn(end - start, self.model.z_dim, device=device)

                    out = self.model.decode(
                        z, c,
                        use_beam_search=use_beam,
                        beam_size=beam_size,
                        beam_max_len=beam_max_len
                    )

                    if not isinstance(out, tuple):
                        raise TypeError("model.decode expected to return a tuple")
                    if len(out) == 5:
                        attrs, acts, ts, ress, _ = out
                    elif len(out) == 4:
                        attrs, acts, ts, ress = out
                    else:
                        raise ValueError(f"model.decode returned {len(out)} elements (expected 4 or 5)")

                    ts = ts.clamp_min(0)

                    # Process attributes per batch
                    trace_attrs = {}
                    for attr in self.dataset_info.trace_attributes:
                        if attr['type'] == 'categorical':
                            attrs[attr['name']] = torch.argmax(attrs[attr['name']], dim=1)
                        elif attr['type'] == 'numerical':
                            attrs[attr['name']] = attr['min_value'] + (attrs[attr['name']] * (attr['max_value'] - attr['min_value']))
                        else:
                            raise Exception(f'Unknown trace attribute type: {attr["type"]}')
                        trace_attrs[attr['name']] = attrs[attr['name']]

                    # Process activities and resources
                    acts_idx = torch.argmax(acts, dim=2)
                    ress_idx = torch.argmax(ress[:, :, :-1], dim=2)

                    # Generate case data for this slice (use global indices)
                    for i in range(end - start):
                        global_i = start + i
                        case_label = sampled_labels[global_i]
                        # Pass batch-local index for tensor selection, but keep a unique global case id
                        case_data = self._generate_case_data(
                            i, acts_idx[i], ts[i], ress_idx[i], trace_attrs, test_set, case_label,
                            case_id_override=f'GEN{global_i}'
                        )
                        new_data.extend(case_data)

                    # Free GPU memory for this chunk
                    del z, c, out, acts, ts, ress, attrs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Create DataFrame and save
            df = pd.DataFrame(new_data)
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
            for col in ['case:concept:name', 'case:label', 'concept:name', 'org:resource']:
                df[col] = df[col].astype(str)

            df = add_trace_attr_relative_timestamp_to_first_activity(df)
            df = add_relative_timestamp_between_activities(df)

            path_xes = os.path.join(gen_path, f'{output_name}{log_id}.xes')
            path_csv = path_xes.replace('.xes', '.csv')
            pm4py.write_xes(df, path_xes, case_id_key='case:concept:name')
            df.to_csv(path_csv, sep=";")
            
            generation_results.append({
                'log_id': log_id,
                'path_xes': path_xes,
                'path_csv': path_csv,
                'num_cases': len(df['case:concept:name'].unique()),
                'num_events': len(df)
            })
            
            print(f'Generated log saved at {path_xes}')

        return generation_results

    def _generate_case_data(self, idx, activities, timestamps, resources, trace_attrs, test_set, case_label="regular", case_id_override: str = None):
        """Builds event records for a single case"""
        case_attrs = {k: v[idx] for k, v in trace_attrs.items()}
        start_datetime = pd.to_datetime(test_set.log['time:timestamp']).min()
        start_offset = datetime.timedelta(minutes=case_attrs['relative_timestamp_from_start'].item())
        current_time = start_datetime + start_offset
        case_id = case_id_override if case_id_override is not None else f'GEN{idx}'

        events = []
        for j, (act, res) in enumerate(zip(activities, resources)):
            activity_name = test_set.n2activity[act.item()]
            resource_name = test_set.n2resource[res.item()]

            if activity_name == 'EOT':
                break
            
            cat_attrs = {k: test_set.i2s[k][v.item()] for k, v in case_attrs.items() if k in test_set.i2s}
            num_attrs = {k: v.item() for k, v in case_attrs.items() if k not in test_set.i2s}

            time_value = timestamps[j].item() * test_set.highest_ts
            # Clamp to reasonable range (max ~10 years in minutes)
            MAX_MINUTES = 60 * 24 * 365 * 2  # 10 years in minutes
            clamped_value = min(time_value, MAX_MINUTES)
            
            delta = datetime.timedelta(minutes=clamped_value)
            current_time += delta

            events.append({
                'case:concept:name': case_id,
                'concept:name': activity_name,
                'org:resource': resource_name,
                'time:timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                **cat_attrs,
                **num_attrs,
                'case:label': case_label
            })

        return events
    
    def compute_evaluation_metrics(self, output_path: str):
        """Compute evaluation metrics"""
        evaluation_cfg = self.config.get('evaluation', {})
        metrics = evaluation_cfg.get('metrics', ["cfld", "2gram", "red", "ctd", "cwd", "conformance"])
        
        metrics_results = evaluate_model(output_path, self.dataset_info, metrics=metrics)
        return metrics_results

    
    def _collect_activity_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a DataFrame with (Activity, Position) where Position is the
        normalized index (0..1) of the event inside its trace.
        """
        # Ensure schema normalization
        df_std = self._prep_eval_df(df)
        records = []
        # Group by case to compute within-trace positions
        for case_id, g in df_std.groupby('case:concept:name'):
            g = g.sort_values('rel_time', kind='stable')
            L = len(g)
            if L == 1:
                # Single event trace -> position 0
                records.append({
                    'Activity': g['concept:name'].iloc[0],
                    'Position': 0.0
                })
                continue
            for idx, (_, row) in enumerate(g.iterrows()):
                pos = idx / (L - 1)
                records.append({
                    'Activity': row['concept:name'],
                    'Position': float(pos)
                })
        return pd.DataFrame(records)


    def create_visualizations(self, output_path: str, metrics: dict):
        """Create visualization plots"""
        plot_cfg = self.config.get('plots', {})
        
        if not plot_cfg:
            return {}
            
        # Load metrics if they're in file format
        if isinstance(metrics, str):
            with open(metrics, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        
        reshaped = reshape_for_boxplot(metrics)
        
        visualization_results = {}
        
        # Create different plot groups (classic metrics)
        plot_groups = {
            'control_flow': plot_cfg.get('control_flow_metrics', ['cfld', '2gram']),
            'temporal': plot_cfg.get('temporal_metrics', ['ctd', 'red']),
            'distance': plot_cfg.get('distance_metrics', ['cwd']),
            'conformance': plot_cfg.get('conformance_metrics', ['conformance'])
        }
        
        for group_name, group_metrics in plot_groups.items():
            group_data = {metric: reshaped[metric] for metric in group_metrics if metric in reshaped}
            if group_data:
                filename = f'{group_name.title()} Metrics.png'
                plot_boxplots(group_data, output_path=output_path, output_filename=filename)
                visualization_results[group_name] = {
                    'filename': filename,
                    'metrics': list(group_data.keys())
                }
        
        # HSIC metric plots (only if present in reshaped)
        # 1) Uniform aggregate
        if 'chsic_uniform' in reshaped:
            hsic_uniform_data = {'chsic_uniform': reshaped['chsic_uniform']}
            filename = 'HSIC_Uniform.png'
            plot_boxplots(hsic_uniform_data, output_path=output_path, output_filename=filename)
            visualization_results['chsic_uniform'] = {
                'filename': filename,
                'metrics': ['chsic_uniform']
            }
        # 2) Frequency-weighted aggregate
        if 'chsic_freq_weighted' in reshaped:
            hsic_w_data = {'chsic_freq_weighted': reshaped['chsic_freq_weighted']}
            filename = 'HSIC_Freq_Weighted.png'
            plot_boxplots(hsic_w_data, output_path=output_path, output_filename=filename)
            visualization_results['chsic_freq_weighted'] = {
                'filename': filename,
                'metrics': ['chsic_freq_weighted']
            }

        # 3) Per-activity HSIC distribution across logs
        # Build a structure: {activity: {TRAIN: [..], GEN: [..]}}
        # from metrics entries containing 'chsic_per_activity'
        activity_collect = {}
        for file_key, mvals in metrics.items():
            if 'chsic_per_activity' not in mvals:
                continue
            # determine category
            if file_key.startswith('train'):
                category = 'TRAIN'
            elif file_key.startswith('gen'):
                category = 'GEN'
            else:
                # Could also check for .xes naming
                if file_key.startswith('gen') or file_key.startswith('train'):
                    category = 'GEN'
                else:
                    category = 'OTHER'
            per_act = mvals['chsic_per_activity']
            for act, val in per_act.items():
                if not isinstance(val, (int, float, np.floating)):
                    continue
                activity_collect.setdefault(act, {}).setdefault(category, []).append(val)

        if activity_collect:
            # Optionally limit number of activities by frequency of appearance for readability
            # For now plot all; build metrics dict of form { 'chsic_act_<act>': {TRAIN:[..], GEN:[..]} }
            per_activity_box = {}
            for act, cat_map in activity_collect.items():
                metric_name = f"chsic_act_{act}"
                per_activity_box[metric_name] = cat_map
            if per_activity_box:
                filename = 'HSIC_Per_Activity.png'
                plot_boxplots(per_activity_box, output_path=output_path, output_filename=filename)
                visualization_results['chsic_per_activity'] = {
                    'filename': filename,
                    'metrics': list(per_activity_box.keys())
                }


        return visualization_results
    
    
    @staticmethod
    def _ensure_relative_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a numeric 'rel_time' column (minutes since first event in the case).
        If the DF already has a suitable column created earlier in the pipeline
        (e.g., 'case:rel_time' or similar), reuse it; otherwise compute it.
        """
        out = df.copy()
        # Try reusing columns created by your preprocessing, otherwise compute.
        candidates = [
            'relative_timestamp_from_first_activity',
            'rel_time',
            'relative_time'
        ]
        for c in candidates:
            if c in out.columns:
                out['rel_time'] = pd.to_numeric(out[c], errors='coerce')
                break
        else:
            # Compute from raw timestamps per case
            if 'time:timestamp' not in out.columns:
                raise ValueError("Missing 'time:timestamp' column for cHSIC.")
            out['time:timestamp'] = pd.to_datetime(out['time:timestamp'])
            first_times = out.groupby('case:concept:name')['time:timestamp'].transform('min')
            dt = (out['time:timestamp'] - first_times).dt.total_seconds() / 60.0
            out['rel_time'] = dt

        # Safety: clamp NaNs/infs
        out['rel_time'] = out['rel_time'].fillna(0.0).astype(float)
        return out

    @staticmethod
    def _prep_eval_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only the columns we need and standardize types.
        Accepts either schema:
          - XES-like: 'case:concept:name', 'concept:name', 'org:resource'
          - Alternative: 'Case ID', 'Activity', 'Resource'
        """
        df = df.copy()

        # Detect schema and unify column names
        if {'case:concept:name', 'concept:name', 'org:resource'}.issubset(df.columns):
            case_col, act_col, res_col = 'case:concept:name', 'concept:name', 'org:resource'
        elif {'Case ID', 'Activity', 'Resource'}.issubset(df.columns):
            case_col, act_col, res_col = 'Case ID', 'Activity', 'Resource'
            # Map to standardized names in a temporary view
            df = df.rename(columns={
                'Case ID': 'case:concept:name',
                'Activity': 'concept:name',
                'Resource': 'org:resource',
            })
            case_col, act_col, res_col = 'case:concept:name', 'concept:name', 'org:resource'
        else:
            raise ValueError(
                "Missing required columns: expected either "
                "['case:concept:name','concept:name','org:resource'] or "
                "['Case ID','Activity','Resource']"
            )

        if 'time:timestamp' not in df.columns:
            raise ValueError("Missing 'time:timestamp' column for cHSIC prep.")

        # Standardize dtypes and compute relative time
        df['concept:name'] = df['concept:name'].astype(str)
        df['org:resource'] = df['org:resource'].astype(str)
        df = CVAEEvaluationPipeline._ensure_relative_time(df)
        return df[['case:concept:name', 'concept:name', 'org:resource', 'rel_time']]

    @staticmethod
    def _rbf_multi(x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale RBF. x: (n,1), y: (m,1).
        """
        # Pairwise squared distances
        d2 = torch.cdist(x, y, p=2).pow(2)  # (n, m)
        K = 0.0
        for s in sigmas:
            K = K + torch.exp(-d2 / (2.0 * (s * s + 1e-12)))
        return K

    @staticmethod
    def _delta_eq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Categorical δ-kernel: 1 if equal else 0.
        x: (n,), y: (m,)
        """
        return (x[:, None] == y[None, :]).float()

    @staticmethod
    def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> float:
        """
        Unbiased HSIC estimator (requires n>=4). Returns scalar float.
        """
        n = K.shape[0]
        if n < 4:
            # fallback to biased for tiny samples
            return CVAEEvaluationPipeline._hsic_biased(K, L)
        K2 = K.clone()
        L2 = L.clone()
        K2.fill_diagonal_(0.0)
        L2.fill_diagonal_(0.0)
        H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
        KH = K2 @ H
        LH = L2 @ H
        val = (KH * LH).sum() / (n * (n - 3))
        return float(val.item())

    @staticmethod
    def _hsic_biased(K: torch.Tensor, L: torch.Tensor) -> float:
        """
        Biased HSIC estimator; stable for very small n.
        """
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
        Kc = H @ K @ H
        Lc = H @ L @ H
        val = (Kc * Lc).sum() / (n * n)
        return float(val.item())

    @staticmethod
    def _median_bandwidth(v: torch.Tensor) -> float:
        """
        Median heuristic per activity for the RBF on time (1D case) with
        memory-safe approximation for large n.
        v: (n,1)
        Strategy:
          - If n <= exact_limit: compute full pairwise distance median (original behavior).
          - Else if n is large: sample up to sample_size points (without replacement) and
            compute median of their pairwise distances.
          - Additionally provide a very cheap fallback using MAD if even sampling would be large.
        This avoids allocating an n x n matrix that can exceed memory (O(n^2)).
        """
        n = v.shape[0]
        if n < 2:
            return 1.0  # arbitrary fallback

        exact_limit = 4000          # 4000^2 ~ 16M entries (~64MB float32) still acceptable
        sample_size = 1024           # size for approximate median if n is large
        mad_fallback_limit = 20000   # beyond this, prefer MAD over large sampling distances

        # Exact computation for moderate n
        if n <= exact_limit:
            d = torch.cdist(v, v, p=2)
            # exclude zeros (diagonal)
            mask = d > 0.0
            if torch.any(mask):
                med = torch.median(d[mask])
            else:
                med = torch.tensor(1.0, device=v.device)
            return float(max(med.item(), 1e-3))

        # Very large n: cheap robust estimate via MAD (Median Absolute Deviation)
        if n > mad_fallback_limit:
            v_flat = v.view(-1)
            median_v = v_flat.median()
            mad = (v_flat - median_v).abs().median()
            # For 1D normal data: median pairwise distance ≈ sqrt(2) * MAD
            approx = float((mad * 1.41421356237).item())
            return max(approx, 1e-3)

        # Sampling path
        with torch.no_grad():
            idx = torch.randperm(n, device=v.device)[:sample_size]
            vs = v[idx]
            d_sample = torch.cdist(vs, vs, p=2)
            mask = d_sample > 0.0
            if torch.any(mask):
                med = torch.median(d_sample[mask])
            else:
                med = torch.tensor(1.0, device=v.device)
        return float(max(med.item(), 1e-3))

    def _chsic_profile_one_df(self, df: pd.DataFrame) -> dict:
        """
        Compute per-activity HSIC(R, T) within the given DF.
        Returns: {activity: hsic_value}
        """
        device = torch.device('cpu')  # keep simple/portable
        out = {}
        # Allow user to set a max points attribute externally; default 4000.
        max_points = getattr(self, 'hsic_max_points', 4000)
        # If extremely large, allow a secondary cap for gen logs (can be set externally too)
        max_points_gen = getattr(self, 'hsic_max_points_gen', max_points)

        is_generated_context = getattr(self, '_chsic_is_generated', False)
        active_cap = max_points_gen if is_generated_context else max_points

        for a, grp in df.groupby('concept:name'):
            # resource categorical → integer codes
            r_codes, _ = pd.factorize(grp['org:resource'], sort=True)
            r = torch.from_numpy(r_codes.astype(np.int64)).to(device)
            # time numeric
            t = torch.from_numpy(grp['rel_time'].values.astype(np.float32)).to(device).view(-1, 1)
            n = t.shape[0]
            if n < 3:
                continue

            # Subsample for memory safety if group is very large
            if n > active_cap:
                # Random subset without replacement
                idx = torch.randperm(n)[:active_cap]
                t = t[idx]
                r = r[idx]
                n = active_cap
                # (Optional) mark that subsampling occurred for this activity
                # Store a negative HSIC sentinel? Instead we keep actual approximate HSIC; add meta elsewhere if needed.

            # Build kernels
            # Multi-scale RBF with 3 scales: {median, 2*median, 4*median}
            s = self._median_bandwidth(t)
            sigmas = torch.tensor([s, 2*s, 4*s], dtype=torch.float32, device=device)
            Kt = self._rbf_multi(t, t, sigmas)  # (n,n)
            Kr = self._delta_eq(r, r)           # (n,n)

            # Unbiased if feasible, else biased
            hsic = self._hsic_unbiased(Kt, Kr) if n >= 4 else self._hsic_biased(Kt, Kr)
            out[str(a)] = hsic
        return out

    @staticmethod
    def _aggregate_profiles(profile: dict, counts: dict) -> dict:
        """
        Compute uniform and frequency-weighted aggregates from per-activity HSICs.
        """
        if not profile:
            return {'uniform': None, 'freq_weighted': None}
        # Uniform
        uniform = float(np.mean(list(profile.values())))
        # Frequency-weighted
        total = 0.0
        acc = 0.0
        for a, v in profile.items():
            w = counts.get(a, 0)
            acc += w * v
            total += w
        freq = float(acc / total) if total > 0 else None
        return {'uniform': uniform, 'freq_weighted': freq}

    def evaluate_chsic_rt_given_activity(self, output_path: str) -> dict:
        """
        Compute HSIC(R, T) within each activity, for:
          - TRAIN/REAL: use datamodule's test log (available in your code)
          - GEN: stack generated CSVs under output_path/generated/*.csv
        Return a dict ready to store in final results.
        """
        # --- Real/Test from datamodule ---
        test_log = self.datamodule.data_test.log
        if isinstance(test_log, pd.DataFrame):
            df_real = test_log.copy()
        else:
            # If your datamodule keeps a dict-like log, turn to DataFrame
            df_real = pd.DataFrame(test_log)
        
        
        df_real = self._prep_eval_df(df_real)
        df_real.to_csv("C://Users//nikol//MT-repo//data//emergency_ORT//emergency_ORT_TEST2_maybe.csv", index=False, sep=';')
        real_profile = self._chsic_profile_one_df(df_real)
        real_counts = df_real['concept:name'].value_counts().to_dict()
        real_aggr = self._aggregate_profiles(real_profile, real_counts)

        # --- Generated from disk ---
        gen_dir = os.path.join(output_path, 'generated')
        gen_csvs = sorted(glob.glob(os.path.join(gen_dir, '*.csv')))
        if not gen_csvs:
            # No generated CSVs found; return only real
            return {
                'per_activity': {
                    'TRAIN': real_profile,
                    'GEN': {}
                },
                'aggregate': {
                    'TRAIN': real_aggr,
                    'GEN': {'uniform': None, 'freq_weighted': None}
                },
                'meta': {
                    'num_real_events': len(df_real),
                    'num_gen_events': 0,
                    'gen_dir': gen_dir
                }
            }

        dfs = []
        for f in gen_csvs:
            df_tmp = pd.read_csv(f, sep=';')
            dfs.append(df_tmp)
        df_gen = pd.concat(dfs, ignore_index=True)
        df_gen = self._prep_eval_df(df_gen)
        gen_profile = self._chsic_profile_one_df(df_gen)
        gen_counts = df_gen['concept:name'].value_counts().to_dict()
        gen_aggr = self._aggregate_profiles(gen_profile, gen_counts)

        # Sort top activities for convenience (by real HSIC)
        top_real = sorted(real_profile.items(), key=lambda kv: kv[1], reverse=True)[:10]
        top_gen  = [(a, gen_profile.get(a, None)) for (a, _) in top_real]

        return {
            'per_activity': {
                'TRAIN': real_profile,
                'GEN': gen_profile
            },
            'aggregate': {
                'TRAIN': real_aggr,
                'GEN': gen_aggr
            },
            'top10_by_real': {
                'activities': [a for a, _ in top_real],
                'real': [v for _, v in top_real],
                'gen': [gen_profile.get(a, None) for a, _ in top_real]
            },
            'meta': {
                'num_real_events': int(len(df_real)),
                'num_gen_events': int(len(df_gen)),
                'gen_dir': gen_dir,
                'num_gen_files': len(gen_csvs)
            }
        }

    def evaluate_chsic_rt_given_activity_metrics_flat(self, output_path: str) -> dict:
        """Compute per-file HSIC aggregates suitable for boxplots and metrics merge.
        Returns a dict mapping file keys to metric dicts, matching evaluate_model keys where possible.
        For generated logs we map CSV base names to XES-style keys (e.g., gen0.xes) to align with metrics_results.
        """
        out = {}

        gen_dir = os.path.join(output_path, 'generated')
        # Train splits created by evaluate_model live here as CSVs
        train_csvs = sorted(glob.glob(os.path.join(gen_dir, 'train_SPLIT_*.csv')))
        for f in train_csvs:
            try:
                df = pd.read_csv(f, sep=';')
                df = self._prep_eval_df(df)
                profile = self._chsic_profile_one_df(df)
                counts = df['concept:name'].value_counts().to_dict()
                aggr = self._aggregate_profiles(profile, counts)
                key = os.path.splitext(os.path.basename(f))[0]  # e.g., train_SPLIT_1
                out[key] = {
                    'chsic_uniform': aggr['uniform'],
                    'chsic_freq_weighted': aggr['freq_weighted'],
                    'chsic_per_activity': profile,
                }
            except Exception:
                continue

        # Generated CSVs; convert to XES-style keys to match metrics_results (genN.xes)
        gen_csvs = sorted(glob.glob(os.path.join(gen_dir, 'gen*.csv')))
        for f in gen_csvs:
            try:
                df = pd.read_csv(f, sep=';')
                df = self._prep_eval_df(df)
                profile = self._chsic_profile_one_df(df)
                counts = df['concept:name'].value_counts().to_dict()
                aggr = self._aggregate_profiles(profile, counts)
                base = os.path.splitext(os.path.basename(f))[0]
                key = f"{base}.xes"  # align with evaluation_cvae keys
                out[key] = {
                    'chsic_uniform': aggr['uniform'],
                    'chsic_freq_weighted': aggr['freq_weighted'],
                    'chsic_per_activity': profile,
                }
            except Exception:
                continue

        return out