"""Benchmark the frozen M1/M2/M3 files with five F2 metrics and raw evidence."""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from data_pipeline.normalization import DEFAULT_STATS, file_sha256, metadata_path
from training.dataset_io import load_dataset
from training.evaluation import classification_metrics
from training.quantise_m2 import verify_baseline
from training.quantization import inspect_full_int8, int8_classification_metrics
from training.train_model import environment_record, write_json
from optimisation.benchmarking import (
    BACKEND, TIMED_RUNS, WARMUPS, LiteRunner, collect_latency, cpu_probe,
    energy_proxy, evaluate_runner, latency_summary, pareto_names,
)


def read_host_profile(path):
    if not path.is_file():
        raise ValueError("Host profile is missing. First verify the Windows CPU name, logical CPU count and TDP estimate; see README_BENCHMARK.md")
    profile = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cpu_name", "tdp_source", "tdp_basis"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f"Host profile needs a nonempty {key}")
    tdp, count = profile.get("tdp_estimate_w"), profile.get("host_logical_cpus")
    if (isinstance(tdp, bool) or not isinstance(tdp,(int,float)) or not math.isfinite(tdp) or tdp <= 0 or
            not isinstance(count,int) or isinstance(count,bool) or count < 1):
        raise ValueError("Host profile needs positive tdp_estimate_w and integer host_logical_cpus")
    return profile


def read_registered_models(arrays, dataset_meta, root=Path("training/models")):
    indices = arrays["validation_indices"]
    validation_hash = hashlib.sha256(indices.tobytes()).hexdigest()
    p1 = root / "baseline_pointer.json"
    baseline = json.loads(p1.read_text(encoding="utf-8"))
    m1_path = Path(baseline["metrics_file"])
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    verify_baseline(baseline,m1,dataset_meta,indices)
    specs = [{"variant":"M1_FP32", "path":Path(baseline["fp32_model"]), "dtype":np.float32,
              "recorded":m1["fp32_tflite"]}]
    protected = [p1,m1_path,Path(baseline["fp32_model"])]
    for name,pointer_file in [("M2_INT8","m2_pointer.json"),("M3_STRUCTURED_INT8","m3_pointer.json")]:
        pointer_path = root / pointer_file
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        metrics_path = Path(pointer["metrics_file"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("dataset_sha256","normalization_stats_sha256"):
            if pointer.get(key) != dataset_meta[key] or metrics.get(key) != dataset_meta[key]:
                raise ValueError(f"{name} was measured with a different dataset/statistics hash")
        if (pointer.get("validation_indices_sha256") != validation_hash or
                metrics.get("validation_indices_sha256") != validation_hash or
                not metrics.get("integer_only_verified") or
                not metrics.get("int8",{}).get("accuracy_requirement_met")):
            raise ValueError(f"{name} has not passed its checks on this frozen validation split")
        path = Path(pointer["model_file"])
        sha = file_sha256(path)
        if sha != pointer.get("model_sha256") or sha != metrics.get("model_sha256"):
            raise ValueError(f"{name} model file differs from its recorded hash")
        audit = inspect_full_int8(path.read_bytes())
        if name == "M3_STRUCTURED_INT8":
            if metrics.get("architecture") != [6,21,10,3] or metrics.get("parameter_count") != 400:
                raise ValueError("M3 is not the verified compact structured model")
            actual = metrics.get("structured_pruning",{})
            if actual.get("removed_hidden_units") != 17 or actual.get("retained_hidden_widths") != [21,10]:
                raise ValueError("M3 structured-removal evidence is missing")
            shapes = [t["shape"] for t in audit["tensors"] if t["dtype"] == "INT8"]
            if any(shape not in shapes for shape in [[21,6],[10,21],[3,10]]):
                raise ValueError("M3 stored Dense shapes do not match the compact architecture")
        specs.append({"variant":name,"path":path,"dtype":np.int8,"recorded":metrics["int8"]})
        protected.extend([pointer_path,metrics_path,path])
    return specs,protected,validation_hash


def guest_cpu_names():
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return [platform.processor()]
    return sorted({line.split(":",1)[1].strip() for line in path.read_text().splitlines()
                   if line.startswith("model name") and ":" in line})


def affinity_record():
    try:
        process=psutil.Process()
        return {"cpu_affinity":process.cpu_affinity() if hasattr(process,"cpu_affinity") else None}
    except (psutil.Error,AttributeError,NotImplementedError) as error:
        # Optional environment metadata may be unavailable. The separate CPU
        # energy probe still fails explicitly if actual process sampling fails.
        return {"cpu_affinity":None,"cpu_affinity_note":"unavailable: "+type(error).__name__}


def run(args):
    if not math.isfinite(args.cpu_probe_seconds) or not 2 <= args.cpu_probe_seconds <= 30:
        raise ValueError("Use a CPU probe duration between 2 and 30 seconds; default is 5")
    arrays,dataset_meta = load_dataset(args.dataset_dir,args.stats)
    specs,protected,validation_hash = read_registered_models(arrays,dataset_meta,args.models_dir)
    print("Registered files:",flush=True)
    for spec in specs:
        print(f"  {spec['variant']}: {spec['path'].stat().st_size} bytes | {spec['path']}",flush=True)
    print("Backend:",BACKEND,flush=True)
    print("Warm-ups / timed calls per model:",WARMUPS,TIMED_RUNS,flush=True)
    print("Validation rows:",len(arrays["validation_indices"]),flush=True)
    print("Guest CPU name(s):",guest_cpu_names(),flush=True)
    print("Guest logical CPUs visible:",psutil.cpu_count(logical=True),flush=True)
    if args.check_only:
        print("Model hashes, validation split, INT8 graphs and M3 structure: verified",flush=True)
        print("Readiness check only; no latency or energy results were produced",flush=True)
        return 0
    profile = read_host_profile(args.host_profile)
    protected.extend([args.host_profile,args.stats,metadata_path(args.stats),
                      args.dataset_dir/"logibridge_dataset.npz",args.dataset_dir/"dataset_manifest.json",
                      args.dataset_dir/"dataset_windows.csv"])
    hashes = {str(path):file_sha256(path) for path in protected}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = args.output_dir or Path("optimisation/results")/f"benchmark_{stamp}_{uuid.uuid4().hex[:6]}"
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Output folder is not empty; use a new folder")
    folder.mkdir(parents=True,exist_ok=True)
    val_indices = arrays["validation_indices"]
    x_val,y_val = arrays["X_normalized"][val_indices],arrays["y"][val_indices]
    runners = [LiteRunner(spec["variant"],spec["path"].read_bytes(),x_val,spec["dtype"]) for spec in specs]
    print("Timing all models in rotating order...",flush=True)
    raw = collect_latency(runners)
    with (folder/"raw_latency_samples.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=[*raw[0],"dataset_index"])
        writer.writeheader()
        for row in raw:
            writer.writerow({**row,"dataset_index":int(val_indices[row["validation_row"]])})
    rows = []
    details = {}
    predictions = {}
    for runner,spec in zip(runners,specs):
        selected = [row for row in raw if row["variant"] == runner.name]
        summary = latency_summary([row["invoke_ms"] for row in selected])
        api = latency_summary([row["api_roundtrip_ms"] for row in selected])
        probabilities = evaluate_runner(runner)
        predictions[runner.name] = probabilities
        if runner.is_int8:
            accuracy = int8_classification_metrics(y_val,probabilities,runner.output_info["quantization"][0])
        else:
            accuracy = classification_metrics(y_val,probabilities)
        print(f"CPU proxy probe: {runner.name} ({args.cpu_probe_seconds:g} seconds)...",flush=True)
        probe = cpu_probe(runner,args.cpu_probe_seconds)
        power = energy_proxy(profile["tdp_estimate_w"],probe["process_cpu_percent"],
                             profile["host_logical_cpus"],summary["mean_ms"])
        row = {
            "variant":runner.name,"mean_latency_ms":summary["mean_ms"],"p95_latency_ms":summary["p95_ms"],
            "model_size_bytes":spec["path"].stat().st_size,"model_KB_decimal":spec["path"].stat().st_size/1000,
            "accuracy_percent":accuracy["accuracy_percent"],"energy_proxy_mJ":power["energy_proxy_mJ"],
            "critical_recall_percent":accuracy["critical_recall_percent"],"critical_support":accuracy["critical_support"],
            "deployment_metric_eligible":accuracy["accuracy_requirement_met"] and accuracy["critical_recall_requirement_met"],
        }
        rows.append(row)
        details[runner.name] = {"invoke_latency":summary,"api_roundtrip_latency":api,"accuracy":accuracy,
                               "recorded_confusion_matrix_matches":accuracy["confusion_matrix"] == spec["recorded"]["confusion_matrix"],
                               "cpu_probe":probe,"energy_proxy":power,"model_file":str(spec["path"]),
                               "model_sha256":file_sha256(spec["path"]),"input_dtype":runner.input_info["dtype"].__name__,
                               "output_dtype":runner.output_info["dtype"].__name__}
    feasible = [row for row in rows if row["deployment_metric_eligible"]]
    primary_objectives = [("mean_latency_ms","min"),("model_KB_decimal","min")]
    full_objectives = [("mean_latency_ms","min"),("p95_latency_ms","min"),("model_KB_decimal","min"),
                       ("accuracy_percent","max"),("energy_proxy_mJ","min")]
    frontier = pareto_names(feasible,primary_objectives)
    full_frontier = pareto_names(feasible,full_objectives)
    with (folder/"benchmark_table.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (folder/"validation_predictions.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant","dataset_index","true_scenario_label","prediction","p_normal","p_warning","p_critical"])
        for name,values in predictions.items():
            for index,label,probs in zip(val_indices,y_val,values):
                writer.writerow([name,int(index),int(label),int(probs.argmax()),*map(float,probs)])
    write_json(folder/"pareto.json",{
        "feasibility_constraints":"accuracy>88% and Critical recall>95%; measured on the same validation split",
        "chart_objectives":primary_objectives,"chart_frontier":frontier,
        "five_metric_objectives":full_objectives,"five_metric_frontier":full_frontier,
        "comparison":"exact values; no statistical-significance adjustment; close timings need cautious interpretation",
    })
    write_json(folder/"environment.json",{
        **environment_record(),"host_profile":profile,"guest_cpu_names":guest_cpu_names(),
        "guest_logical_cpu_count":psutil.cpu_count(logical=True),"platform":platform.platform(),
        "backend":BACKEND,"interpreter_threads":1,"perf_counter_resolution_s":time.get_clock_info("perf_counter").resolution,
        **affinity_record(),
    })
    from optimisation.plots import save_figures
    save_figures(rows,frontier,folder)
    if {str(path):file_sha256(path) for path in protected} != hashes:
        raise ValueError("Source models, dataset, statistics or host profile changed during the benchmark")
    result = {
        "schema_version":1,"warmup_calls_per_model":WARMUPS,"timed_calls_per_model":TIMED_RUNS,
        "backend":BACKEND,"timing_scope":"Interpreter.invoke only; set_tensor, get_tensor, quantization and preprocessing excluded",
        "api_roundtrip_scope":"diagnostic set_tensor+invoke+get_tensor including timer overhead; pre-encoded inputs; not full sensor-to-alert latency",
        "energy_formula":"P_proxy_W=TDP_W*Process.cpu_percent/(100*Windows_host_logical_cpus); E_proxy_mJ=P_proxy_W*mean_invoke_ms",
        "energy_scope":"CPU/TDP proxy from a separate sustained loop; not electrical energy or a power-meter reading",
        "dataset_sha256":dataset_meta["dataset_sha256"],"normalization_stats_sha256":dataset_meta["normalization_stats_sha256"],
        "validation_indices_sha256":validation_hash,"host_profile":profile,"source_hashes":hashes,
        "label_note":dataset_meta["label_note"],"rows":rows,"details":details,
        "pareto_latency_size":frontier,"pareto_five_metric":full_frontier,"source_files_unchanged":True,
        "all_recorded_confusion_matrices_match":all(value["recorded_confusion_matrix_matches"] for value in details.values()),
    }
    write_json(folder/"benchmark_results.json",result)
    lines=["# F2 benchmark record","",f"Backend: {BACKEND}","",
           "Primary timing: invoke only; ten warm-ups excluded, 200 measured calls per model.","",
           "| Variant | Mean ms | p95 ms | KB (decimal) | Accuracy % | Energy proxy mJ | Critical recall % (support) |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['mean_latency_ms']:.6f} | {row['p95_latency_ms']:.6f} | "
                     f"{row['model_KB_decimal']:.3f} | {row['accuracy_percent']:.2f} | {row['energy_proxy_mJ']:.6f} | "
                     f"{row['critical_recall_percent']:.2f} ({row['critical_support']}) |")
    lines.extend(["",f"Feasible mean-latency/file-size frontier: {', '.join(frontier) or 'none'}.","",
                  f"TDP estimate: {profile['tdp_estimate_w']} W; basis: {profile['tdp_basis']}; source: {profile['tdp_source']}.","",
                  "Energy is a CPU-utilization/TDP proxy, with Python loop overhead and a separate sustained CPU probe. "
                  "It is not a direct electrical measurement. The logical-CPU normalization assumes equal-capacity logical CPUs; "
                  "hybrid cores, SMT, boost, idle power, DVFS and virtualization limit accuracy.","",
                  "These measurements are from the team's laptop/WSL/Docker CPU runtime. They do not measure Raspberry Pi, Hailo, "
                  "Jetson or MCU performance. Kernel latency does not establish the full 90-second sensor-to-alert SLA.","",
                  "A final recommendation must also consider p95, the energy proxy, actual device memory/runtime support, "
                  "and the measured full pipeline. Preserve all raw samples and state the Pareto objectives."])
    (folder/"F2_Benchmark_Record.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    latest = Path("optimisation/results/benchmark_pointer.json")
    latest.parent.mkdir(parents=True,exist_ok=True)
    write_json(latest,{"run_dir":str(folder),"results_file":str(folder/"benchmark_results.json"),
                       "dataset_sha256":dataset_meta["dataset_sha256"],"normalization_stats_sha256":dataset_meta["normalization_stats_sha256"]})
    print("\nVariant | mean_ms | p95_ms | KB_decimal | accuracy_% | energy_proxy_mJ | Critical_% (support)",flush=True)
    for row in rows:
        print(f"{row['variant']} | {row['mean_latency_ms']:.6f} | {row['p95_latency_ms']:.6f} | "
              f"{row['model_KB_decimal']:.3f} | {row['accuracy_percent']:.2f} | {row['energy_proxy_mJ']:.6f} | "
              f"{row['critical_recall_percent']:.2f} ({row['critical_support']})",flush=True)
    print("Feasible Pareto frontier (mean latency / file size):",frontier,flush=True)
    print("Feasible frontier using all five metrics:",full_frontier,flush=True)
    print("Warm-ups excluded / timed calls per model:",WARMUPS,TIMED_RUNS,flush=True)
    print("Recorded confusion matrices match:",result["all_recorded_confusion_matrices_match"],flush=True)
    print("Source files unchanged:",True,flush=True)
    print("Results:",folder/"benchmark_results.json",flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",type=Path,default=Path("training/dataset"))
    parser.add_argument("--stats",type=Path,default=DEFAULT_STATS)
    parser.add_argument("--models-dir",type=Path,default=Path("training/models"))
    parser.add_argument("--host-profile",type=Path,default=Path("hardware/benchmark_host.json"))
    parser.add_argument("--cpu-probe-seconds",type=float,default=5.0)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"Benchmark stopped: {error}",file=sys.stderr,flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
