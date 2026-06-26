import subprocess
import sys
import argparse

def run_command(command):
    print(f"===========================================================")
    print(f"Running: {' '.join(command)}")
    print(f"===========================================================")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
        sys.exit(result.returncode)
    print("Success!\n")

def main():
    parser = argparse.ArgumentParser(description="Main script to run FIDeL pipeline")
    parser.add_argument("--task_name", type=str, required=True, help="Task name for computing threshold scores (e.g. domotic_setTheTable_anomaly)")
    parser.add_argument("--labels_dir", type=str, default="../Labels", help="Directory for labels")
    parser.add_argument("--score_dir", type=str, default="./results/score", help="Directory containing score results")
    parser.add_argument("--safe_labels_dir", type=str, default=None, help="Directory containing safety ground truth labels (optional)")
    
    # Allow passing Hydra overrides directly to the underlying scripts
    args, unknown = parser.parse_known_args()

    # 1. Run store_memory_data.py
    run_command([sys.executable, "store_memory_data.py"] + unknown)

    # 2. Run eval.py
    run_command([sys.executable, "eval.py"] + unknown)

    # 3. Run compute_threshold_score.py
    eval_task_name = args.task_name
    if not eval_task_name.endswith("_anomaly"):
        eval_task_name = f"{eval_task_name}_anomaly"

    from hydra import compose, initialize
    try:
        with initialize(version_base="1.3.2", config_path="cfgs"):
            cfg = compose(config_name="config", overrides=unknown)
        
        cmd_plot = [
            sys.executable, "plot/compute_threshold_score.py",
            "--task_name", eval_task_name,
            "--labels_dir", args.labels_dir,
            "--score_dir", args.score_dir,
            "--encoder", str(cfg.encoder),
            "--ad_type", str(cfg.anomaly_detection.name),
            "--threshold_type", str(cfg.threshold.name)
        ]
        if "distance_type" in cfg.anomaly_detection:
            cmd_plot.extend(["--distance_type", str(cfg.anomaly_detection.distance_type)])
    except Exception as e:
        print(f"Failed to load hydra config, falling back to default plotting arguments. Error: {e}")
        cmd_plot = [
            sys.executable, "plot/compute_threshold_score.py",
            "--task_name", eval_task_name,
            "--labels_dir", args.labels_dir,
            "--score_dir", args.score_dir
        ]

    if args.safe_labels_dir:
        cmd_plot.extend(["--safe_labels_dir", args.safe_labels_dir])
    
    run_command(cmd_plot)

if __name__ == "__main__":
    main()
