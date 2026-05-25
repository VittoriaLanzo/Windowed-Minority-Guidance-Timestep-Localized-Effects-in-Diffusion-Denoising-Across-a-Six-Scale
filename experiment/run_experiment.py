import argparse, csv, json, os, subprocess, sys, time
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(REPO_DIR, "data", "experiment_runs.csv")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
RUNTIME_LOG = os.path.join(REPO_DIR, "logs", "runtime.log")

CSV_FIELDS = [
    "run_id", "condition", "guidance_window", "t_start", "t_end",
    "guidance_scale", "seed", "n_generated",
    "classifier_mean_confidence", "classifier_mean_loss",
    "runtime_seconds", "artifact_path", "timestamp",
]


def log(msg):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(RUNTIME_LOG), exist_ok=True)
    with open(RUNTIME_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_existing_run_ids():
    if not os.path.exists(DATA_CSV):
        return set()
    with open(DATA_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        return {row["run_id"] for row in reader}


def write_csv_row(row):
    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    write_header = not os.path.exists(DATA_CSV)
    with open(DATA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def find_npz(result_dir):
    if not os.path.isdir(result_dir):
        return None
    for fname in os.listdir(result_dir):
        if fname.endswith(".npz"):
            return os.path.join(result_dir, fname)
    return None


def extract_metrics(npz_path):
    script = os.path.join(REPO_DIR, "scripts", "extract_metrics.py")
    result = subprocess.run(
        [sys.executable, script,
         "--npz_path", npz_path,
         "--classifier_path", "ckpt/mc_lsun.pt",
         "--f_extractor_path", "ckpt/256x256_classifier.pt",
         "--class_id", "99"],
        capture_output=True, text=True, cwd=REPO_DIR
    )
    if result.returncode != 0:
        raise RuntimeError(f"extract_metrics failed:\n{result.stderr}")
    return json.loads(result.stdout.strip())


def run_sampler(run_id, condition, guidance_window, guidance_scale,
                t_start, t_end, seed, num_samples, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "stdout.log")
    err_path = os.path.join(result_dir, "stderr.log")
    rel_result_dir = os.path.relpath(result_dir, REPO_DIR)

    cmd = [
        sys.executable, "windowed_classifier_sample.py",
        "--attention_resolutions", "32,16,8",
        "--class_cond", "False",
        "--diffusion_steps", "1000",
        "--dropout", "0.1",
        "--image_size", "256",
        "--learn_sigma", "True",
        "--noise_schedule", "linear",
        "--num_channels", "256",
        "--num_head_channels", "64",
        "--num_res_blocks", "2",
        "--resblock_updown", "True",
        "--use_fp16", "False",
        "--use_scale_shift_norm", "True",
        "--latent_size", "8",
        "--in_channels", "512",
        "--out_channels", "100",
        "--classifier_attention_resolutions", "8",
        "--classifier_depth", "2",
        "--classifier_width", "128",
        "--classifier_pool", "attention",
        "--classifier_resblock_updown", "False",
        "--classifier_use_scale_shift_norm", "True",
        "--classifier_use_fp16", "False",
        "--batch_size", "1",
        "--num_samples", str(num_samples),
        "--timestep_respacing", "250",
        "--classifier_scale", str(guidance_scale),
        "--use_manual_class", "True",
        "--manual_class_id", "99",
        "--seed", str(seed),
        "--t_start", str(t_start),
        "--t_end", str(t_end),
        "--model_path", "ckpt/lsun_bedroom.pt",
        "--classifier_path", "ckpt/mc_lsun.pt",
        "--f_extractor_path", "ckpt/256x256_classifier.pt",
    ]

    env = os.environ.copy()
    env["OPENAI_LOGDIR"] = rel_result_dir

    log(f"[{run_id}] window={guidance_window} t=[{t_start},{t_end}) scale={guidance_scale} seed={seed} n={num_samples}")
    t0 = time.time()
    with open(log_path, "w") as out_f, open(err_path, "w") as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, cwd=REPO_DIR, env=env)
    runtime = time.time() - t0

    if proc.returncode != 0:
        with open(err_path) as ef:
            err_tail = ef.read()[-1500:]
        raise RuntimeError(f"Sampler exit {proc.returncode}:\n{err_tail}")

    npz = find_npz(result_dir)
    if npz is None:
        raise RuntimeError(f"Sampler completed but no .npz in {result_dir}")

    log(f"[{run_id}] Done in {runtime:.1f}s -> {npz}")
    return npz, runtime


def execute_run(run_spec, existing_ids, dry_run=False):
    run_id = run_spec["run_id"]
    if run_id in existing_ids:
        log(f"[{run_id}] Already in CSV - skipping.")
        return

    condition       = run_spec["condition"]
    guidance_window = run_spec["guidance_window"]
    guidance_scale  = run_spec["guidance_scale"]
    t_start         = run_spec["t_start"]
    t_end           = run_spec["t_end"]
    seed            = run_spec["seed"]
    num_samples     = run_spec.get("num_samples", 1)
    result_dir      = os.path.join(RESULTS_DIR, run_id)

    if dry_run:
        log(f"[DRY-RUN] {run_id}: window={guidance_window} t=[{t_start},{t_end}) seed={seed}")
        return

    pre_npz = find_npz(result_dir)
    if pre_npz:
        log(f"[{run_id}] Pre-existing .npz found - extracting metrics only.")
        npz_path, runtime = pre_npz, None
    else:
        npz_path, runtime = run_sampler(
            run_id, condition, guidance_window, guidance_scale,
            t_start, t_end, seed, num_samples, result_dir
        )

    log(f"[{run_id}] Extracting metrics...")
    metrics = extract_metrics(npz_path)

    row = {
        "run_id": run_id, "condition": condition,
        "guidance_window": guidance_window, "t_start": t_start, "t_end": t_end,
        "guidance_scale": guidance_scale, "seed": seed,
        "n_generated": metrics["n"],
        "classifier_mean_confidence": metrics["classifier_mean_confidence"],
        "classifier_mean_loss": metrics["classifier_mean_loss"],
        "runtime_seconds": runtime if runtime is not None else "",
        "artifact_path": npz_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    required = ["run_id", "condition", "guidance_window", "t_start", "t_end",
                "guidance_scale", "seed", "n_generated",
                "classifier_mean_confidence", "classifier_mean_loss"]
    for field in required:
        if row[field] == "" or row[field] is None:
            raise ValueError(f"Incomplete row for {run_id}: field '{field}' missing.")

    write_csv_row(row)
    existing_ids.add(run_id)
    log(f"[{run_id}] CSV OK  confidence={row['classifier_mean_confidence']:.4f}  loss={row['classifier_mean_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args()

    with open(args.matrix) as f:
        matrix = json.load(f)

    existing_ids = load_existing_run_ids()
    log(f"Existing CSV run_ids: {sorted(existing_ids) or '(none)'}")

    runs = sorted(matrix["planned_runs"], key=lambda r: r.get("priority", 99))
    for run_spec in runs:
        try:
            execute_run(run_spec, existing_ids, dry_run=args.dry_run)
        except Exception as e:
            log(f"[{run_spec['run_id']}] ERROR: {e}")
            log("Halting.")
            sys.exit(1)

    log("All targeted runs complete.")


if __name__ == "__main__":
    main()
