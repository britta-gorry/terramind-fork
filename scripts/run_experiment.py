#!/usr/bin/env python3
"""
run_experiment.py
=================
Fill in the start_config.yaml, then run interactively:

    pixi run python run_experiment.py

Or submit to Aqua by running submit_hpc.aqua (see that file).
The PBS job just calls this same script — nothing else changes.
"""
from __future__ import annotations
import csv, logging, os, shlex, subprocess, sys, time, yaml, copy, argparse
from ruamel.yaml import YAML
from datetime import datetime, timedelta
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# RUN CONDITIONS
# ──────────────────────────────────────────────────────────────────
_defaults = dict(
    start_cfg       = Path("scripts/start_config.yaml"),
    backbone_cfg    = Path("scripts/model_config.yaml"),
    datasets_cfg    = Path("scripts/datasets_config.yaml")
)

# window size -- used for divider length
_window = 90

# ──────────────────────────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────────────────────────

def get_cli_args(args) -> dict:
    """ Compile args from CLI for smooth overwriting"""
    cli_args = dict()
    for arg_flag, arg_value in vars(args).items():
        if arg_value is not None and arg_flag not in _defaults:
            cli_args[arg_flag] = arg_value
    return cli_args

def config(args, cli_args: dict) -> tuple[Path, Path]:
    """ Generate run_config.yaml by patching model and dataset configs to run_conditions.yaml."""
    # Setup ruamel parser to preserve formatting, spacing, and comments
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    # --- Load separate config files ---
    with open(args.start_cfg, "r") as f:
        run_config = yaml.load(f)
    with open(args.backbone_cfg, "r") as f:
        model_config = yaml.load(f)
    with open(args.datasets_cfg, "r") as f:
        datasets = yaml.load(f)
    # Overwrite from CLI args
    for key, value in cli_args.items():
        run_config["setup"][key] = value
    # Get run details if not from CLI args
    run_name = find_key(run_config["setup"], "name")
    target_dataset = find_key(run_config["setup"], "dataset")  
    target_backbone = find_key(run_config["setup"], "backbone")
    # --- Setup Output Directory and Logging ---
    if "output" in os.environ:
        out = Path(os.environ["output"])
    else:
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        out = Path("output") / f"{run_name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "run.log"),
        ],
    )
    # --- Patch together run_config ---
    # Extract dataset block from datasets_config.yaml
    dataset_info = find_key(datasets, target_dataset)
    # Extract keys from model config to patch data-specific info
    init_args = model_config["model"]["init_args"]
    backbone_args = init_args["model_args"]
    # Ensure model paths exist
    if "model" not in model_config: model_config["model"] = {}
    if "init_args" not in model_config["model"]: model_config["model"]["init_args"] = {}
    if "model_args" not in model_config["model"]["init_args"]: model_config["model"]["init_args"]["model_args"] = {}
    # Patch 1: class names
    class_names = find_key(dataset_info, "class_names")
    if class_names:
        init_args["class_names"] = class_names
    # Patch 2: backbone properties (if they exist for this specific dataset)
    backbone_keys = ["backbone_modalities", "backbone_tim_modalities", "backbone_merge_method", "backbone_bands"]
    tim_compatibility(target_backbone, backbone_keys)
    for key in backbone_keys:
        found_value = find_key(dataset_info, key)
        if found_value is not None:
            backbone_args[key] = found_value
    # Patch 3: model backbone
    if target_backbone == "terramind_v1_base":
        backbone_args["backbone"] = find_key(run_config, "backbone", target_backbone)
    elif target_backbone == "terramind_v1_base_tim":
        backbone_args["backbone"] = find_key(run_config, "backbone", target_backbone)
    elif target_backbone == "terramind_v1_large":
        backbone_args["backbone"] = find_key(run_config, "backbone", target_backbone)
        # Find and update the indices inside the list
        if "necks" in backbone_args:
            for neck in backbone_args["necks"]:
                if neck.get("name") == "SelectIndices":
                    neck["indices"] = [5, 11, 17, 23]
    elif target_backbone == "terramind_v1_large_tim":
        backbone_args["backbone"] = find_key(run_config, "backbone", target_backbone)
        # Find and update the indices inside the list
        if "necks" in backbone_args:
            for neck in backbone_args["necks"]:
                if neck.get("name") == "SelectIndices":
                    neck["indices"] = [5, 11, 17, 23]
    else:
        print(f"Target backbone {target_backbone} could not be parsed. Check run_conditions_yaml")
    # Patch 4: freeze conditions and max epochs
    if "freeze_backbone" in run_config["setup"]["overrides"]:
        init_args["freeze_backbone"] = find_key(run_config, "freeze_backbone", False)
    if "freeze_decoder" in run_config["setup"]["overrides"]:
        init_args["freeze_decoder"] = find_key(run_config, "freeze_decoder", False)
    if "max_epochs" in run_config["setup"]["overrides"]:
        model_config["trainer"]["max_epochs"] = find_key(run_config, "max_epochs", 10) # default 10 as fall back
    # Overwrite blocks in run_config
    run_config["seed_everything"] = model_config.get("seed_everything", 42)
    run_config["dataset_notes"] = dataset_info.get("dataset_notes", "")
    run_config["generation"] = dataset_info.get("generation", "")
    run_config["data"] = dataset_info.get("data", {})
    run_config["model"] = model_config.get("model", {})
    run_config["trainer"] = model_config.get("trainer", {})
    run_config["optimizer"] = model_config.get("optimizer", {})
    run_config["lr_scheduler"] = model_config.get("lr_scheduler", {})
    # --- Save as run_config.yaml in output directory ---
    cfg_path = out / "run_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(run_config, f)
    # Replace output directory placeholder
    cfg_text = cfg_path.read_text()
    cfg_text = cfg_text.replace("__OUTPUT_DIR__", str(out))
    cfg_path.write_path(cfg_text) if hasattr(cfg_path, 'write_path') else cfg_path.write_text(cfg_text)
    # --- Return the output directory and the path to the final run_config.yaml ---
    return out, cfg_path

def cli_config(cfg_path: Path) -> tuple[Path, Path]:
    """Load the final run_config.yaml for use in CLI commands, filtering only keys for terratorch CLI use.
    One cfg has CSVLogger, one without to prevent overwriting metrics.csv from reinitialisation."""
    if not cfg_path.exists():
        _log(f"ERROR: Config file not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r") as f:
        run_config = yaml.safe_load(f)
    # Filter the config to include only keys relevant for terratorch CLI use
    cli_keys = ["seed_everything", "data", "model", "trainer", "optimizer", "lr_scheduler"]
    
    cli_config = {key: run_config.get(key) for key in cli_keys if key in run_config}
    # Save the filtered config to a new YAML file
    cli_cfg_path = cfg_path.parent / "cli_config.yaml"
    with open(cli_cfg_path, "w") as f:
        yaml.dump(cli_config, f)
    # WITHOUT CSV logger (for test/predict)
    cli_config_no_log = copy.deepcopy(cli_config)
    if "trainer" in cli_config_no_log and "logger" in cli_config_no_log["trainer"]:
        logger_block = find_key(cli_config_no_log["trainer"], "logger")
        # Keep only the loggers that are NOT the CSVLogger
        cli_config_no_log["trainer"]["logger"] = [
            logger for logger in logger_block 
            if logger.get("class_path") != "lightning.pytorch.loggers.CSVLogger"
        ]
    cli_cfg_no_log_path = cfg_path.parent / "cli_config_no_logger.yaml"
    with open(cli_cfg_no_log_path, "w") as f:
        yaml.dump(cli_config_no_log, f)
    # Return both cfg with and without logger to be used for different steps
    return cli_cfg_path, cli_cfg_no_log_path

def tim_compatibility(backbone: str, backbone_keys):
    """TiM needs all pretrained bands and cannot specify subset.
    Without TiM, backbone_tim_modalities must be removed."""
    if "_tim" not in backbone:
        backbone_keys.remove("backbone_tim_modalities")
    elif "_tim" in backbone:
        backbone_keys.remove("backbone_bands")

def _log(msg: str):
    logging.getLogger().info(msg)

# ──────────────────────────────────────────────────────────────────
#  COMMAND RUNNER
# ──────────────────────────────────────────────────────────────────

def run(cmd: str) -> bool:
    """Run a shell command, stream output to the log, raise error on failure."""
    _log(f"$ {cmd}")
    try:
        # ── datamodule workaround: tell terratorch where to find custom modules ──
        env = os.environ.copy()
        env["TERRATORCH_CUSTOM_MODULE_PATH"] = "scripts"
        # ── end ──
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=env,   # pass the modified environment for custom modules
        )
        for line in proc.stdout:
            _log("  " + line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            _log(f"  ← exited with code {proc.returncode}")
            raise RuntimeError(f"Command failed with exit code {proc.returncode}")
        return True
    except FileNotFoundError:
        _log(f"  ERROR: command not found ({shlex.split(cmd)[0]}) — is pixi available?")
        sys.exit(1)
    except Exception as exc:
        _log(f"  ERROR: {exc}")
        sys.exit(1)

def find_checkpoint(out_dir: Path) -> Path | None:
    ckpt_dir = out_dir / "checkpoints"
    for name in ("best.ckpt", "last.ckpt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    hits = list(ckpt_dir.glob("*.ckpt"))
    return hits[0] if hits else None


# ──────────────────────────────────────────────────────────────────
#  PRINT RUN DETAILS TO TERMINAL
# ──────────────────────────────────────────────────────────────────

def find_key(source_dict, target_key, default=None):
    """Recursively search a nested dictionary for a specific key."""
    if target_key in source_dict:
        return source_dict[target_key]
    # If not, look inside every sub-dictionary at this level
    for value in source_dict.values():
        if isinstance(value, dict):
            # Dig deeper
            result = find_key(value, target_key)
            if result is not None:
                return result
    return default

def get_setup_details(run_config) -> dict:
    """Compile setup summary from config details for cleaner terminal display. """
    # Get key blocks for easier navigation
    # setup_block = dict(run_config.get("setup", {}))
    dataset_notes_block = dict(run_config.get("dataset_notes", ""))
    data_block = dict(run_config.get("data", {}))
    model_block = dict(run_config.get("model", {}))
    trainer_block = dict(run_config.get("trainer", {}))
    optimizer_block = dict(run_config.get("optimizer", {}))
    generation_block = dict(run_config.get("generation", {}))
    # lr_scheduler_block = run_config.get("lr_scheduler", {}),
    # --- Tidy key-value pairs that print with messy formatting ---
    # Modalities
    raw_data_modalities = find_key(data_block, "modalities", [])
    if isinstance(raw_data_modalities, list):
        # Extract the 'name' field if the items are dictionaries
        data_modality_names = [m["name"] for m in raw_data_modalities if isinstance(m, dict) and "name" in m]
        # Fallback to standard string conversion if it's already a list of strings
        if not data_modality_names:
            data_modality_names = [str(m) for m in raw_data_modalities if not isinstance(m, dict)]
        data_modalities_str = ", ".join(data_modality_names)
    else:
        data_modalities_str = str(raw_data_modalities)
    backbone_modalities = find_key(model_block, "backbone_modalities", [])
    backbone_modalities_str = ", ".join(backbone_modalities) if isinstance(backbone_modalities, list) else str(backbone_modalities)
    # Class names
    class_names = find_key(model_block, "class_names", [])
    classes_str = ", ".join(class_names) if isinstance(class_names, list) else str(class_names)
    # Model task and data module
    model_task_path = find_key(model_block, "class_path", "Unknown Model Task")
    task_class = model_task_path.split(".")[-1]
    data_module_path = find_key(data_block, "class_path", "Unknown Data Module")
    data_module = data_module_path.split(".")[-1]
    # Necks
    neck_list = find_key(model_block["init_args"], "necks", [])
    model_neck = ", ".join([n["name"] for n in neck_list if "name" in n]) if neck_list else "None"
    # --- Compile summary dictionary ---
    setup_details = {
        "setup": run_config.get("setup", {}),
        # Model
        "model_summary": {
            "backbone": find_key(model_block, "backbone", "N/A"),
            "backbone_pretrained": find_key(model_block, "backbone_pretrained", False),
            "backbone_modalities": backbone_modalities_str,
            "backbone_merge": find_key(model_block, "backbone_merge_method", "None"),
            "task": task_class,
            "classes": classes_str,
            "neck": model_neck,
            "decoder": f"{find_key(model_block, 'decoder', 'N/A')} ({find_key(model_block, 'decoder_channels', 'N/A')} channels)" if hasattr(model_block, 'get') else "N/A",
            "freeze_backbone": find_key(model_block, "freeze_backbone", False),
            "freeze_decoder": find_key(model_block, "freeze_decoder", False),
        },
        # Dataset
        "dataset_summary": {
            "data_name": find_key(dataset_notes_block, "name", "N/A"),
            "data_module": data_module,
            "task": find_key(data_block, "task", "N/A"),
            "modalities": data_modalities_str,
        },
        # Dataset Notes
        "dataset_notes": dataset_notes_block,
        # Trainer
        "trainer_summary": {
            "max_epochs": find_key(trainer_block, "max_epochs", "N/A"),
            "loss": find_key(model_block, "loss", "N/A"),
            "precision": find_key(trainer_block, "precision", "N/A"),
            "optimizer": find_key(optimizer_block, "class_path", "Unknown Optimizer").split(".")[-1],
            "lr": find_key(optimizer_block, "lr", "N/A"),
        },
        # Generation
        "generation_summary": generation_block,
    }
    return setup_details

def print_setup(out_dir: Path, setup_details: dict):
    """Print and log the experiment setup conditions in a readable box."""
    W = _window - 2   # inner width (W = window size - 2 border chars)
    def row(text: str) -> str:
        content = "  " + text
        if len(content) > W:
            content = content[:W - 1] + "…"
        return "║" + content.ljust(W) + "║"
    def div(label: str = "") -> str:
        if label:
            fill = W - len(label) - 4
            return "╠══ " + label + " " + "═" * fill + "╣"
        return "╠" + "═" * W + "╣"
    # Set key info
    NAME    = setup_details["setup"]["name"]
    AIM     = setup_details["setup"]["aim"]
    RUN_NOTES = setup_details["setup"]["run_notes"]
    # Steps
    GENERATE  = setup_details["setup"]["steps"]["generate"]
    FIT       = setup_details["setup"]["steps"]["fit"]
    TEST      = setup_details["setup"]["steps"]["test"]
    PREDICT   = setup_details["setup"]["steps"]["predict"]
    VISUALISE = setup_details["setup"]["steps"]["visualise"]
    steps   = (
        f"{'[✓]' if GENERATE else '[✗]'} generate   "
        f"{'[✓]' if FIT else '[✗]'} fit   "
        f"{'[✓]' if TEST else '[✗]'} test   "
        f"{'[✓]' if PREDICT else '[✗]'} predict   "
        f"{'[✓]' if VISUALISE else '[✗]'} visualise"
    )
    # Formatted summary
    box = [
        "╔" + "═" * W + "╗",
        row(f"TERRAMIND EXPERIMENT  ·  {NAME}"),
        row(f"AIM:  {AIM}"),
        row(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M')}")]
    box += [div("NOTES")]
    box.append(row(RUN_NOTES)),
    box += [div("DATASET")]
    for key, value in setup_details["dataset_summary"].items():
        box.append(row(f"{key:<20}  {value}")),
    box += [div("DATASET NOTES")]
    for key, value in setup_details["dataset_notes"].items():
        # Print nested lists for multimodal datasets
        if key == "modalities" and isinstance(value, list):
            modality_notes = find_key(setup_details["dataset_notes"], "modalities")
            mod_num = len(modality_notes)
            box.append(row(f"{key:<20}  {mod_num}"))
            indent = "     "
            for mod in modality_notes:
                for key, value in mod.items():
                    if key == "name":
                        box.append(row(f"{indent}└─ {value}"))
                    else: 
                        box.append(row(f"{indent}{indent}{key:<10}  {value}"))
        else:
            # Print all other standard single-line notes normally
            box.append(row(f"{key:<20}  {value}"))
    box += [div("MODEL")]
    for key, value in setup_details["model_summary"].items():
        box.append(row(f"{key:<20}  {value}")),
    box += [div("TRAINING")]
    for key, value in setup_details["trainer_summary"].items():
        box.append(row(f"{key:<20}  {value}")),
    box += [div("STEPS")]
    box.append(row(steps)),
    box += [div("OUTPUT DIRECTORY")]
    box.append(row(f"output → {out_dir}")),
    box.append("╚" + "═" * W + "╝"),
    # text = "\n".join(str(item) for item in box)
    # print("\n" + text + "\n")
    for line in box:
        _log(line)

def print_summary(out_dir: Path, results: dict, metrics: dict, elapsed_s: float):
    """Print the end-of-run summary and write it to run_summary.txt."""
    W  = _window - 2
    hr = "╠" + "═" * W + "╣"

    def row(text):
        content = "  " + text
        if len(content) > W:
            content = content[:W - 1] + "…"
        return "║" + content.ljust(W) + "║"

    dur = str(timedelta(seconds=int(elapsed_s)))
    box = [
        "╔" + "═" * W + "╗",
        row(f"RESULTS"),
        row(f"Duration: {dur}"),
        hr,
    ]
    for step, status in results.items():
        icon = "✓" if status == "ok" else ("✗" if "FAIL" in status else "—")
        box.append(row(f"  [{icon}] {step:<12}  {status}"))
    if metrics:
        box.append(hr)
        for k, v in metrics.items():
            box.append(row(f"  {k:<26}  {v}"))
    box += [
        hr,
        row(f"All output in:  {out_dir}"),
        "╚" + "═" * W + "╝",
    ]

    # text = "\n".join(box)
    # print("\n" + text + "\n")
    for line in box:
        _log(line)
    # (out_dir / "run_summary.txt").write_text(text)

# ──────────────────────────────────────────────────────────────────
#  GENERATE PSEUDOLABELS
# ──────────────────────────────────────────────────────────────────

def generate_pseudolabels(out_dir: Path, generation_dict: dict):
    """Generate pseudolabels by producing LULC from input modalities."""
    try: 
        gens_out = out_dir / "generations"
        gens_out.mkdir(parents=True, exist_ok=True)
        generation_args = {
            "--input_dir": generation_dict.get("generate_data_root"),
            "--output_dir": gens_out,
            "--modality": generation_dict.get("modality"),
            "--img_glob": generation_dict.get("img_glob"),
            "--split_file": generation_dict.get("generate_split"),
        }
        args_str = ''.join([f"{k} {v} " for k,v in generation_args.items()])
        print(args_str)
        ok = run(f"pixi run python scripts/generate_pseudolabels.py {args_str}")
    except Exception as exc:
        _log(f"  Could not generate: {exc}")
        return


# ──────────────────────────────────────────────────────────────────
#  METRICS and VISUALISATION
# ──────────────────────────────────────────────────────────────────

def read_metrics(out_dir: Path) -> dict:
    """Parse best val/mIoU and test metrics from CSVLogger's metrics.csv."""
    csv_path = out_dir / "metrics.csv"
    if not csv_path.exists():
        return {}
    out = {}
    best_miou, best_ep = -1.0, None
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ep = row.get("epoch", "")
                if not ep or ep == "nan":
                    continue
                ep = int(float(ep))
                vm = row.get("val/mIoU", "")
                if vm and vm != "nan":
                    v = float(vm)
                    if v > best_miou:
                        best_miou, best_ep = v, ep
                for col in ("test/mIoU", "test/loss", "test/F1Score"):
                    val = row.get(col, "")
                    if val and val != "nan":
                        out[col] = f"{float(val):.4f}"
        if best_miou >= 0:
            out["best_val_mIoU"] = f"{best_miou:.4f}  (epoch {best_ep})"
    except Exception as exc:
        _log(f"  Could not parse metrics.csv: {exc}")
    return out

def make_plots(out_dir: Path, name: str):
    """Plot val/mIoU and loss curves from metrics.csv.  Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        _log("  matplotlib not available — skipping plots.")
        return

    csv_path = out_dir / "metrics.csv"
    if not csv_path.exists():
        _log("  No metrics.csv yet — skipping plots.")
        return

    epochs, val_miou, val_loss = [], [], []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ep = row.get("epoch", "")
                if not ep or ep == "nan":
                    continue
                vm = row.get("val/mIoU", "")
                if vm and vm != "nan":
                    epochs.append(int(float(ep)))
                    val_miou.append(float(vm))
                    vl = row.get("val/loss", "")
                    val_loss.append(float(vl) if vl and vl != "nan" else float("nan"))
    except Exception as exc:
        _log(f"  Could not read metrics.csv for plotting: {exc}")
        return

    if not epochs:
        _log("  No val/mIoU values in metrics.csv yet.")
        return

    vis_dir = out_dir / "visualisations"
    vis_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{name} — Training Curves", fontsize=11)

    ax1.plot(epochs, val_miou, "b-o", markersize=3, label="val/mIoU")
    best = max(val_miou)
    best_ep = epochs[val_miou.index(best)]
    ax1.axvline(best_ep, color="r", ls="--", alpha=0.5,
                label=f"best {best:.4f} @ epoch {best_ep}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("mIoU")
    ax1.set_title("Validation mIoU"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    clean_vl = [(e, v) for e, v in zip(epochs, val_loss) if v == v]
    if clean_vl:
        ep_vl, vl = zip(*clean_vl)
        ax2.plot(ep_vl, vl, "r-o", markersize=3, label="val/loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_png = vis_dir / "training_curves.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Saved: {out_png}")

def map_predictions(dataset: str, out_dir: Path):
    """Map predictions compared to input image."""
    if dataset == "sen1floods11":
        ok = run(f"pixi run python scripts/map_pred_sen1floods11.py {out_dir}")
    elif dataset == "burnscars" or dataset == "burnscars12bands":
        ok = run(f"pixi run python scripts/map_pred_burnscars.py {out_dir}")
    else:
        _log(f"  Could not map predictions for dataset: {dataset}")


# ──────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    # ── ARGS ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run Terratorch experiments.")
    # Experiment name
    parser.add_argument("--name", type=str, default=None,
        help=f"String for terramind experiment name."
    )
    # Backbone string
    parser.add_argument("--backbone", type=str, default=None,
        help=f"String for terramind model (terramind_v1_base, terramind_v1_base_tim, etc.)"
    )
    # Dataset string
    parser.add_argument("--dataset", type=str, default=None,
        help=f"String for terramind dataset (sen1floods11, burnscars, etc.)"
    )
    # Start conditions config
    parser.add_argument("--start_cfg", type=Path, default=_defaults["start_cfg"],
        help="Path to the starter run conditions YAML (default: %(default)s)"
    )
    # Model config
    parser.add_argument("--backbone_cfg", type=Path, default=_defaults["backbone_cfg"],
        help="Path to the MODEL config YAML (default: %(default)s)"
    )
    # Datasets config
    parser.add_argument("--datasets_cfg", type=Path, default=_defaults["datasets_cfg"],
        help="Path to the DATASETS config YAML with all data information (default: %(default)s)"
    )
    args = parser.parse_args()
    cli_args = get_cli_args(args)

    # ── SETUP ──────────────────────────────────────────────────────
    print("\n\t   >>> TERRAMIND -- SETUP")
    print("\t  " + "=" * _window) # =================================
    
    out, cfg = config(args, cli_args)
    cli_cfg, cfg_no_log = cli_config(cfg)

    with open(cfg, "r") as f:
        RUN_CONFIG = yaml.safe_load(f)
    
    setup_details = get_setup_details(RUN_CONFIG)
    label_num = find_key(setup_details["dataset_notes"],"label_num")

    NAME    = setup_details["setup"]["name"]
    DATASET = setup_details["setup"]["dataset"]

    GENERATE  = setup_details["setup"]["steps"]["generate"]
    FIT       = setup_details["setup"]["steps"]["fit"]
    TEST      = setup_details["setup"]["steps"]["test"]
    PREDICT   = setup_details["setup"]["steps"]["predict"]
    VISUALISE = setup_details["setup"]["steps"]["visualise"]

    CHECKPOINT = setup_details["setup"]["overrides"]["checkpoint"]
    print(f"Checkpont from setup: {CHECKPOINT}")

    t0 = time.time()
    results: dict[str, str] = {}
    
    _log(f"Starting experiment with args:")
    if len(cli_args) > 0 :
        for key, value in cli_args.items():
            print(f"\t\t· {key:<13}: {value}")
    else :
        for key, value in _defaults.items():
            print(f"\t\t· {key:<13}: {value}")

    print_setup(out, setup_details)

    # ── GENERATE ───────────────────────────────────────────────────
    _log("\n\t   >>> TERRAMIND -- GENERATE")
    _log("=" * _window) # =============================================
    if DATASET == "burnscars" or DATASET == "burnscars12bands":
        if GENERATE:
            _log(f"Unable to perform TerraMind generation on {DATASET} due to mismatch in number of bands.")
            _log(f"Band adaptation applies transforms via cli yaml, but does not work with this self-contained \ngeneration script.")
            _log(f"Skipping GENERATE -- remove this if the issue is solved.")
            results["generate"] = "skipped (burnscars)"
    if GENERATE and DATASET == "sen1floods11":
        _log("─" * 50)
        generate_pseudolabels(out, setup_details.get("generation_summary", "N/A"))
        results["generate"] = "ok"
    else:
        _log(f"GENERATE: {GENERATE}, skipping")
        results["generate"] = "skipped"
    
    # ── FIT ─────────────────────────────────────────────────────────
    _log("\n\t   >>> TERRAMIND -- FIT")
    _log("=" * _window) # =============================================
    if FIT:
        _log("─" * 50)
        fit_cmd = (
            f"pixi run terratorch fit -c {cli_cfg} "
        )
        ok = run(fit_cmd)
        results["fit"] = "ok" if ok else "FAILED — see run.log"
    else:
        _log(f"FIT: {FIT}, skipping")
        results["fit"] = "skipped"

    # ── resolve checkpoint ──────────────────────────────────────────
    if CHECKPOINT is None:
        ckpt = find_checkpoint(out)
    else:
        ckpt = Path(CHECKPOINT)
    if ckpt:
        _log(f"Checkpoint: {ckpt}")
    else:
        _log("No checkpoint found — test and predict will be skipped.")

    # ── TEST ────────────────────────────────────────────────────────
    _log("\n\t   >>> TERRAMIND -- TEST")
    _log("=" * _window) # =============================================
    if TEST and label_num > 0:
        if ckpt:
            _log("─" * 50)
            test_cmd = (
            f"pixi run terratorch test -c {cfg_no_log} --ckpt_path {ckpt} "
        )
            ok = run(test_cmd)
            results["test"] = "ok" if ok else "FAILED — see run.log"
        else:
            results["test"] = "skipped (no checkpoint)"
    elif TEST and label_num == 0:
        results["test"] = "skipped (label_num = 0 — no ground truth to evaluate against)"
    else:
        _log(f"TEST: {TEST}, skipping")
        results["test"] = "skipped"

    # ── PREDICT ─────────────────────────────────────────────────────
    _log("\n\t   >>> TERRAMIND -- PREDICT")
    _log("=" * _window) # =============================================
    if PREDICT:
        predict_root = find_key(RUN_CONFIG["data"], "predict_data_root", None)
        if ckpt:
            _log("─" * 50)
            predict_cmd = (f"pixi run terratorch predict -c {cfg_no_log} --ckpt_path {ckpt} "
            f"--predict_output_dir {out / 'predictions'} "
            f"--data.init_args.predict_data_root '{predict_root}' "
            )
            ok = run(predict_cmd)
            results["predict"] = "ok" if ok else "FAILED — see run.log"
        else:
            results["predict"] = "skipped (no checkpoint)"
    else:
        _log(f"PREDICT: {PREDICT}, skipping")
        results["predict"] = "skipped"

    # ── VISUALISE ───────────────────────────────────────────────────
    _log("\n\t   >>> TERRAMIND -- VISUALISE")
    _log("=" * _window) # =============================================
    if VISUALISE:
        _log("─" * 50)
        make_plots(out, NAME)
        map_predictions(DATASET, out)
        results["visualise"] = "ok"
    else:
        _log(f"VISUALISE: {VISUALISE}, skipping")
        results["visualise"] = "skipped"

    # ── SUMMARY ─────────────────────────────────────────────────────
    _log("\n\t   >>> TERRAMIND -- SUMMARY")
    _log("=" * _window) # =============================================
    metrics = read_metrics(out)
    print_summary(out, results, metrics, time.time() - t0)


if __name__ == "__main__":
    main()
