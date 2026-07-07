#!/usr/bin/env python3
from __future__ import annotations  # Python 3.9 type-hint compat
"""
experiment_runner.py — TerraMind Unified Experiment Runner
===========================================================
One script for all TerraMind workflows: training, evaluation,
prediction, generation, and visualisation — on Aqua HPC or locally.

HOW TO USE
----------
1. Edit the CONFIGURATION block (between the ═══ lines). That is the
   only part you need to touch between experiments.

2. Run:
     Interactive (in your pixi environment or an interactive PBS node):
         pixi run python experiment_runner.py

     Submit to Aqua as a batch job (set SUBMIT_TO_HPC = True first):
         pixi run python experiment_runner.py

   When SUBMIT_TO_HPC = True the script writes a .aqua job file, calls
   qsub, and exits. The PBS job then re-runs this same script; it detects
   the PBS_JOBID environment variable and proceeds directly to the steps.

KEY CONCEPTS
------------
• HAS_LABELS controls which workflow is allowed — it gates steps but
  does not override your STEPS toggles. If HAS_LABELS = False you
  cannot run fit or test (there is nothing to train against). You CAN
  still run predict if you already have a checkpoint from a previous run.

• WHY YOU CANNOT JUST COMMENT OUT LABELS IN THE YAML: TerraTorch's
  GenericMultiModalDataModule always expects label files when task =
  'segmentation'. There is no way around this in the YAML. The correct
  approach for unlabelled data is to leave the segmentation YAML alone
  and use the generate / reconstruct steps instead — those use a
  completely separate code path that does not require labels.

• CHECKPOINT_PATH: set this explicitly to use a checkpoint from a
  previous run for test / predict without re-running fit.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  Edit this block before each run.
#  All other sections of the file stay the same between experiments.
# ═══════════════════════════════════════════════════════════════════════════

# ── Experiment identity ────────────────────────────────────────────────────
EXPERIMENT_NAME = "sen1floods11_base"
#  Short slug for the output folder.  Use lowercase + underscores only.
#  Examples: "burnscars_base", "sen1floods11_base_tim", "antarctica_hs_v1"

NOTES = "Baseline reproduction: S2L1C + S1GRD, UNetDecoder, AdamW lr=2e-5"
#  Free-text reminder saved in the run summary — write whatever helps you
#  remember why you ran this.

# ── Dataset & YAML config ─────────────────────────────────────────────────
CONFIG_FILE  = "configs/terramind_v1_base_sen1floods11.yaml"
#  Path to the YAML config relative to this script. The original is never
#  modified — the script copies and patches it into the output directory.

DATASET_NAME = "Sen1Floods11"
#  Human-readable label used only in the run summary.

HAS_LABELS   = True
#  True  → labelled workflow:   fit → test → predict → visualise
#  False → unlabelled workflow: generate / reconstruct → visualise
#          (fit and test are automatically blocked when False)

# ── Steps to run  (True = attempt, False = skip) ──────────────────────────
STEPS = {
    "fit":         True,   # Fine-tune TerraMind + decoder. Requires HAS_LABELS.
    "test":        True,   # Evaluate on test split.   Requires checkpoint + labels.
    "predict":     False,  # Write per-scene prediction GeoTIFFs. Requires checkpoint.
    "generate":    False,  # Any-to-any modality generation. No labels needed.
    "reconstruct": False,  # Tokenizer encode/decode — quantifies domain gap.
    "visualise":   True,   # Plot metric curves + prediction / generation maps.
}

# ── Checkpoint ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = None
#  None  → auto-detect best.ckpt inside this run's output/checkpoints/ after fit.
#  Set an explicit path to skip fit and reuse an existing checkpoint, e.g.:
#  CHECKPOINT_PATH = "output/sen1floods11_base_20260407_2038/checkpoints/best.ckpt"

# ── Generation / reconstruction (only relevant when those steps are True) ─
GENERATION = {
    # Point to your existing generation script, or leave None to use the
    # inline stub (which prints instructions and exits with a clear error).
    "script":             None,
    # e.g. "scripts/terramind_any_to_any_generation.py"

    "model_name":         "terramind_v1_base",
    "input_modality":     "S2L2A",
    "output_modalities":  ["S1GRD"],
    "input_image":        "path/to/your/image.tif",  # ← set before using
}

# ── HPC settings (only used when SUBMIT_TO_HPC = True) ────────────────────
SUBMIT_TO_HPC = False
HPC = {
    "ncpus":    8,
    "ngpus":    1,
    "mem":      "64gb",
    "walltime": "08:00:00",          # max 48:00:00 on Aqua
    "email":    "your@email.edu.au", # ← set your email for job notifications
    "modules":  ["cuda/12.4"],       # loaded inside the PBS job
}

# ═══════════════════════════════════════════════════════════════════════════
#  END OF CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

import csv
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Optional imports — visualisation degrades gracefully if unavailable ────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _MPL = True
except ImportError:
    _MPL = False

try:
    import rasterio
    import numpy as np
    _RIO = True
except ImportError:
    _RIO = False

# Module-level logger placeholder; replaced by setup_run().
log = logging.getLogger("runner")


# ──────────────────────────────────────────────────────────────────────────
#  RUN TRACKER  —  accumulates metadata; writes the summary at the end
# ──────────────────────────────────────────────────────────────────────────

class RunTracker:
    _STATUS_ICON = {
        "pending": "  ·  ",
        "running": "  ⟳  ",
        "ok":      "  ✓  ",
        "fail":    "  ✗  ",
        "skip":    "  —  ",
    }

    def __init__(self, output_dir: Path):
        self.output_dir  = output_dir
        self.start_wall  = time.time()
        self.start_dt    = datetime.now()
        self.steps       = {k: {"status": "pending", "elapsed": None, "detail": ""}
                            for k in STEPS}
        self.metrics     = {}
        self.files_saved = []
        self.checkpoint  = None
        self._t0         = {}

    # ── Step lifecycle ─────────────────────────────────────────────────────
    def begin(self, step: str):
        self.steps[step]["status"] = "running"
        self._t0[step] = time.time()
        log.info(f"{'─'*12} STEP: {step.upper()} {'─'*12}")

    def ok(self, step: str, detail: str = ""):
        self._close(step, "ok", detail)
        log.info(f"   ✓ {step}  {detail}  ({_elapsed(self._t0.get(step))})")

    def fail(self, step: str, detail: str = ""):
        self._close(step, "fail", detail)
        log.error(f"   ✗ {step} FAILED — {detail}")

    def skip(self, step: str, reason: str = ""):
        self.steps[step]["status"] = "skip"
        self.steps[step]["detail"] = reason
        log.info(f"   — {step} skipped  {reason}")

    def _close(self, step, status, detail):
        self.steps[step]["status"]  = status
        self.steps[step]["elapsed"] = time.time() - self._t0.get(step, time.time())
        self.steps[step]["detail"]  = detail

    # ── Summary output ─────────────────────────────────────────────────────
    def write_summary(self):
        total = time.time() - self.start_wall

        # JSON (machine-readable archive)
        summary = {
            "experiment":      EXPERIMENT_NAME,
            "notes":           NOTES,
            "timestamp":       self.start_dt.isoformat(),
            "total_elapsed_s": round(total, 1),
            "dataset":         DATASET_NAME,
            "has_labels":      HAS_LABELS,
            "config_file":     CONFIG_FILE,
            "output_dir":      str(self.output_dir),
            "checkpoint_used": str(self.checkpoint) if self.checkpoint else None,
            "steps":           self.steps,
            "metrics":         self.metrics,
            "files_saved":     [str(f) for f in self.files_saved],
        }
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )

        # Human-readable text for meetings
        W  = 60
        hr = "│" + "─" * W + "│"

        def row(text="", right="", width=W):
            content = f" {text}"
            if right:
                content = content.ljust(width - len(right) - 1) + right
            return "│" + content.ljust(width) + "│"

        lines = []
        lines.append("┌" + "─" * W + "┐")
        lines.append(row("TerraMind Run Summary"))
        lines.append(hr)
        lines.append(row(f"Experiment : {EXPERIMENT_NAME}"))
        lines.append(row(f"Date       : {self.start_dt.strftime('%Y-%m-%d  %H:%M')}"))
        lines.append(row(f"Duration   : {_elapsed(self.start_wall, from_now=True)}"))
        lines.append(hr)
        lines.append(row(f"Dataset    : {DATASET_NAME:<24}  Labels: {'Yes' if HAS_LABELS else 'No '}"))
        lines.append(row(f"Config     : {Path(CONFIG_FILE).name}"))
        out_s = str(self.output_dir)
        if len(out_s) > W - 13:
            out_s = "…" + out_s[-(W - 14):]
        lines.append(row(f"Output dir : {out_s}"))
        if self.checkpoint:
            ck = str(self.checkpoint)
            if len(ck) > W - 13:
                ck = "…" + ck[-(W - 14):]
            lines.append(row(f"Checkpoint : {ck}"))
        lines.append(hr)
        lines.append(row("Steps"))
        for step, info in self.steps.items():
            icon = self._STATUS_ICON.get(info["status"], "     ")
            el   = f"  ({_elapsed(None, info['elapsed'])})" if info.get("elapsed") else ""
            det  = f"  {info['detail']}" if info.get("detail") else ""
            lines.append(row(f"{icon}{step:<12}{el}{det}"))
        if self.metrics:
            lines.append(hr)
            lines.append(row("Metrics"))
            for k, v in self.metrics.items():
                lines.append(row(f"    {k:<26}  {v}"))
        if self.files_saved:
            lines.append(hr)
            lines.append(row(f"Visualisations  ({len(self.files_saved)} files in output/visualisations/)"))
        lines.append("└" + "─" * W + "┘")

        txt = "\n".join(lines)
        (self.output_dir / "run_summary.txt").write_text(txt)
        print("\n" + txt + "\n")


# ──────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────

def _elapsed(t0, seconds=None, from_now=False):
    """Return a HH:MM:SS string from a start timestamp or a duration."""
    if seconds is not None:
        s = int(seconds)
    elif from_now and t0 is not None:
        s = int(time.time() - t0)
    elif t0 is not None:
        s = int(time.time() - t0)
    else:
        return ""
    return str(timedelta(seconds=s))


def _run(cmd: list, label: str) -> bool:
    """
    Run a subprocess command, stream every line to the logger, and return
    True on success.  The full output also ends up in run.log.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    log.info(f"  $ {cmd_str}")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            log.info("    " + line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            log.error(f"  {label} exited with return code {proc.returncode}")
            return False
        return True
    except FileNotFoundError:
        log.error(f"  Command not found: {cmd[0]}  — is pixi available?")
        return False
    except Exception as exc:
        log.error(f"  {label} raised an unexpected error: {exc}")
        return False


def _find_checkpoint(output_dir: Path) -> Path | None:
    """Return best.ckpt, or last.ckpt, or any .ckpt found in checkpoints/."""
    ckpt_dir = output_dir / "checkpoints"
    for name in ("best.ckpt", "last.ckpt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    candidates = list(ckpt_dir.glob("*.ckpt"))
    return candidates[0] if candidates else None


# ──────────────────────────────────────────────────────────────────────────
#  SETUP  —  output directory, logging, YAML patching
# ──────────────────────────────────────────────────────────────────────────

def setup_run() -> tuple[Path, Path]:
    """
    Create (or reuse) the timestamped output directory, initialise logging,
    and write the patched YAML config.

    When called inside a PBS job the output dir is read from the
    TERRAMIND_OUTPUT_DIR environment variable so that all output lands in
    the same directory that was created during submission.
    """
    global log

    # ── Output directory ──────────────────────────────────────────────────
    if "TERRAMIND_OUTPUT_DIR" in os.environ:
        output_dir = Path(os.environ["TERRAMIND_OUTPUT_DIR"])
    else:
        ts         = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = Path("output") / f"{EXPERIMENT_NAME}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────
    log = logging.getLogger("runner")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if not log.handlers:
        fh = logging.FileHandler(output_dir / "run.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)

    log.info("=" * 60)
    log.info(f"Experiment : {EXPERIMENT_NAME}")
    log.info(f"Notes      : {NOTES}")
    log.info(f"Output dir : {output_dir}")
    log.info(f"HAS_LABELS : {HAS_LABELS}")
    log.info(f"Steps      : {[k for k, v in STEPS.items() if v]}")
    log.info("=" * 60)

    # ── Patch YAML config (replace __OUTPUT_DIR__ placeholder) ────────────
    cfg_src = Path(CONFIG_FILE)
    if not cfg_src.exists():
        log.error(f"Config file not found: {cfg_src}")
        log.error("Check that CONFIG_FILE in the CONFIGURATION block is correct.")
        sys.exit(1)

    cfg_text = cfg_src.read_text()
    cfg_text = cfg_text.replace("__OUTPUT_DIR__", str(output_dir))
    patched_cfg = output_dir / "run_config.yaml"
    patched_cfg.write_text(cfg_text)
    log.info(f"Patched config written to: {patched_cfg}")

    return output_dir, patched_cfg


# ──────────────────────────────────────────────────────────────────────────
#  STEP: FIT
# ──────────────────────────────────────────────────────────────────────────

def step_fit(tracker: RunTracker, cfg: Path) -> bool:
    if not STEPS["fit"]:
        tracker.skip("fit", "(disabled — set STEPS['fit'] = True to enable)")
        return True
    if not HAS_LABELS:
        tracker.skip("fit", "(HAS_LABELS=False — fit requires labelled data)")
        return True

    tracker.begin("fit")
    ok = _run(["pixi", "run", "terratorch", "fit", "-c", str(cfg)], "terratorch fit")
    if ok:
        tracker.ok("fit")
    else:
        tracker.fail("fit", "See run.log for the full terratorch error output.")
    return ok


# ──────────────────────────────────────────────────────────────────────────
#  STEP: TEST
# ──────────────────────────────────────────────────────────────────────────

def step_test(tracker: RunTracker, cfg: Path, ckpt: Path | None) -> bool:
    if not STEPS["test"]:
        tracker.skip("test", "(disabled — set STEPS['test'] = True to enable)")
        return True
    if not HAS_LABELS:
        tracker.skip("test", "(HAS_LABELS=False — test requires ground-truth labels)")
        return True
    if ckpt is None:
        tracker.skip(
            "test",
            "(no checkpoint found — run fit first or set CHECKPOINT_PATH explicitly)"
        )
        return True

    tracker.begin("test")
    ok = _run(
        ["pixi", "run", "terratorch", "test", "-c", str(cfg), "--ckpt_path", str(ckpt)],
        "terratorch test"
    )
    if ok:
        metrics = _parse_metrics(tracker.output_dir)
        tracker.metrics.update(metrics)
        tracker.ok("test", f"best val/mIoU = {metrics.get('best_val_mIoU', 'see metrics.csv')}")
    else:
        tracker.fail("test", "See run.log.")
    return ok


# ──────────────────────────────────────────────────────────────────────────
#  STEP: PREDICT
# ──────────────────────────────────────────────────────────────────────────

def step_predict(tracker: RunTracker, cfg: Path, ckpt: Path | None) -> bool:
    if not STEPS["predict"]:
        tracker.skip("predict", "(disabled — set STEPS['predict'] = True to enable)")
        return True
    if ckpt is None:
        tracker.skip(
            "predict",
            "(no checkpoint — run fit first or set CHECKPOINT_PATH explicitly)"
        )
        return True

    tracker.begin("predict")
    ok = _run(
        ["pixi", "run", "terratorch", "predict", "-c", str(cfg), "--ckpt_path", str(ckpt)],
        "terratorch predict"
    )
    if ok:
        pred_dir = tracker.output_dir / "predictions"
        n = len(list(pred_dir.glob("*.tif"))) if pred_dir.exists() else 0
        tracker.ok("predict", f"({n} prediction GeoTIFFs written)")
    else:
        tracker.fail("predict", "See run.log.")
    return ok


# ──────────────────────────────────────────────────────────────────────────
#  STEP: GENERATE  (any-to-any generation — no labels required)
# ──────────────────────────────────────────────────────────────────────────

def step_generate(tracker: RunTracker) -> bool:
    if not STEPS["generate"]:
        tracker.skip("generate", "(disabled — set STEPS['generate'] = True to enable)")
        return True

    tracker.begin("generate")
    gen_dir = tracker.output_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    # Option A: call your existing generation script.
    script = GENERATION.get("script")
    if script:
        script_path = Path(script)
        if not script_path.exists():
            tracker.fail("generate", f"Generation script not found: {script_path}")
            return False
        ok = _run(
            ["pixi", "run", "python", str(script_path),
             "--input",   GENERATION["input_image"],
             "--in_mod",  GENERATION["input_modality"],
             "--out_dir", str(gen_dir)],
            "generation script"
        )
        if ok:
            n = len(list(gen_dir.glob("*.tif")))
            tracker.ok("generate", f"({n} generated files in generated/)")
        else:
            tracker.fail("generate", "See run.log.")
        return ok

    # Option B: no script configured — write a clear explanation.
    log.error("  GENERATION['script'] is not set.")
    log.error("  To use the generate step:")
    log.error("    1. Point GENERATION['script'] to your any-to-any generation script.")
    log.error("    2. Also set GENERATION['input_image'] to a valid .tif path.")
    log.error("  Your existing large-tile generation script from Santiago's example")
    log.error("  should work — just point 'script' to it.")
    log.error("  Alternatively, use the TerraMind notebook workflow for now and set")
    log.error("  STEPS['generate'] = False until the script path is configured.")
    tracker.fail("generate", "GENERATION['script'] not configured — see run.log for instructions.")
    return False


# ──────────────────────────────────────────────────────────────────────────
#  STEP: RECONSTRUCT  (tokenizer encode/decode — domain-gap diagnostic)
# ──────────────────────────────────────────────────────────────────────────

def step_reconstruct(tracker: RunTracker) -> bool:
    if not STEPS["reconstruct"]:
        tracker.skip("reconstruct", "(disabled — set STEPS['reconstruct'] = True to enable)")
        return True

    tracker.begin("reconstruct")
    recon_dir = tracker.output_dir / "reconstruction"
    recon_dir.mkdir(exist_ok=True)

    script = GENERATION.get("script")
    if script:
        # Look for a reconstruction variant alongside the generation script.
        recon_candidates = [
            Path(script).parent / "tokenizer_reconstruction.py",
            Path(script).parent / "terramind_tokenizer_reconstruction.py",
            Path(script).parent / "reconstruction.py",
        ]
        recon_script = next((p for p in recon_candidates if p.exists()), None)
        if recon_script:
            ok = _run(
                ["pixi", "run", "python", str(recon_script),
                 "--input",   GENERATION["input_image"],
                 "--out_dir", str(recon_dir)],
                "reconstruction script"
            )
            if ok:
                tracker.ok("reconstruct", f"Reconstruction outputs in {recon_dir.name}/")
            else:
                tracker.fail("reconstruct", "See run.log.")
            return ok

    log.error("  No reconstruction script found.")
    log.error("  Set GENERATION['script'] to any script in the same folder as your")
    log.error("  tokenizer_reconstruction.py — the runner will find it automatically.")
    tracker.fail("reconstruct", "No reconstruction script found — see run.log.")
    return False


# ──────────────────────────────────────────────────────────────────────────
#  STEP: VISUALISE
# ──────────────────────────────────────────────────────────────────────────

def step_visualise(tracker: RunTracker) -> bool:
    if not STEPS["visualise"]:
        tracker.skip("visualise", "(disabled — set STEPS['visualise'] = True to enable)")
        return True
    if not _MPL:
        tracker.skip("visualise", "(matplotlib not available in this environment)")
        return True

    tracker.begin("visualise")
    vis_dir = tracker.output_dir / "visualisations"
    vis_dir.mkdir(exist_ok=True)
    saved = []

    # 1. Training / validation curves from metrics.csv
    p = _plot_training_curves(tracker.output_dir, vis_dir)
    if p:
        saved.append(p)

    # 2. Prediction maps (from terratorch predict output)
    pred_dir = tracker.output_dir / "predictions"
    if pred_dir.exists() and _RIO:
        for tif in sorted(pred_dir.glob("*.tif"))[:6]:
            p = _plot_prediction_map(tif, vis_dir)
            if p:
                saved.append(p)
    elif pred_dir.exists() and not _RIO:
        log.info("  rasterio not available — skipping prediction map plots.")

    # 3. Generated modality outputs
    gen_dir = tracker.output_dir / "generated"
    if gen_dir.exists() and _RIO:
        for tif in sorted(gen_dir.glob("*.tif"))[:4]:
            p = _plot_generated_tif(tif, vis_dir)
            if p:
                saved.append(p)

    tracker.files_saved.extend(saved)
    tracker.ok("visualise", f"({len(saved)} PNG files in visualisations/)")
    return True


# ── Visualisation helpers ──────────────────────────────────────────────────

def _plot_training_curves(output_dir: Path, vis_dir: Path) -> Path | None:
    """Plot val/mIoU and loss curves from metrics.csv produced by CSVLogger."""
    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists():
        log.info("  metrics.csv not found — training curves skipped.")
        return None
    try:
        epochs, val_miou, val_loss, train_loss = [], [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ep = row.get("epoch", "")
                if not ep or ep == "nan":
                    continue
                ep = int(float(ep))
                vm = row.get("val/mIoU", "")
                if vm and vm != "nan":
                    epochs.append(ep)
                    val_miou.append(float(vm))
                    vl = row.get("val/loss", "")
                    val_loss.append(float(vl) if vl and vl != "nan" else float("nan"))
                    tl = row.get("train/loss_epoch", row.get("train/loss", ""))
                    train_loss.append(float(tl) if tl and tl != "nan" else float("nan"))

        if not epochs:
            log.info("  No val/mIoU values in metrics.csv yet — curves skipped.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"{EXPERIMENT_NAME}  —  Training Curves", fontsize=11)

        # mIoU
        ax1.plot(epochs, val_miou, "b-o", markersize=3, label="val/mIoU")
        best = max(val_miou)
        best_ep = epochs[val_miou.index(best)]
        ax1.axvline(best_ep, color="r", linestyle="--", alpha=0.5,
                    label=f"best {best:.4f} @ ep {best_ep}")
        ax1.set_title("Validation mIoU")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("mIoU")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # Loss
        _vl = [v for v in val_loss if v == v]   # drop NaN
        _tl = [v for v in train_loss if v == v]
        if _vl:
            ax2.plot(epochs[:len(_vl)], _vl, "r-o", markersize=3, label="val/loss")
        if _tl:
            ax2.plot(epochs[:len(_tl)], _tl, "b-o", markersize=3, label="train/loss")
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = vis_dir / "training_curves.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: {out.name}")
        return out
    except Exception as exc:
        log.warning(f"  Could not plot training curves: {exc}")
        return None


def _plot_prediction_map(pred_tif: Path, vis_dir: Path) -> Path | None:
    """Quick-look map of one prediction GeoTIFF."""
    try:
        with rasterio.open(pred_tif) as src:
            data = src.read()
        mask = np.argmax(data, axis=0) if data.shape[0] > 1 else data[0]
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(mask, cmap="tab10", interpolation="nearest",
                       vmin=0, vmax=max(1, data.shape[0] - 1))
        ax.set_title(pred_tif.stem[:48], fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = vis_dir / f"pred_{pred_tif.stem}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: {out.name}")
        return out
    except Exception as exc:
        log.warning(f"  Could not plot {pred_tif.name}: {exc}")
        return None


def _plot_generated_tif(gen_tif: Path, vis_dir: Path) -> Path | None:
    """Quick-look plot of a generated modality GeoTIFF (up to 3 bands)."""
    try:
        with rasterio.open(gen_tif) as src:
            data = src.read()
        n = min(data.shape[0], 3)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            band = data[i].astype(float)
            p2, p98 = np.percentile(band[np.isfinite(band)], [2, 98])
            ax.imshow(np.clip(band, p2, p98), cmap="viridis")
            ax.set_title(f"Band {i + 1}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Generated: {gen_tif.stem[:50]}", fontsize=8)
        plt.tight_layout()
        out = vis_dir / f"gen_{gen_tif.stem}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: {out.name}")
        return out
    except Exception as exc:
        log.warning(f"  Could not plot {gen_tif.name}: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────
#  METRICS PARSING
# ──────────────────────────────────────────────────────────────────────────

def _parse_metrics(output_dir: Path) -> dict:
    """Extract best val/mIoU and any test metrics from CSVLogger's metrics.csv."""
    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists():
        return {}
    result = {}
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
                for col in ("test/mIoU", "test/loss", "test/F1Score", "test/IoU_Flood"):
                    val = row.get(col, "")
                    if val and val != "nan":
                        result[col] = f"{float(val):.4f}"
        if best_miou >= 0:
            result["best_val_mIoU"] = f"{best_miou:.4f}  (epoch {best_ep})"
    except Exception as exc:
        log.warning(f"Could not parse metrics.csv: {exc}")
    return result


# ──────────────────────────────────────────────────────────────────────────
#  HPC SUBMISSION  —  writes an .aqua job file and calls qsub
# ──────────────────────────────────────────────────────────────────────────

def submit_to_hpc(output_dir: Path) -> str | None:
    """
    Generate a PBS batch script (.aqua) and submit it.

    The script uses TERRAMIND_OUTPUT_DIR so the PBS job writes all output
    into the same directory created here during submission — not a new one.

    The PBS job re-runs this exact experiment_runner.py; it detects
    PBS_JOBID and skips re-submission, running the steps directly.
    """
    modules = "\n".join(f"module load {m}" for m in HPC.get("modules", []))
    script_abs = Path(__file__).resolve()

    pbs = textwrap.dedent(f"""\
        #!/bin/bash -l
        #PBS -N {EXPERIMENT_NAME}
        #PBS -l select=1:ncpus={HPC['ncpus']}:ngpus={HPC['ngpus']}:mem={HPC['mem']}
        #PBS -l walltime={HPC['walltime']}
        #PBS -m abe
        #PBS -M {HPC['email']}
        #PBS -j oe
        #PBS -o {output_dir}/pbs_output.log

        cd $PBS_O_WORKDIR
        {modules}

        # Pass the pre-created output directory to the Python script so all
        # output lands in the same place rather than a new timestamped folder.
        export TERRAMIND_OUTPUT_DIR={output_dir}

        echo "Job ID      : $PBS_JOBID"
        echo "Started     : $(date)"
        echo "Working dir : $PBS_O_WORKDIR"
        echo "Output dir  : $TERRAMIND_OUTPUT_DIR"

        pixi run python {script_abs}

        echo "Finished    : $(date)"
    """)

    job_path = output_dir / f"{EXPERIMENT_NAME}.aqua"
    job_path.write_text(pbs)
    log.info(f"PBS job script written to: {job_path}")

    result = subprocess.run(["qsub", str(job_path)], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip()
        log.info(f"Job submitted successfully: {job_id}")
        log.info(f"Monitor:   qstat -f {job_id}")
        log.info(f"Job log:   {output_dir}/pbs_output.log")
        log.info(f"Run log:   {output_dir}/run.log")
        log.info(f"Summary:   {output_dir}/run_summary.txt  (written at end of job)")
        return job_id
    else:
        log.error(f"qsub failed: {result.stderr.strip()}")
        log.error("Try SUBMIT_TO_HPC = False to run interactively and debug.")
        return None


# ──────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────

def main():
    output_dir, patched_cfg = setup_run()
    tracker = RunTracker(output_dir)

    inside_pbs = "PBS_JOBID" in os.environ

    # ── If SUBMIT_TO_HPC and not already inside a PBS job: submit and exit ─
    if SUBMIT_TO_HPC and not inside_pbs:
        log.info("SUBMIT_TO_HPC = True  →  generating PBS job script and submitting.")
        job_id = submit_to_hpc(output_dir)
        if job_id:
            log.info("Submission complete. This script's work is done.")
            log.info(f"All results will appear in:  {output_dir}/")
        else:
            log.error("Submission failed. Set SUBMIT_TO_HPC = False to debug interactively.")
        return

    if inside_pbs:
        log.info(f"Running inside PBS job: {os.environ['PBS_JOBID']}")
    else:
        log.info("Running interactively (SUBMIT_TO_HPC = False).")

    # ── Resolve checkpoint ─────────────────────────────────────────────────
    ckpt: Path | None = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else None
    if ckpt and not ckpt.exists():
        log.warning(f"CHECKPOINT_PATH does not exist: {ckpt}")
        log.warning("test and predict will be skipped unless fit produces a checkpoint.")
        ckpt = None

    # ── Labelled workflow ──────────────────────────────────────────────────
    step_fit(tracker, patched_cfg)

    # Auto-detect checkpoint produced by fit (if not set explicitly)
    if ckpt is None:
        ckpt = _find_checkpoint(output_dir)
        if ckpt:
            tracker.checkpoint = ckpt
            log.info(f"Checkpoint auto-detected: {ckpt}")
        else:
            log.info("No checkpoint found after fit — test and predict will be skipped.")

    step_test(tracker, patched_cfg, ckpt)
    step_predict(tracker, patched_cfg, ckpt)

    # ── Unlabelled / generative workflow ───────────────────────────────────
    step_generate(tracker)
    step_reconstruct(tracker)

    # ── Visualisation (always attempted regardless of workflow) ────────────
    step_visualise(tracker)

    # ── Run summary ────────────────────────────────────────────────────────
    # Collect any metrics that may have been written by test (final pass)
    tracker.metrics.update(_parse_metrics(output_dir))
    tracker.write_summary()


if __name__ == "__main__":
    main()
