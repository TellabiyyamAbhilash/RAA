"""Write a FedAS PA/CS ablation report from PFLlib .h5 result files."""
import argparse
import json
import os
from datetime import datetime

import h5py
import numpy as np


VARIANTS = [
    {
        "goal": "abl_none",
        "name": "Baseline",
        "pa": False,
        "cs": False,
        "extra": "--no-pa --no-cs",
        "blurb": "PA disabled, CS disabled. Standard sample-count aggregation is used.",
    },
    {
        "goal": "abl_PAonly",
        "name": "PA only",
        "pa": True,
        "cs": False,
        "extra": "--no-cs",
        "blurb": "PA enabled, CS disabled. Tests whether Parameter Alignment improves accuracy by itself.",
    },
    {
        "goal": "abl_CSonly",
        "name": "CS only",
        "pa": False,
        "cs": True,
        "extra": "--no-pa",
        "blurb": "PA disabled, CS enabled. Tests whether t-FIM-weighted aggregation improves accuracy by itself.",
    },
    {
        "goal": "abl_PA_CS",
        "name": "Complete FedAS",
        "pa": True,
        "cs": True,
        "extra": "",
        "blurb": "PA enabled, CS enabled. Represents the complete FedAS method.",
    },
]


def pct(value):
    return f"{value * 100:.2f}%"


def signed_pp(delta):
    if delta is None:
        return "—"
    return f"{delta:+.2f}"


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def optional_series(hf, key):
    """Return a 1-D float array, or None if the dataset is missing/empty.

    PFLlib always writes `rs_test_auc`, but FedAS never appends to it, so the
    dataset exists with length 0.
    """
    if key not in hf:
        return None
    arr = np.array(hf[key], dtype=np.float64).reshape(-1)
    return arr if arr.size else None


def at(arr, idx):
    if arr is None or arr.size == 0:
        return None
    if idx < 0:
        idx += len(arr)
    if idx < 0 or idx >= len(arr):
        return None
    return float(arr[idx])


def read_h5(path):
    with h5py.File(path, "r") as hf:
        acc = optional_series(hf, "rs_test_acc")
        if acc is None:
            raise ValueError(f"{path} has no rs_test_acc values")
        return {
            "acc": acc,
            "auc": optional_series(hf, "rs_test_auc"),
            "loss": optional_series(hf, "rs_train_loss"),
        }


def collect_results(results_dir, dataset, algorithm, times):
    rows = []
    for variant in VARIANTS:
        path = os.path.join(
            results_dir, f"{dataset}_{algorithm}_{variant['goal']}_{times}.h5"
        )
        row = dict(variant)
        row["path"] = path
        if not os.path.exists(path):
            row["missing"] = True
            rows.append(row)
            continue

        series = read_h5(path)
        acc = series["acc"]
        best_idx = int(np.argmax(acc))
        row.update({
            "missing": False,
            "acc": acc,
            "auc": series["auc"],
            "loss": series["loss"],
            "best_acc": float(acc[best_idx]),
            "best_round": best_idx,
            "final_acc": float(acc[-1]),
            "best_auc": at(series["auc"], best_idx),
            "final_loss": at(series["loss"], -1),
        })
        rows.append(row)
    return rows


def command_block(dataset, num_classes, model, algorithm, args, variant):
    extra = f" {variant['extra']}" if variant["extra"] else ""
    return (
        "```powershell\n"
        f"python main.py -data {dataset} -ncl {num_classes} -m {model} -algo {algorithm} `\n"
        f"  -gr {args.global_rounds} -ls {args.local_epochs} -lbs {args.batch_size} "
        f"-lr {args.learning_rate} -nc {args.num_clients} -jr {args.join_ratio} `\n"
        f"  -dev {args.device} -go {variant['goal']}{extra}\n"
        "```"
    )


def interpret(rows):
    finished = [r for r in rows if not r["missing"]]
    if not finished:
        return ["No result files were found, so no interpretation is available yet."]

    lines = []
    baseline = next((r for r in finished if r["goal"] == "abl_none"), None)
    complete = next((r for r in finished if r["goal"] == "abl_PA_CS"), None)
    pa_only = next((r for r in finished if r["goal"] == "abl_PAonly"), None)
    cs_only = next((r for r in finished if r["goal"] == "abl_CSonly"), None)

    if baseline:
        for other in (pa_only, cs_only, complete):
            if not other:
                continue
            delta = (other["best_acc"] - baseline["best_acc"]) * 100
            if abs(delta) < 0.05:
                lines.append(
                    f"- **{other['name']}** matched the baseline best accuracy "
                    f"({pct(baseline['best_acc'])}); the difference is smaller than 0.05 pp."
                )
            elif delta > 0:
                lines.append(
                    f"- **{other['name']}** improved the best accuracy from "
                    f"{pct(baseline['best_acc'])} to {pct(other['best_acc'])} ({delta:+.2f} pp)."
                )
            else:
                lines.append(
                    f"- **{other['name']}** was below the baseline best accuracy "
                    f"({pct(baseline['best_acc'])} → {pct(other['best_acc'])}, {delta:+.2f} pp)."
                )

    if pa_only and cs_only:
        gap = (cs_only["best_acc"] - pa_only["best_acc"]) * 100
        if abs(gap) < 0.05:
            lines.append("- PA alone and CS alone produced essentially the same best accuracy.")
        elif gap > 0:
            lines.append("- CS produced a larger standalone improvement than PA in these runs.")
        else:
            lines.append("- PA produced a larger standalone improvement than CS in these runs.")

    if complete and baseline and pa_only and cs_only:
        best = max(finished, key=lambda r: r["best_acc"])
        lines.append(f"- The highest best-accuracy in this ablation is **{best['name']}** at {pct(best['best_acc'])}.")
        combo = complete["best_acc"] - max(pa_only["best_acc"], cs_only["best_acc"])
        if combo > 0.0005:
            lines.append("- Combining PA and CS beat either component used alone.")
        elif combo < -0.0005:
            lines.append("- Combining PA and CS did not beat the stronger single-component run.")
        else:
            lines.append("- Combining PA and CS was about as good as the stronger single-component run.")

    missing = [r["name"] for r in rows if r["missing"]]
    if missing:
        lines.append("- Not yet generated: " + ", ".join(missing) + ".")

    if not lines:
        lines.append("- Results were written, but there is not enough overlap to compare variants.")
    return lines


def write_report(args):
    results_dir = os.path.abspath(args.results_dir)
    config = load_json(args.config)
    rows = collect_results(results_dir, args.dataset, args.algorithm, args.times)
    finished = [r for r in rows if not r["missing"]]
    baseline = next((r for r in finished if r["goal"] == "abl_none"), None)

    ncl = config.get("num_classes", args.num_classes)
    nfeat = config.get("num_features")
    classes = config.get("classes") or []
    alpha = config.get("alpha")
    nclients = config.get("num_clients", args.num_clients)
    partition = config.get("partition", "dir")

    lines = [
        f"# FedAS Ablation Experiment Report — {args.dataset}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Objective",
        "",
        "This experiment evaluates the contribution of the two FedAS components:",
        "",
        "- **PA — Parameter Alignment:** adapts the shared backbone to work better with each client’s personalized classifier.",
        "- **CS — Client Synchronization:** uses Fisher Information (t-FIM) to give different aggregation weights to clients.",
        "",
        "All runs use the same FedAS model and settings. Only PA and CS are enabled or disabled.",
        "",
        "## Dataset",
        "",
        "Edge-IIoTset is a tabular IoT/IIoT intrusion-detection dataset. "
        "Host addresses, timestamps and raw payloads are dropped before training, "
        "then the remaining fields are encoded and standardized.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset | `{args.dataset}` |",
        f"| Classes | {ncl} attack types |",
        f"| Features | {nfeat if nfeat is not None else 'see config.json'} |",
        f"| Clients | {nclients} |",
        f"| Partition | non-IID `{partition}`" + (f", Dirichlet α = {alpha}" if alpha is not None else "") + " |",
        f"| Result files | `{args.dataset}_{args.algorithm}_abl_*_{args.times}.h5` |",
        "",
    ]

    if classes:
        lines.append("Class index mapping:")
        lines.append("")
        for i, name in enumerate(classes):
            lines.append(f"- `{i}`: {name}")
        lines.append("")

    lines.extend(["## Commands used", ""])
    for i, variant in enumerate(VARIANTS, start=1):
        lines.append(f"### {i}. {variant['name']}")
        lines.append("")
        lines.append(command_block(args.dataset, ncl, args.model, args.algorithm, args, variant))
        lines.append("")
        lines.append(variant["blurb"])
        lines.append("")

    lines.extend([
        "## Parameters",
        "",
        "| Argument | Value | Meaning |",
        "|---|---:|---|",
        f"| `-data` | `{args.dataset}` | Tabular Edge-IIoTset client splits |",
        f"| `-ncl` | `{ncl}` | Number of output classes |",
        f"| `-m` | `{args.model}` | TabularMLP (auto-selected for this dataset) |",
        f"| `-algo` | `{args.algorithm}` | FedAS training algorithm |",
        f"| `-gr` | `{args.global_rounds}` | Global rounds; result files also store the round-0 evaluation |",
        f"| `-ls` | `{args.local_epochs}` | Local training epochs per selected client |",
        f"| `-lbs` | `{args.batch_size}` | Local batch size |",
        f"| `-lr` | `{args.learning_rate}` | SGD learning rate |",
        f"| `-nc` | `{args.num_clients}` | Total number of clients |",
        f"| `-jr` | `{args.join_ratio}` | Fraction of clients selected per round |",
        f"| `-dev` | `{args.device}` | Training device |",
        "| `-go` | varies | Unique result-file name |",
        "| `--no-pa` | optional | Disables Parameter Alignment |",
        "| `--no-cs` | optional | Disables Client Synchronization |",
        "",
        "## Results",
        "",
        "The paper reports the **highest test accuracy obtained across all rounds**, so the same metric is used here.",
        "",
        "| Variant | PA | CS | Best accuracy | Best round | Final accuracy | Best AUC | Final train loss | Improvement |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])

    for row in rows:
        pa = "On" if row["pa"] else "Off"
        cs = "On" if row["cs"] else "Off"
        if row["missing"]:
            lines.append(f"| {row['name']} | {pa} | {cs} | Not run | — | — | — | — | — |")
            continue
        delta = None
        if baseline and row["goal"] != "abl_none":
            delta = (row["best_acc"] - baseline["best_acc"]) * 100
        auc = pct(row["best_auc"]) if row["best_auc"] is not None else "—"
        loss = f"{row['final_loss']:.4f}" if row["final_loss"] is not None else "—"
        lines.append(
            f"| {row['name']} | {pa} | {cs} | **{pct(row['best_acc'])}** | "
            f"{row['best_round']} | {pct(row['final_acc'])} | {auc} | {loss} | "
            f"{signed_pp(delta)} |"
        )

    if finished:
        n_rounds = max(len(r["acc"]) for r in finished)
        lines.extend(["", "### Per-round test accuracy", ""])
        header = "| Round | " + " | ".join(r["name"] for r in rows) + " |"
        sep = "|---:|" + "|".join("---:" for _ in rows) + "|"
        lines.append(header)
        lines.append(sep)
        for t in range(n_rounds):
            cells = []
            for row in rows:
                if row["missing"] or t >= len(row["acc"]):
                    cells.append("—")
                else:
                    cells.append(f"{row['acc'][t] * 100:.2f}%")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Interpretation", ""])
    lines.extend(interpret(rows))
    lines.extend([
        "",
        "## Important limitation",
        "",
        "These are short ablation runs. They can show whether PA and CS help on Edge-IIoT, "
        "but the exact numbers should not be compared with the paper’s CIFAR tables. "
        "The paper used more global rounds, an image CNN, and a persistent straggler setup. "
        "This report uses a tabular MLP, randomly selected clients each round, and the "
        f"`-gr {args.global_rounds}` budget above.",
        "",
        "## Result files",
        "",
    ])
    for row in rows:
        status = "found" if not row["missing"] else "missing"
        lines.append(f"- `{os.path.basename(row['path'])}` ({status})")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return rows, os.path.abspath(args.output)


def print_summary(rows, output):
    print(f"\nReport written to {output}")
    print(f"{'Variant':<18} {'PA':<4} {'CS':<4} {'Best':>8} {'Round':>6} {'Final':>8}")
    for row in rows:
        pa = "On" if row["pa"] else "Off"
        cs = "On" if row["cs"] else "Off"
        if row["missing"]:
            print(f"{row['name']:<18} {pa:<4} {cs:<4} {'missing':>8}")
            continue
        print(
            f"{row['name']:<18} {pa:<4} {cs:<4} "
            f"{row['best_acc']*100:7.2f}% {row['best_round']:6d} {row['final_acc']*100:7.2f}%"
        )


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    system_dir = os.path.abspath(os.path.join(here, ".."))
    pfl_root = os.path.abspath(os.path.join(system_dir, ".."))
    default_results = os.path.join(pfl_root, "results")
    default_config = os.path.join(pfl_root, "dataset", "EdgeIIoT", "config.json")
    default_output = os.path.join(default_results, "EdgeIIoT_FedAS_abl_report.md")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="EdgeIIoT")
    parser.add_argument("--algorithm", default="FedAS")
    parser.add_argument("--model", default="MLP")
    parser.add_argument("--times", type=int, default=0, help="Result-file suffix, usually 0.")
    parser.add_argument("--results-dir", default=default_results)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("-ncl", "--num-classes", type=int, default=15)
    parser.add_argument("-gr", "--global-rounds", type=int, default=10)
    parser.add_argument("-ls", "--local-epochs", type=int, default=5)
    parser.add_argument("-lbs", "--batch-size", type=int, default=16)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.005)
    parser.add_argument("-nc", "--num-clients", type=int, default=20)
    parser.add_argument("-jr", "--join-ratio", type=float, default=0.6)
    parser.add_argument("-dev", "--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows, output = write_report(args)
    print_summary(rows, output)
    if all(r["missing"] for r in rows):
        raise SystemExit(
            "No ablation .h5 files found. Run the four FedAS commands first, "
            "or use run_edgeiiot_ablation.ps1 without -ReportOnly."
        )
