# FedAS Reproduction Plan (PFLlib)

**Paper:** Yang, Huang, Ye. *FedAS: Bridging Inconsistency in Personalized Federated Learning.* CVPR 2024, pp. 11986–11995.
Local copy: `C:\Users\abhil\OneDrive\Documents\FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.pdf`
Authors' code: https://github.com/xiyuanyang45/FedAS

**Code base used here:** PFLlib (`./PFLlib`), which contains its own `FedAS` reimplementation
(`system/flcore/servers/serveras.py`, `system/flcore/clients/clientas.py`).

**Machine:** Windows, CPU only (no CUDA device detected). Conda env `pfllib`
(Python 3.11, torch 2.0.1+cpu, torchvision 0.15.2+cpu, numpy 1.26.4, cvxpy 1.4.4).

> Read Section 7 (Known Mismatches) before comparing any number to the paper.
> Several of the paper's settings do not map onto PFLlib exactly, and PFLlib's FedAS is
> not a literal implementation of the paper's algorithm.

---

## 1. Fixed setup (paper §4.1, "Implementation Details")

These are held constant across every experiment in the paper.

| Paper setting | Paper value | PFLlib flag | Notes |
|---|---|---|---|
| Number of clients `M` | 20 | `-nc 20` | also hardcoded `num_clients = 20` in the generator scripts |
| Global rounds `Eg` | 40 | `-gr 40` | PFLlib loops `range(gr+1)` → 41 evaluations (rounds 0–40) |
| Local epochs `El` | 5 | `-ls 5` | |
| Optimizer | SGD | — | fixed in `clientbase.py`, not configurable |
| Learning rate | 5e-3 | `-lr 0.005` | |
| Learning rate decay | 1e-3 | *(no equivalent)* | see §7.4 |
| Batch size | 16 | `-lbs 16` | |
| Model | 6-layer CNN | `-m CNN` → 4-layer | see §7.1 |
| Output classes | 10 / 100 | `-ncl 10` / `-ncl 100` | **must set `-ncl 100` for Cifar100**; default is 10 |
| Seeds | torch = 0, numpy = 0 | torch only | see §7.5 |
| Reported metric | highest top-1 accuracy | `Best accuracy.` log line | PFLlib also prints AUC; the paper does not use it |

### Swept variables

| Symbol | Meaning | Values | How to set in PFLlib |
|---|---|---|---|
| `β` | Dirichlet concentration (data heterogeneity) | 0.1, 0.3, 0.5 | edit `alpha` in `dataset/utils/dataset_utils.py`, then regenerate (§3) |
| `P` | client online ratio (smaller = more stragglers) | 0.2, 0.4, 0.6, 1.0 | `-jr 0.2 / 0.4 / 0.6 / 1.0` — **semantics differ**, see §7.3 |

---

## 2. Environment activation (every new terminal)

`conda activate` does not work in a fresh PowerShell on this machine until conda is
initialised. Either run `conda init powershell` once and reopen the terminal, or do this
per-session:

```powershell
& "C:\Users\abhil\miniconda3\shell\condabin\conda-hook.ps1"
conda activate pfllib
```

Verify before running anything:

```powershell
$env:CONDA_DEFAULT_ENV                 # expect: pfllib
(Get-Command python).Source            # expect: ...\miniconda3\envs\pfllib\python.exe
python -c "import torch, numpy, cvxpy; print(torch.__version__, numpy.__version__, cvxpy.__version__)"
```

---

## 3. Dataset generation

`β` is **not** a command-line argument. It is a module-level constant:

```
PFLlib/dataset/utils/dataset_utils.py

10 | batch_size = 10
11 | train_ratio = 0.75   # merges original train+test, then re-splits 75/25
12 | alpha = 0.1          # <-- this is the paper's beta
```

Procedure for each `β`:

1. Edit line 12 to `alpha = 0.1` (or `0.3`, `0.5`).
2. Regenerate:

```powershell
cd "C:\Users\abhil\OneDrive\Desktop\Projects\Fed Learning\PFLlib\dataset"
python generate_Cifar10.py noniid - dir
python generate_Cifar100.py noniid - dir
```

`check()` compares the stored `alpha` against the current one, so changing it correctly
forces regeneration. **But it overwrites the same `Cifar10/` and `Cifar100/` folders.**
Finish one `β` sweep completely before moving to the next, or archive between runs:

```powershell
Copy-Item -Recurse .\Cifar10 .\Cifar10_beta0.1
```

Expected output per dataset: `config.json`, `train/0.npz … train/19.npz`, `test/0.npz … test/19.npz`.

> Note: `batch_size = 10` on line 10 is the *dataset config* batch size (used for the
> `least_samples` guarantee and written into `config.json`). It is independent of the
> training batch size `-lbs 16`. Leave it alone.

---

## 4. Result files

Each run writes `PFLlib/results/{dataset}_{algorithm}_{goal}_{times_index}.h5` containing
three arrays: `rs_test_acc`, `rs_test_auc`, `rs_train_loss` (one entry per evaluated round).

The `{goal}` component comes from `-go` (default `test`). **Runs with the same
dataset + algorithm + goal overwrite each other**, so tag every run in the grid:

```powershell
-go b0.1_P0.6
```

Reading a result back:

```powershell
python -c "import h5py,numpy as np; f=h5py.File('../results/Cifar10_FedAS_b0.1_P0.6_0.h5','r'); a=np.array(f['rs_test_acc']); print('rounds',len(a),'best',a.max())"
```

`a.max()` is the paper's reported metric (highest top-1 accuracy).

> **The saved model is not tagged by `-go`.** `save_global_model()` always writes
> `system/models/{dataset}/{algorithm}_server.pt`, so **every** FedAS run overwrites the
> same file — the `.h5` files are distinct but the `.pt` is not. E5 loads
> `models/Cifar10/FedAS_server.pt` and would silently pick up whichever run finished last.
> Copy the `.pt` aside immediately after any run you intend to use for E5.

---

## 5. The experiments

Ten distinct experiments. Status legend:
**[CLI]** runnable now with flags · **[LOG]** already produced, just parse output · **[PATCH]** needs a code change · **[NEW]** needs new code.

### E1 — Main comparison, Cifar100 (Table 2) · [CLI, partly PATCH]

- **Varies:** 10 methods × β ∈ {0.1, 0.3, 0.5} × P ∈ {0.2, 0.4, 0.6, 1.0}
- **Runs:** 120
- **Measures:** highest top-1 personalized accuracy
- **Methods and PFLlib `-algo` strings:**

| Paper method | `-algo` value | Extra flags needed |
|---|---|---|
| FedAvg-P | `FedAvg` | + local fine-tune before eval (§7.2) |
| FedProx-P | `FedProx` | `-mu 0.01` **required**, + fine-tune (§7.2, §7.6) |
| FedPer | `FedPer` | — |
| Ditto | `Ditto` | `-mu`, `-pls` (paper does not specify) |
| LG-FedAvg | `LG-FedAvg` | — (note the hyphen) |
| FedBABU | `FedBABU` | `-fte` (default 10) |
| FedProto | `FedProto` | `-lam` (default 1.0) |
| FedALA | `FedALA` | `-et`, `-s`, `-p` (defaults 1.0 / 80 / 2) |
| FedPAC | `FedPAC` | `-lam` (default 1.0) |
| FedAS (ours) | `FedAS` | — |

Command template (one cell of the grid):

```powershell
cd "C:\Users\abhil\OneDrive\Desktop\Projects\Fed Learning\PFLlib\system"
python main.py -data Cifar100 -ncl 100 -m CNN -algo FedAS `
  -gr 40 -ls 5 -lbs 16 -lr 0.005 -nc 20 -jr 0.6 `
  -dev cpu -go b0.1_P0.6
```

### E2 — Main comparison, Cifar10 (Table 3) · [CLI, partly PATCH]

Identical to E1 with `-data Cifar10 -ncl 10`. **Runs: 120.**

```powershell
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS `
  -gr 40 -ls 5 -lbs 16 -lr 0.005 -nc 20 -jr 0.6 `
  -dev cpu -go b0.1_P0.6
```

### E3 — Accuracy curves vs. global round (Figure 5) · [LOG]

- **Fixed:** P = 0.6, β = 0.1; both Cifar10 and Cifar100
- **Measures:** accuracy at every global round (convergence speed)
- **No extra work:** `rs_test_acc` in the `.h5` file already *is* this curve. Plot it.

Reuse the E1/E2 runs at `β=0.1, P=0.6`, then:

```powershell
python -c "import h5py,numpy as np,matplotlib.pyplot as plt; f=h5py.File('../results/Cifar10_FedAS_b0.1_P0.6_0.h5','r'); plt.plot(np.array(f['rs_test_acc'])); plt.xlabel('global round'); plt.ylabel('test acc'); plt.savefig('curve.png')"
```

### E4 — Ablation on PA and CS (Table 4) · [PATCH]

- **Varies:** 4 module combinations × P ∈ {0.2, 0.4, 0.6, 1.0}
- **Fixed:** Cifar10, β = 0.1
- **Runs:** 16
- **Combinations:** (neither), PA only, CS only, PA+CS
- **Important:** the paper's "neither" row equals the FedPer row of Table 3 exactly
  (88.51 / 89.17 / 89.65 / 89.79), so the ablation baseline **is** plain FedPer.
- **Blocker:** `serveras.py` has no toggles; both modules are always on. Needs §6.3.

Once patched:

```powershell
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -gr 40 -ls 5 -lbs 16 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -go abl_noPA_noCS --no-pa --no-cs
python main.py ... -go abl_PA_only   --no-cs
python main.py ... -go abl_CS_only   --no-pa
python main.py ... -go abl_PA_CS
```

### E5 — FIM validity / parameter substitution (Figure 3a) · [NEW]

- **Procedure:** take a trained client model; replace the top-5% of parameters ranked by
  FIM diagonal magnitude; measure accuracy drop. Repeat with a random 5% as control.
- **Expected result:** top-5% substitution causes catastrophic degradation; random 5%
  causes only minor change → FIM magnitude identifies information-bearing parameters.
- **Blocker:** nothing in PFLlib does this. Standalone script required; it can load
  `PFLlib/models/Cifar10/FedAS_server.pt` and reuse the FIM code in `clientas.py` (lines 60–78).

### E6 — t-FIM growth during training (Figure 3b) · [LOG]

- **Measures:** t-FIM per client per round; expect monotonic rise for online clients and
  persistently lower values for stragglers.
- **Already implemented:** `serveras.py::print_fim_histories()` prints, every round,
  each client's full t-FIM history plus `Avg Sum_T_FIM`.
- **Action:** capture stdout and parse. Run E1/E2 with a transcript:

```powershell
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -gr 40 -ls 5 -lbs 16 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -go fim_trace 2>&1 | Tee-Object -FilePath fedas_fim.log
```

### E7 — Aggregation weight per client (Figure 4a) · [PATCH]

- **Fixed:** straggler portion 0.5 → `-jr 0.5`
- **Measures:** normalized aggregation weight `α'_i` per client; expect online clients to
  receive visibly higher weight than stragglers.
- **Blocker:** `FIM_weight_list` is computed but never printed. One-line patch, §6.2.

### E8 — Alignment manner comparison (Figure 4b) · [PATCH]

- **Variants:** serial vs. parallel alignment (i.e. PA *before* local training vs. alongside
  it), and aligning to the **self/local** prototype vs. the **global** prototype.
- **Expected result:** serial + local-prototype (the paper's choice) wins.
- **Blocker:** requires editing `clientAS.set_parameters` (`clientas.py` lines 136–193)
  and the call order in `serveras.py::train`.

### E9 — Normalization compatibility (Figure 6a) · [PATCH]

- **Variants:** proportional (paper default, Eq. 7), softmax, min-max
- **Expected result:** all three perform comparably → method is robust to the choice.
- **Blocker:** one-line change at the normalization step, §6.2.

### E10a — Decoupling compatibility (Figure 6b) · [NEW]

- **Procedure:** apply FedAS's Parameter Alignment on top of CD2-pFed
  (channel-wise decoupling, Shen et al. CVPR 2022) instead of head/backbone split.
- **Blocker:** CD2-pFed is **not in PFLlib at all**. Would need implementing from scratch.
  Lowest priority.

### E10b — Hyperparameter sensitivity (Figure 7) · [CLI]

- **Varies:** batch size ∈ {8, 16, 32, 64} and local epochs ∈ {1, 2, 5, 10}
- **Constraint:** the paper adjusts `Eg` so that **total update steps stay constant**.
  With `El = 5, Eg = 40` as the reference (200 local-epoch-equivalents):

| `-ls` | `-gr` to hold steps constant |
|---|---|
| 1 | 200 |
| 2 | 100 |
| 5 | 40 |
| 10 | 20 |

  For batch size, steps-per-epoch scale as `1/BS`, so holding update steps constant while
  changing `-lbs` means scaling `-gr` proportionally to `BS` (8 → 20, 16 → 40, 32 → 80, 64 → 160).
- **Runs:** 8 (4 batch sizes + 4 local-epoch settings), no code changes needed.

```powershell
# local-epoch sweep, steps held constant
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -lbs 16 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -ls 1  -gr 200 -go hp_ls1
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -lbs 16 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -ls 2  -gr 100 -go hp_ls2
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -lbs 16 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -ls 5  -gr 40  -go hp_ls5
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -lbs 16 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -ls 10 -gr 20  -go hp_ls10

# batch-size sweep, steps held constant
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -ls 5 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -lbs 8  -gr 20  -go hp_bs8
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -ls 5 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -lbs 16 -gr 40  -go hp_bs16
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -ls 5 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -lbs 32 -gr 80  -go hp_bs32
python main.py -data Cifar10 -ncl 10 -m CNN -algo FedAS -ls 5 -lr 0.005 -nc 20 -jr 0.6 -dev cpu -lbs 64 -gr 160 -go hp_bs64
```

### Run count summary

| Experiment | Runs | Status |
|---|---|---|
| E1 Cifar100 grid | 120 | CLI (8 of 10 methods) |
| E2 Cifar10 grid | 120 | CLI (8 of 10 methods) |
| E3 curves | 0 (reuses E1/E2) | LOG |
| E4 ablation | 16 | PATCH |
| E5 FIM substitution | ~4 | NEW |
| E6 t-FIM trace | 0 (reuses E1/E2) | LOG |
| E7 aggregation weights | 1 | PATCH |
| E8 alignment manner | ~4 | PATCH |
| E9 normalization | 3 | PATCH |
| E10a decoupling | ~2 | NEW |
| E10b hyperparameters | 8 | CLI |
| **Total** | **~278** | |

---

## 6. Required code changes

### 6.1 Fix the numpy seed (do this first — affects every run)

`system/main.py` seeds only PyTorch:

```
68 | torch.manual_seed(0)
```

NumPy is never seeded on the training side, so client selection and straggler draws differ
run to run. The paper fixes both to 0. Add:

```python
np.random.seed(0)
```

### 6.1b Remove the dangling `avg_generalization_metrics` call (**critical** — done)

`serveras.py::train` called `self.avg_generalization_metrics()` at line 114. That method does
not exist anywhere in PFLlib (grep returns only this call site) — it is a leftover from the
upstream author's private code. Because it raised `AttributeError` *after* the training loop
but *before* `self.save_results()` / `self.save_global_model()`, **every FedAS run completed
its full wall-clock time and then wrote no `.h5` and no `.pt`.** The three lines have been
deleted. Verify with a 1-round run that `results/*_FedAS_*.h5` appears.

### 6.2 Log and vary the aggregation weights (E7, E9)

In `serveras.py::aggregate_wrt_fisher` (lines 40–55), the normalization is:

```python
FIM_weight_list = [FIM_value/sum(FIM_weight_list) for FIM_value in FIM_weight_list]
```

**Done.** `aggregate_wrt_fisher` now calls two new helpers:

- **E7** — `print_aggregation_weights()` prints, every round,
  `Agg weights [<norm>] : <cid>:<weight>(raw <t-FIM>) ...` for each uploaded client.
  Every run emits this for free; no dedicated run needed.
- **E9** — `normalize_fim_weights()` selects the normalization via
  `-fasn {proportional,softmax,minmax}` (default `proportional` = the paper's Eq. 7).

Two caveats when interpreting E9:

1. **min-max collapses to proportional whenever a zero-t-FIM client is in the cohort.**
   `(v - min)/(max - min)` with `min = 0` is just `v/max`, which renormalizes to the same
   weights as proportional. Client 17 (single-class shard, t-FIM ≡ 0) is in the cohort
   ~60% of rounds at `-jr 0.6`, so the two rows will often be identical. Verified
   numerically on the smoke run's traces.
2. **softmax on raw t-FIM traces is degenerate.** Traces span 0 → 5568, so
   `exp(v - max)` is one-hot to float precision. The implementation divides the shifted
   values by `mean(|v|)` first, keeping the exponent usable; on the smoke-run traces this
   still assigns 0.88 to the outlier client. That mean-scaling is **our** choice, not the
   paper's — report it alongside any E9 number.

A zero-sum guard falls back to uniform weights and prints a warning, so the degenerate
case cannot produce a divide-by-zero mid-run.

### 6.3 Add PA / CS toggles (E4) — done

- `--no-pa` → `clientAS.set_parameters` takes an early return through the new
  `substitute_base()` helper (plain base substitution, the previously commented-out block).
- `--no-cs` → `serveras.py::train` calls the base `aggregate_parameters()`
  (sample-count weighting) instead of `aggregate_wrt_fisher()`.

Both use `action="store_true"`, **not** `type=bool` — the rest of PFLlib's bool flags are
broken in the usual argparse way (`-ld False` evaluates to `True`). The server prints
`FedAS modules: PA=... CS=... FIM norm=...` at startup so each log states its own config.

**`--no-pa --no-cs` does *not* reduce to PFLlib's FedPer.** The paper's Table 4 row 1 equals
its FedPer row exactly, but that identity does not carry over here, so do **not** treat the
two as a correctness test. Three residual differences survive both toggles:

1. **Distribution scope.** `FedPer` calls the base `send_models()`, which overwrites the
   base of **all** `self.clients` every round. `serveras.py::send_selected_models` overwrites
   only the **selected** clients. At `-jr 0.6` that leaves 8 clients holding a stale local
   base under FedAS but refreshed under FedPer. This is deliberate in FedAS (cf. §7.3) and
   is the dominant difference.
2. **Evaluation timing.** `FedPer` runs `send_models()` → `evaluate()`; `FedAS` runs
   `evaluate()` → `send_selected_models()`. FedPer therefore measures *post*-download and
   FedAS *pre*-download, a one-round offset in the accuracy curve on top of (1).
3. **What is aggregated** (benign). `serverper.py` overrides `receive_models` to upload
   `client.model.base`, so its `global_model` is a bare base module. FedAS inherits the base
   `receive_models`, uploads the full `client.model`, aggregates base **and** head, then
   discards the head at download time. Same effective base update — but FedAS burns work
   aggregating a head nobody ever reads.

Consequence for E4: the PFLlib ablation baseline is *"FedPer with partial base
participation"*, a different algorithm from `-algo FedPer`. Report the four ablation cells
against each other, and do not expect row 1 to match the FedPer row of your own Table 3.
Closing the gap would need a third toggle making `send_selected_models` fall back to
`send_models()` — not done, and arguably not desirable since it changes what FedAS *is*.

### 6.4 Add a 6-layer CNN (optional, for absolute-number comparison)

Add to `system/flcore/trainmodel/models.py` and wire into the `model_str` dispatch in
`main.py` (around line 91).

### 6.5 Fixed straggler cohort — done

Implemented in `serverbase.py`, so **every** algorithm gets it (the paper applies `P`
uniformly across all methods, so patching only FedAS would bias the comparison).

- `Server.__init__` draws the online/straggler split **once** via `np.random.permutation`
  and prints both lists. It is stored as client **ids**, because subclasses call
  `set_clients()` *after* `super().__init__()`, so `self.clients` is still empty at that point.
- `select_clients()` delegates to the new `select_clients_from_cohort()` when enabled.
- Flags: `-sc/--straggler_cohort` (enable) and `-sr/--straggler_rate` (default `0.1`).

**Why stragglers are not simply excluded.** A strict reading of the paper — stragglers never
participate — cannot be right. At `P = 0.2` it would leave 80% of clients never trained,
yet Table 3 reports 89.92% for FedAS. The paper itself says stragglers "seldom participate"
and "communicate and train with the server **less frequently**", not never. So
`-sr` is a per-round participation probability for cohort members:

| setting | behaviour |
|---|---|
| `-sr 0.0` | strict — stragglers never participate (matches Fig. 4a literally) |
| `-sr 0.1` | each straggler joins ~10% of rounds (default; the intended reading) |

Measured at `-nc 20 -jr 0.6 -sc -sr 0.1` over 200 simulated rounds: the 12 online clients
participate every round, cohort size averages 12.78 (range 12–16), and each of the 8
stragglers participates 13–26 times (~7–13%).

> `select_clients_from_cohort()` must also update `self.current_num_join_clients`, because
> `receive_models()` does `random.sample(self.selected_clients, int((1-drop) * self.current_num_join_clients))`
> — a stale value there raises `ValueError: Sample larger than population` once stragglers
> are mixed in and the cohort grows past `num_join_clients`.

**This is what unblocks E7 / Figure 4a.** Without it there is no persistent straggler cohort
for the t-FIM weighting to discriminate against, so the aggregation-weight plot has nothing
to show. Run E7 with `-sc`.

**Decided: `P` means the online-cohort size, not mean participation.** At
`-jr 0.6 -sc -sr 0.1`, 12 of 20 clients are always online and the stragglers add a little on
top, so *mean* participation is 12.78/20 ≈ **64%**, not 60%. This matches the paper, which
defines `P` as the client **online ratio** while saying stragglers participate "less
frequently" — both cannot hold unless total participation sits slightly above `P`. Two
consequences to carry into the write-up:

- `-sc` runs cost roughly **10–25% more wall clock** than the same `-jr` without it, because
  the cohort is 12–16 clients rather than a flat 12. `-sc` and non-`-sc` runs are therefore
  **not runtime-comparable**; don't mix them in a timing table.
- State the `-sr` value next to any reported `P`. `P = 0.6` means something different at
  `-sr 0.0` than at `-sr 0.1`.

### 6.6 FedAvg-P / FedProx-P (E1/E2, 24 of the 240 cells)

Needs a fine-tune-then-evaluate step on the final global model. `-ften` /
`fine_tuning_epoch_new` only applies when `-nnc > 0` (new clients), so it cannot be reused
directly.

---

## 7. Known mismatches — read before comparing numbers

### 7.1 The model is not the paper's
The paper uses a 6-layer CNN. `-m CNN` gives `FedAvgCNN(in_features=3, dim=1600)` =
2 conv + 2 fc = **4 layers**. Absolute accuracies will not match Tables 2/3 regardless of
what else you fix. Relative ordering between methods should still be informative.

### 7.2 FedAvg-P / FedProx-P have no PFLlib equivalent
The "-P" suffix means the global model is fine-tuned locally before evaluation. Not
implemented. Without the fine-tune, those rows measure the *global* model — a different
objective, and will look far worse than the paper's numbers.

### 7.3 `-jr` is not the paper's `P` — fixed by §6.5, but off by default
The paper's stragglers are a **persistent cohort** that seldom participates. Stock
`Server.select_clients()` draws a **fresh random subset every round**, so across 40 rounds
every client participates roughly equally and there is no persistent straggler cohort at
all. Plain `-jr` is therefore *not* `P`.

§6.5 adds `-sc` to fix the cohort once. **It is opt-in, so any run without `-sc` still has
the old semantics** — decide which one you are reporting and keep it consistent across the
whole grid.

Note also that `serveras.py` has *all* clients compute their t-FIM each round while only
selected clients receive the global model and train — see §7.11 for what that costs and
why most of it is wasted.

### 7.4 Learning-rate decay is not mappable
Paper: "learning rate decay value 1e-3". PFLlib offers `-ld True` + `-ldg <gamma>`
(ExponentialLR, default gamma 0.99) — a different decay family. Closest approximation is
`-ld True -ldg 0.99`; document whichever you choose.

### 7.5 Seeding is incomplete
See §6.1. Until patched, repeated runs of the same command give different numbers.

### 7.6 FedProx with default `-mu 0.0` **is** FedAvg
`-mu` defaults to `0.0`, which zeroes the proximal term. The paper does not state its μ.
Pick a value (0.001 / 0.01 / 0.1) and record it, or the FedProx-P row is meaningless.

### 7.7 PFLlib's Parameter Alignment ≠ the paper's Eq. 2–4
The paper aligns the new backbone's **per-sample feature embeddings** to the previous local
backbone's embeddings `H_i` under MSE. `clientAS.set_parameters` (`clientas.py` lines 136–193)
instead computes **per-class mean prototypes** from the local model and aligns the global
backbone to those class means, using a **hardcoded** SGD at lr 0.01 for 1 epoch at batch 16.
That is a class-prototype variant of the method. Also, the `progress` argument is computed
in `send_selected_models` and passed in, then never used. Any result here reproduces a
reimplementation, not the published algorithm.

### 7.8 The paper's β sentence is inverted
§4.2 says "a larger β value represents a more non-iid distribution." This is backwards from
the standard Dirichlet convention *and* from the paper's own results (β = 0.1 yields the
highest personalized accuracy, which is what more label skew produces). PFLlib's `alpha`
follows the standard convention. Set `alpha = β` numerically and ignore the sentence.

### 7.9 Train/test split is PFLlib's, not the paper's
`train_ratio = 0.75` — PFLlib merges the original CIFAR train and test sets, then re-splits
75/25 per client. The paper does not state its split.

### 7.10 At β = 0.1 the generated split contains degenerate shards

From the actual `Cifar10/config.json` (β = 0.1, 20 clients):

- **Client 17 holds exactly one class** — 3,175 samples, all label 1. Its local model
  collapses to predicting that class, so NLL → 0, gradients → 0, and its **t-FIM trace is
  identically 0.0 in every round** (confirmed in the smoke run). Under proportional
  normalization its aggregation weight is therefore **exactly 0** — whenever it is
  selected, its update is silently discarded. This is FedAS working as designed (a client
  with no new information earns no weight), but it means the effective cohort is one client
  smaller than `-jr` implies. Report it; do not "fix" it.
- **Client 19 is the leftover bucket** — 2,651 samples of class 5 plus *exactly one*
  sample of each of the other nine classes. Those singletons are PFLlib's `least_samples`
  guarantee, not Dirichlet mass. The same pattern appears in Cifar100's client 19 (all 100
  classes present, most with a count of 1).
- **Consequence for the headline metric:** with several near-single-class shards,
  personalized test accuracy is trivially near 100% for those clients. The smoke run's
  `Std Test Accuracy: 0.30` reflects this. `Averaged Test Accuracy` is **not**
  difficulty-adjusted — always report the std alongside it. This is inherent to the paper's
  own setup and is why β = 0.1 shows the highest accuracy in Tables 2–3.

### 7.14 On this split, t-FIM measures shard composition, not training progress

**This is the most consequential finding so far, and it partly undercuts the premise of CS.**

Evidence, from the `cohort_check` run at `-jr 0.6 -sc -sr 0.1`. In **round 1** the cohort was
*exactly* the 12 always-online clients — zero stragglers drawn — so all 12 had received
identical training (both rounds, every round). Their aggregation weights should therefore be
similar. Instead they spanned **0.0000 to 0.4207**, a ~400× spread:

| cid | samples | eff. classes | raw t-FIM |
|---|---|---|---|
| 5 | 4264 | 1.89 | 3860.6 |
| 6 | 1440 | 2.79 | 1477.2 |
| 8 | 2974 | 1.71 | 1054.6 |
| 10 | 1164 | 2.32 | 661.1 |
| 13 | 1527 | 2.17 | 534.2 |
| 18 | 5449 | 1.21 | 471.1 |
| 1 | 1814 | 1.83 | 463.9 |
| 14 | 3100 | 1.26 | 318.1 |
| 2 | 1211 | 1.60 | 236.6 |
| 4 | 5128 | 1.03 | 63.7 |
| 19 | 2660 | 1.03 | 36.6 |
| 17 | 3175 | 1.00 | 0.0 |

("eff. classes" = `exp(entropy)` of the client's label distribution.) Correlations against
raw t-FIM: **samples 0.14**, **eff. classes 0.45**, **samples × eff. classes 0.69**.

So t-FIM here is driven by **how many classes a client effectively has**, with a secondary
size term — not by how much training it has had. The mechanism is the one identified in
§7.10: a near-single-class shard (clients 17, 19, 4 all at eff. classes ≈ 1.0) collapses to
confident predictions, so NLL → 0, gradients → 0, t-FIM → 0. Client 5 tops the table
because its 2 classes are genuinely balanced (2832 vs 1432), giving real gradient signal.

**Consequences:**

- **CS on this data is a label-diversity reweighting, not a straggler filter.** It
  systematically suppresses clients with concentrated label distributions whether or not they
  are stragglers, and the effect swamps any straggler signal.
- **E7 / Figure 4a needs a confound control.** Comparing online vs. straggler weights
  across *different* clients will mostly measure their shard composition. Compare the
  **same** client's weight when online vs. when straggling, or report weight against
  eff. classes to show the confound explicitly.
- **t-FIM is a sum, not a mean.** `clientas.py` accumulates `fim_trace_sum` over batches with
  no division, so it carries a dataset-size factor. Empirically size alone correlates only
  0.14 here, so the collapse effect dominates — but normalizing per-sample would at least
  remove one confound. Not implemented; it would change the method and so needs its own flag
  and disclosure.

This does **not** mean FedAS does not work — it may well still beat the baselines. It means
the *stated mechanism* ("stragglers have low t-FIM") is not what is driving the weights in
this reimplementation on this split, so do not claim Figure 4a's conclusion from these runs
without the control above.

#### 7.14a The confound is strong enough to *invert* Figure 4a

Once `print_aggregation_weights` tags clients O/S and reports group means, the round-0
cohort of the `cohort_check` run reads:

```
Agg weight means :  online n=12 mean=0.0359  straggler n=3 mean=0.1899
```

**Stragglers received 5.3× more weight than online clients** — the reverse of the paper's
claim. The cause is visible in the startup shard table: the `np.random.permutation` draw
happened to put the high-diversity shards in the straggler group.

| group | mean eff. classes |
|---|---|
| online (12 clients) | 1.65 |
| straggler (8 clients) | 2.77 |

The straggler cohort has **1.67× the label diversity** of the online cohort, purely by luck
of the seed. Since t-FIM tracks diversity (§7.14), that swamps and reverses the straggler
signal CS is supposed to detect.

**Therefore E7 / Figure 4a is not merely confounded, it is uninterpretable on an unstratified
draw** — the comparison reports the seed, not the method. Fixed in §6.5a.

Note that `eff_classes` is the right statistic to stratify on, not raw class count: client 19
has **10 classes but eff_classes = 1.03**, because nine of them hold exactly one sample
(§7.10). Raw class count would call it the most diverse client in the split; it is
effectively the least.

### 6.5a Stratified cohort splitting — done

`build_straggler_cohort()` now **stratifies the online/straggler split by effective class
count** so the two cohorts are compositionally matched, making the E7 comparison about
participation rather than about shard composition. `--cohort-random` restores the old
unstratified draw (needed only to reproduce §7.14a).

Method: sort clients by `eff_classes`, then take every `(N / n_straggler)`-th one from a
random start (systematic sampling). Both cohorts then span the full diversity range.

> **An obvious-looking alternative is wrong.** Cutting the sorted clients into
> `n_straggler` bins and drawing one straggler from each *seems* equivalent, but when
> `n_straggler > N/2` the bins are uneven — `np.array_split(20, 16)` yields sizes
> `[2,2,2,2,1,1,...]` — so all the leftover online clients come from the four oversized
> bins, which sit at the **low-diversity end** of the sort. Measured at `-jr 0.2`, that
> made stratification *worse* than random (mean gap 0.967 vs 0.495). Systematic sampling
> has no such bias.

Measured |online − straggler| mean `eff_classes` gap over 200 seeds:

| `-jr` | random (mean / max) | stratified (mean / max) |
|---|---|---|
| 0.8 | 0.432 / 1.549 | **0.266 / 0.448** |
| 0.6 | 0.366 / 1.341 | **0.164 / 0.296** |
| 0.4 | 0.368 / 1.173 | **0.164 / 0.296** |
| 0.2 | 0.439 / 1.714 | **0.265 / 0.448** |

Worst-case imbalance drops ~3.8×. At seed 0, `-jr 0.6` the cohorts match exactly
(2.10 vs 2.10 against an overall mean of 2.10).

> The `cohort_check` run in `results/` predates this and used the unstratified draw, so its
> 5.3× inverted weight ratio is the §7.14a artefact, not a baseline. Re-run E7 with the
> current code.

### 7.11 Most of the per-round t-FIM computation is discarded

`clientas.py::train` runs a **full forward + backward pass over the client's entire dataset**
for *every* client each round — the `else` branch handles the non-selected ones. But
`aggregate_wrt_fisher` only reads `self.uploaded_ids`, which `receive_models` builds from the
**selected** clients. So at `-jr 0.6` the t-FIM of 8 of 20 clients is computed and then used
for nothing except the `print_fim_histories()` log line.

That log line is not worthless — it *is* E6 / Figure 3b, and with `-sc` it is also the only
way to observe stragglers' t-FIM staying low. But if you are running the main E1/E2 grid and
do not need the trace, skipping the non-selected clients would cut a large slice of the
~285 s/round. Not implemented; flagged as the cheapest available speedup.

### 7.12 PFLlib's PA is a *class-mean* self-prototype, not the paper's per-sample embeddings

Worth stating precisely, because §7.7 undersells how close it is. Paper Eq. 2 builds
`H_i = {f(θ^{t-1}_i, D_i)}` — one feature vector **per sample**. `clientas.py` averages
features into **one mean prototype per class** and aligns to those. Section 3.4 of the paper
draws the distinction that matters to it — "we are aligning **self**-prototype, instead of
global prototype" — and PFLlib *is* on the self-prototype side. The loss is within-class
variation: class means cannot express the spread the paper's per-sample targets retain.
Same spirit, blunter instrument.

### 7.13 Eq. 8 vs. what the code does

Paper Eq. 8 reads `θ^{t+1} = θ^t + (1/m) Σ α'_i ∆θ^t_i`, which double-counts — `α'` is
already normalized to sum to 1 by Eq. 7, so the extra `1/m` shrinks the update by a factor
of `m`. `aggregate_wrt_fisher` instead forms the straightforward weighted average
`θ^{t+1} = Σ α'_i W_i`. The code's reading is the sensible one; treat the `1/m` in Eq. 8 as
a typo rather than a mismatch to reproduce.

---

## 8. Feasibility on this machine

Measured baseline: MNIST + FedAvg, 20 clients, `-ls 1`, `-lbs 10`, CPU → **~130 s/round**.

**Measured, Cifar10 + FedAS**, β=0.1, `-nc 20 -jr 0.6 -ls 5 -lbs 16`, CPU
→ **~285 s/round** (round 0: 219 s; rounds 1–2: 284 s, 285 s). Best acc after 3 rounds: 72.7%.

So one 40-round FedAS run ≈ **41 × 285 s ≈ 3.2 h** — about 4× better than the original
10–20 min/round guess. FedAvg / FedPer / LG-FedAvg / FedBABU should be *faster* than this,
since FedAS uniquely runs a FIM pass over **all 20** clients every round plus an alignment
pass for the selected ones.

Still, at ~3 h/run the 278-run plan is **~35 days of wall clock**. Not feasible on CPU;
the full E1/E2 grids need a GPU. The reduced plan below is feasible (see next subsection).

### Suggested reduced plan (first pass, CPU)

Cifar10 only, β = 0.1 only, P ∈ {0.6, 1.0}, four methods, `-gr 10`:

```powershell
cd "C:\Users\abhil\OneDrive\Desktop\Projects\Fed Learning\PFLlib\system"
foreach ($algo in @('FedAvg','FedPer','FedALA','FedAS')) {
  foreach ($jr in @(0.6, 1.0)) {
    python main.py -data Cifar10 -ncl 10 -m CNN -algo $algo `
      -gr 10 -ls 5 -lbs 16 -lr 0.005 -nc 20 -jr $jr `
      -dev cpu -go "b0.1_P$jr"
  }
}
```

8 runs. Validates the pipeline end to end and gives a directional result (does FedAS beat
FedPer/FedALA, and does the gap widen as P shrinks — the paper's central claim, where
Δ grows from +0.31 at P=1 to ... see §9). Then move the full grid to a GPU.

---

## 9. Paper's reported numbers (for side-by-side review)

Highest top-1 accuracy (%). Use these to fill in your own column per run.

### Table 2 — Cifar100

| Method | β=.1 P=.2 | .4 | .6 | 1 | β=.3 P=.2 | .4 | .6 | 1 | β=.5 P=.2 | .4 | .6 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg-P | 27.61 | 36.42 | 41.61 | 47.57 | 27.87 | 32.67 | 36.35 | 38.67 | 27.44 | 30.66 | 32.07 | 33.82 |
| FedProx-P | 27.42 | 35.40 | 40.41 | 45.96 | 27.27 | 32.25 | 35.29 | 37.85 | 26.99 | 30.22 | 31.36 | 33.46 |
| FedPer | 44.96 | 45.12 | 44.32 | 44.40 | 34.78 | 34.23 | 33.66 | 33.08 | 29.77 | 28.71 | 27.79 | 27.17 |
| Ditto | 32.27 | 38.19 | 41.90 | 44.48 | 24.90 | 26.89 | 29.40 | 30.91 | 25.71 | 27.39 | 27.58 | 27.92 |
| LG-FedAvg | 42.68 | 44.04 | 46.51 | 46.37 | 29.95 | 31.67 | 33.26 | 33.31 | 24.97 | 26.32 | 26.96 | 27.20 |
| FedBABU | 49.85 | 49.79 | 49.88 | 49.78 | 38.18 | 38.89 | 38.81 | 38.69 | 33.60 | 33.88 | 33.18 | 33.38 |
| FedProto | 26.66 | 44.32 | 49.85 | 51.88 | 20.66 | 33.70 | 36.99 | 38.23 | 17.62 | 28.46 | 31.58 | 32.19 |
| FedALA | 45.52 | 48.40 | 50.67 | 51.06 | 36.54 | 40.41 | 41.20 | 42.08 | 31.19 | 35.07 | 36.85 | 37.15 |
| FedPAC | 45.08 | 48.63 | 50.87 | 52.03 | 34.15 | 37.33 | 38.60 | 39.80 | 29.99 | 33.56 | 34.97 | 35.94 |
| **FedAS** | **52.41** | **55.20** | **55.76** | **56.37** | **40.68** | **43.51** | **44.47** | **44.91** | **35.48** | **38.21** | **38.70** | **39.57** |
| Δ | +6.89 | +5.41 | +4.89 | +4.34 | +4.14 | +3.10 | +3.27 | +2.83 | +4.29 | +3.14 | +1.85 | +2.42 |

### Table 3 — Cifar10

| Method | β=.1 P=.2 | .4 | .6 | 1 | β=.3 P=.2 | .4 | .6 | 1 | β=.5 P=.2 | .4 | .6 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg-P | 56.95 | 67.04 | 78.71 | 85.26 | 57.71 | 68.26 | 73.64 | 76.60 | 63.47 | 68.21 | 72.15 | 73.59 |
| FedProx-P | 56.84 | 66.03 | 78.04 | 84.67 | 57.39 | 67.50 | 73.09 | 76.18 | 62.14 | 67.45 | 71.54 | 72.86 |
| FedPer | 88.51 | 89.17 | 89.65 | 89.79 | 77.83 | 77.39 | 77.24 | 76.99 | 73.71 | 72.70 | 72.51 | 72.17 |
| Ditto | 83.59 | 86.00 | 87.26 | 87.86 | 66.58 | 71.26 | 73.58 | 74.97 | 60.41 | 62.60 | 65.23 | 66.80 |
| LG-FedAvg | 87.86 | 88.52 | 88.66 | 88.95 | 75.40 | 75.94 | 76.87 | 77.13 | 67.29 | 67.77 | 68.48 | 68.47 |
| FedBABU | 88.45 | 88.66 | 88.48 | 88.57 | 78.35 | 78.34 | 78.16 | 78.26 | 73.23 | 72.75 | 72.83 | 72.71 |
| FedProto | 61.17 | 84.18 | 87.47 | 88.30 | 64.12 | 74.13 | 76.49 | 77.88 | 58.58 | 66.47 | 69.22 | 70.55 |
| FedALA | 89.61 | 89.89 | 90.00 | 90.06 | 79.03 | 80.92 | 81.02 | 81.40 | 75.05 | 76.24 | 76.52 | 76.75 |
| FedPAC | 87.48 | 88.36 | 88.60 | 87.08 | 78.36 | 80.52 | 81.55 | 82.04 | 74.98 | 76.47 | 77.23 | 77.89 |
| **FedAS** | **89.92** | **90.60** | **90.81** | **91.03** | **80.23** | **81.58** | **82.01** | **82.12** | **75.93** | **77.29** | **77.35** | **77.97** |
| Δ | +0.31 | +0.71 | +0.81 | +0.97 | +1.20 | +0.66 | +0.46 | +0.08 | +0.95 | +0.82 | +0.12 | +0.08 |

### Table 4 — Ablation (Cifar10, β = 0.1)

| PA | CS | P=0.2 | P=0.4 | P=0.6 | P=1 |
|---|---|---|---|---|---|
| — | — | 88.51 | 89.17 | 89.65 | 89.79 |
| ✓ | — | 89.34 | 89.50 | 89.72 | 89.91 |
| — | ✓ | 89.05 | 89.39 | 89.90 | 89.94 |
| ✓ | ✓ | 89.92 (+0.58) | 90.60 (+1.10) | 90.81 (+0.91) | 91.03 (+1.09) |

---

## 10. Execution checklist

- [x] §6.1 add `np.random.seed(0)` to `system/main.py`
- [x] §6.1b delete the `avg_generalization_metrics` call in `serveras.py` (blocked all result saving)
- [ ] §2 activate `pfllib` and verify the interpreter path
- [ ] §3 set `alpha = 0.1`, generate Cifar10 + Cifar100
- [ ] §8 run the 8-run reduced plan, confirm `.h5` files appear with distinct `-go` tags
- [ ] E6 capture a FedAS log and extract t-FIM histories (free)
- [ ] E3 plot accuracy curves from `rs_test_acc` (free)
- [x] §6.2 patch aggregation-weight logging + selectable normalization → E7, E9
- [x] §6.3 add PA/CS toggles → E4
- [ ] Decide on GPU access before attempting E1/E2 in full
- [x] §6.5 persistent straggler cohort (`-sc`, `-sr`) → makes `-jr` behave like `P`, unblocks E7
- [x] §6.5a stratify the cohort split by `eff_classes` → makes E7 measure participation, not shards
- [ ] §6.4 / §6.6 only if absolute comparison to Tables 2–3 is required
- [ ] E5, E8, E10a last (new code, lowest leverage)
