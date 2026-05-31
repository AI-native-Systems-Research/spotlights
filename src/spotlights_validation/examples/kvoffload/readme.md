# Running a `validation_plan.json`

This guide explains how to execute a validation plan produced by the MVP
planning step against a target source tree using the `spotlights-validation`
CLI.

## What is `validation_plan.json`?

A `validation_plan.json` is the artifact produced by the planning phase. It
contains an ordered list of harness entries (unit tests, integration tests,
benchmarks) plus per-entry metadata: priority, `halt_on_failure`, the
`invoke` command to run, and any benchmark workloads. The runner walks the
entries in priority order and produces a single pass / fail / conditional
verdict.

A reference plan lives at `src/spotlights_validation/sample/kvoffload/validation_plan.json`.

## Cloning the required repositories

3 repos are needed: 
- `spotlights-validation`: this CLI and the reference
- `vllm`: the target source tree the plan runs against
- `kv-offload-lab`: Contains the proposed kv offloading new policy for vLLM, and benchmark script wrapper

```bash
# 1. spotlights-validation — the CLI and the reference validation plan
git clone git@github.com:AI-native-Systems-Research/spotlights.git
cd spotlights-validation
git checkout mvp_1
cd ..

# 2. vllm — the target source tree the plan runs against
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.18.0
cd ..

# 3. kv-offload-lab — provides the multi-turn KV offload benchmark
#    invoked by the reference plan
git clone <kv-offload-lab-repo>
cd kv-offload-lab
git checkout mvp_1
cd ..
```

The vLLM working install recipe under
[Concrete example: the reference `kvoffload` plan against `vllm`](#concrete-example-the-reference-kvoffload-plan-against-vllm)
pins a different commit (`346cf163a…`, a nightly-channel head with
precompiled wheels). Use whichever pin matches the plan you intend to
run — `v0.18.0` for the released-tag baseline, or the nightly commit if
you need the precompiled-wheel install path.

## Prerequisites

- Python ≥ 3.11
- The target source tree to validate against (e.g. a clone of `vllm`)
- Runtime dependencies the harness commands themselves need — see
  [Installing harness runtime dependencies](#installing-harness-runtime-dependencies).
- Optional: `pytest-json-report` — when installed, entries with
  `output_format: "pytest-json"` produce a structured pass/fail/skip count
  *and* per-failure detail (file, line, message, longrepr) in the result
  JSON. Without it the runner falls back to exit-code-only parsing and
  errors degrade to a single line scraped from stdout.

### Environment preconditions for the reference `kvoffload` plan

Several entries (`tests/v1/kv_connector/unit/test_offloading_connector.py`,
`tests/basic_correctness/`) exercise real model loads and torch-inductor
compile, so a few extra environment variables are required *in addition
to* the venv:

| Var / setting | Why | Failure if missing |
| --- | --- | --- |
| `HF_TOKEN=<token>` | Tests load gated repos (`meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-3-1b-it`). | `huggingface_hub.errors.GatedRepoError: 401 Client Error` raised inside `LLM(...)`. |
| `nvcc` reachable on `PATH` (e.g. `export PATH=/opt/share/cuda-12.8.1/bin:$PATH`) | torch-inductor shells out to `nvcc --version` while compiling mamba/Falcon-H1 graphs. | `torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'` during engine init. |
| `HF_HOME=<path>` + pre-download (optional) | Lets the engine-core child run with `HF_HUB_OFFLINE=1` and avoid live network calls during the run. | Without it the engine retries HTTP on every cold start; intermittent 5xx errors fail tests. |
| `HF_HUB_OFFLINE=1` (after pre-download) | Forces transformers/`huggingface_hub` to resolve every file from the local `HF_HOME` cache. Avoids flaky live HTTP fetches mid-run. | If a needed repo is *not* fully cached, you'll see `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files` from `transformers/utils/hub.py` — even when DNS to huggingface.co works. The message blames the network; the real cause is a missing/incomplete cache entry. |

Recommended bring-up: activate the venv, install the HF CLI
(`pip install -U "huggingface_hub[cli]"`), log in with `HF_TOKEN`
(`hf auth login --token "$HF_TOKEN"`), pre-download both gated repos
into `HF_HOME` (`hf download meta-llama/Llama-3.2-1B-Instruct`,
`hf download google/gemma-3-1b-it`), then export `HF_HUB_OFFLINE=1`
and run the plan.

> ⚠️ **Order matters.** Export `HF_HUB_OFFLINE=1` *after* the `hf
> download` commands, never before. With `HF_HUB_OFFLINE=1` set, `hf
> download` and any in-process `from_pretrained` call will refuse to
> reach the network — the downloads will silently fall back to
> already-cached files (or fail if the repo isn't cached), so the
> "pre-download" step does nothing. If a downstream test then needs a
> file that wasn't cached, you'll hit the `OSError` row above. 

> 💡 **Verifying a cache is complete.** A model directory with only
> `LICENSE.txt` and `README.md` under `<HF_HOME>/hub/models--…/snapshots/<rev>/`
> is the partial-download signature — `hf download` was likely
> interrupted or run while gated access was still pending. Re-run
> `hf download <repo>` (with offline mode unset) to finish populating
> the snapshot before flipping `HF_HUB_OFFLINE=1` back on.

#### Skip-instead-of-token alternative

If you can't obtain the gated tokens, the plan already filters the
gated parametrizations out with `pytest -k`:

- `unit-kv-connector-offloading`: `-k 'not meta-llama and not tiering and not test_request_preemption'`
- `correctness-basic`: `-k 'not meta-llama and not tiering'`

So those two entries pass without any HF token. Other entries that load
gated repos (e.g. `unit-kv-prefetch-offload` for `Llama-3.2-1B-Instruct`)
still require `HF_TOKEN` or a populated `HF_HOME` cache.

#### Deselected upstream-broken tests

The plan currently deselects tests that are broken or environmentally
incompatible and unrelated to the kv-offload change under test:

- `unit-cache-kernels`: `-k 'not test_gather_cache_oob'`. Upstream commit
  `77e10c9ca` (#28029) added a `token_to_seq` Tensor arg to the kernel
  signature but did not update `test_gather_cache_oob`, so the test
  fails with `Expected Tensor for token_to_seq but found int`. The
  entry's `halt_on_failure` is set to `false` and a `_comment` field
  in the JSON records the rationale.
- `correctness-basic`: `-k 'not test_cumem and not test_cpu_offload and
  not Gemma2 and not test_prefetch_offload'` (in addition to the
  `not meta-llama and not tiering` skip). Reasons:
  - `test_cumem` family `os.fork()`s after CUDA is initialized, which
    raises `RuntimeError: Cannot re-initialize CUDA in forked
    subprocess` on this host.
  - `test_cpu_offload` and `test_models[…Gemma2…]` fail with
    `CUDA error: cudaErrorDevicesUnavailable` during engine
    `mem_get_info`. Check `nvidia-smi -q | grep "Compute Mode"` — if
    it reports `Exclusive_Process`, the previous test's CUDA context
    hasn't been released by the time the next test inits, and the
    second context is denied. Re-enable these tests on a host with
    `Default` compute mode.
  - `test_prefetch_offload_llama` loads gated
    `meta-llama/Llama-3.2-1B-Instruct`. It is skipped here because the
    active HF token's gated-access request was still under review. To
    re-enable: get the access grant, then
    `HF_HOME=/u/<user>/models hf download meta-llama/Llama-3.2-1B-Instruct`,
    and remove `not test_prefetch_offload` from the `-k` filter.
- `unit-kv-connector-offloading`: also `-k 'not test_request_preemption'` (in
  addition to `not meta-llama and not tiering`). The
  `test_request_preemption[True]` case hangs indefinitely under this
  serial_pytest invocation: with serial_pytest one case per pytest
  process, `test_offloading_connector[True/False]` complete in ~65s
  each, but `test_request_preemption[True]` never returns and the
  entry's 1800s timeout fires. The `[False]` parametrization is
  excluded too because the `-k` filter matches by base test name. Job
  1225022 (2026-05-26) is the most recent reproduction; full stdout in
  `src/spotlights_validation/sample/kvoffload/validation_job.stdout`. To re-enable: run
  `pytest -v tests/v1/kv_connector/unit/test_offloading_connector.py::test_request_preemption -p no:cacheprovider --timeout=120`
  standalone, capture the stack from `pytest --timeout=…` to identify
  what it is waiting on, then drop the filter once fixed.
- `unit-kv-offload-tiering`: `-k 'not test_cpu_offloading'`. `test_cpu_offloading`
  loads `meta-llama/Llama-3.2-1B-Instruct` inside the test body (not in the
  test name), so `-k 'not meta-llama'` has no effect — the filter must target
  the test function name. Access request to the gated repo is pending review.
  > **Important:** once the HuggingFace access request for
  > `meta-llama/Llama-3.2-1B-Instruct` is approved, download the model
  > (`HF_HOME=<hf-cache>/hub huggingface-cli download meta-llama/Llama-3.2-1B-Instruct`)
  > and revert the `-k` filter in `src/spotlights_validation/sample/kvoffload/validation_plan.json`
  > back to `pytest -v tests/v1/kv_offload/` (no filter).
- `integration-engine`:
  `--ignore=tests/v1/engine/test_async_llm.py
  --ignore=tests/v1/engine/test_engine_core_client.py`. Both files fail
  at *collection* (module import), not at test execution, so a `-k`
  filter is too late — pytest reports `2 errors during collection` and
  the entry's exit code becomes 2 with `summary.total = 0`. Reasons:
  - `test_async_llm.py` evaluates `AsyncEngineArgs(model="Qwen/Qwen2-VL-2B-Instruct")`
    at module load. `Qwen2-VL-2B-Instruct` isn't in the local cache, so
    `HF_HUB_OFFLINE=1` makes `snapshot_download` raise
    `LocalEntryNotFoundError`. Re-enable by `hf download
    Qwen/Qwen2-VL-2B-Instruct` into `HF_HOME` and dropping the
    `--ignore`.
  - `test_engine_core_client.py` calls
    `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")`
    at module load. `transformers.AutoTokenizer` does **not** go
    through vLLM's `arg_utils.get_model_path` shim (the one that
    rewrites `model_id` to `model_path` when `HF_HUB_OFFLINE=1`), so
    even though the snapshot is in `HF_HOME` the tokenizer lookup hits
    `OSError: We couldn't connect to 'https://huggingface.co'`.
    Re-enable by ensuring `HUGGINGFACE_HUB_CACHE` / `HF_HOME` points at
    a fully-populated cache containing the tokenizer files, or running
    with `HF_HUB_OFFLINE=0`.

## Installing harness runtime dependencies

The runner doesn't install anything for you — it just runs each entry's
`invoke` command (e.g. `pytest -v tests/v1/kv_offload/`) inside
`--source-tree`. Whatever those commands need has to already be importable
from the Python interpreter on `PATH` when you launch the CLI.

In practice that means **four layers, all in a single venv**:

1. **The target project itself**, installed editable so its modules are
   importable by the tests.
2. **The target project's test dependencies** (pytest, fixtures, mocks,
   etc.) listed in its requirements files.
3. **`spotlights-validation` itself**, so the `python -m
   spotlights_validation.cli` entry point is on `PATH`.
4. **`pytest-json-report`**, if any plan entry uses
   `output_format: "pytest-json"` and you want structured counts instead of
   exit-code-only parsing.

All four must live in the **same virtualenv**. The CLI uses
`subprocess.run(...)` to invoke each harness command, and the child
process inherits the parent's interpreter and `PATH` — so if `pytest` and
the target project aren't importable from the venv that launched the CLI,
collection will fail.

### Where to put the venv

The default vLLM convention is `<vllm-root>/.venv`. Use that **only if the
filesystem hosting the vLLM clone has enough free quota** — the install
needs ~10 GB (CUDA-enabled torch + triton + test deps). Check first:

```bash
df -h /path/to/vllm
```

If quota is tight (this happened on `<filesystem>`, which is at
100%), put the venv on a roomier filesystem instead. Concrete layout that
worked here:

- vLLM source tree: `<workdir>/vllm` (read/execute only,
  no install artifacts written here)
- venv: `<home>/venvs/vllm` (home filesystem, ~10 GB consumed)
- uv cache: default `~/.cache/uv` — keep it on the **same filesystem as
  the venv** so uv can hardlink wheel files instead of copying them
  (cross-filesystem fallback to copy doubles disk usage and can blow
  through quota mid-install).

Why not co-locate venv with `spotlights-validation`: that side is a small
pure-Python package; the venv "belongs to" the heavy side (vLLM) and
should sit where it has room.

`--source-tree` is independent of where the venv lives — the runner
activates the venv via `PATH` and `cd`s into `--source-tree` for each
harness command.

### Concrete example: the reference `kvoffload` plan against `vllm`

The reference plan's entries all run `pytest …` inside the cloned `vllm`
tree. vLLM mandates `uv` for env management (see
`vllm/AGENTS.md`). From `<workdir>/vllm`:

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install a uv-managed Python (avoids missing system python3.12 headers
#    that break sdist builds like arctic-inference)
uv python install 3.12

# 3. Create the venv on a filesystem with quota headroom (NOT inside
#    <workdir>/vllm if that fs is full)
mkdir -p <home>/venvs
uv venv --python 3.12 <home>/venvs/vllm
source <home>/venvs/vllm/bin/activate

# 4. Install vllm editable, pinned to a commit that has precompiled wheels
#    published on wheels.vllm.ai. Pin the source tree to the same commit so
#    the editable Python matches the wheel's compiled C++ ABI.
cd <workdir>/vllm
git fetch origin
git checkout 346cf163a11b55e069aa3143ae2878967393ddc2
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=346cf163a11b55e069aa3143ae2878967393ddc2 \
  uv pip install -e . --torch-backend=auto

# 5. Install vllm test dependencies (pytest et al.)
uv pip install -r requirements/test/cuda.in   # any platform
# or, on x86_64 with pinned versions:
uv pip install -r requirements/test/cuda.txt

# 6. Install spotlights-validation and pytest-json-report into the same venv
uv pip install -e <workdir>/spotlights-validation \
                  pytest-json-report
```

The pinned commit `346cf163a…` was the head of the `nightly` channel on
`wheels.vllm.ai` at the time of writing. To pick a current one, fetch
`https://wheels.vllm.ai/nightly/cu130/vllm/metadata.json` and read the
commit referenced in the `path` field.

### Smoke checks before running the plan

```bash
# CLI is importable from this venv
python -m spotlights_validation.cli run --help

# pytest can discover the kv_offload tests in the source tree
cd <workdir>/vllm
pytest --collect-only tests/v1/kv_offload/ -q   # expect ~80 tests collected
```

### Troubleshooting common errors

| Symptom | Cause | Fix |
| --- | --- | --- |
| `urllib.error.HTTPError: HTTP Error 404` while fetching `wheels.vllm.ai/<sha>/cu130/vllm/metadata.json` during `uv pip install -e .` | No precompiled wheel published for this main-HEAD commit on your CUDA variant | Pin to a commit from the `nightly` channel via `VLLM_PRECOMPILED_WHEEL_COMMIT=…` and `git checkout` the same commit |
| `Disk quota exceeded (os error 122)` mid-install | Target filesystem is out of quota; uv is also copying instead of hardlinking because cache and venv are on different filesystems | Move venv to a filesystem with quota; keep `~/.cache/uv` on the **same** filesystem as the venv |
| `Failed to hardlink files; falling back to full copy` warning | uv cache and venv on different filesystems | As above — co-locate cache and venv |
| `CMake Error … Could NOT find Python (missing: Python_INCLUDE_DIRS)` while building `arctic-inference` (or any sdist) | System Python lacks `-devel` headers | `uv python install 3.12` and recreate the venv with `uv venv --python <home>/.local/share/uv/python/cpython-3.12-…/bin/python3.12` |

After that, run the validation plan from anywhere with the venv active:

```bash
python -m spotlights_validation.cli run \
    --plan <workdir>/spotlights-validation/src/spotlights_validation/sample/kvoffload/validation_plan.json \
    --source-tree <workdir>/vllm \
    --change-ref "kv-offload-lab@HEAD" \
    --out result.json
```

### Verifying before a full run

Two quick checks save time:

```bash
# (a) Confirm pytest can collect the entry's tests in the source tree
cd <workdir>/vllm
pytest --collect-only tests/v1/kv_offload/ -q

# (b) Dry-run the plan to see every command that would execute
python -m spotlights_validation.cli run \
    --plan src/spotlights_validation/sample/kvoffload/validation_plan.json \
    --source-tree <workdir>/vllm \
    --dry-run
```

### For other target projects

Substitute the project's documented install steps in place of the vLLM
recipe above. The contract is the same: every binary referenced by an
entry's `invoke` (typically `pytest`, sometimes a benchmark script) and
every module those commands import must be available in the active
environment.

## Install

From the repo root:

```bash
pip install -e .
```

This exposes both `python -m spotlights_validation.cli` and the
`spotlights-validation` console script.

## Running the validation

The validation plan is designed to run **twice** — once as a baseline and
once with the evolved policy applied — so that the evolved policy's
benchmark results can be compared against LRU/ARC baselines collected
under the same conditions.

| Script | What it does | Benchmark entries run |
| --- | --- | --- |
| `run_baseline.sh` | Runs on the **original** vLLM 0.18.0 (no code changes) | LRU + ARC (`benchmark-multi-turn-kv-offload-lab-lru`, `-arc`) |
| `run_change_validation.sh` | Applies the **evolved policy** (`manager_freq_init.py`) on top of vLLM 0.18.0, runs validation, then **reverts** `manager.py` back to the original | Evolved (`benchmark-multi-turn-kv-offload-lab`) |

Both scripts run all non-benchmark entries (unit, integration, stress,
correctness) in addition to their respective benchmark subset. Entries
not applicable to a given run are marked `skipped` in the generated plan
file.

### How entry skipping works

Each script uses `jq` to derive a variant plan from the canonical
`validation_plan.json`:

- `run_baseline.sh` → `validation_plan_baseline.json` — sets
  `"skipped": true` on `benchmark-multi-turn-kv-offload-lab` (the evolved
  policy entry).
- `run_change_validation.sh` → `validation_plan_change.json` — sets
  `"skipped": true` on `benchmark-multi-turn-kv-offload-lab-lru` and
  `benchmark-multi-turn-kv-offload-lab-arc`.

The runner respects `entry.skipped == true` and logs a skip reason
without executing the entry.

### Evolved policy application and revert

`run_change_validation.sh` performs three extra steps around the
validation run:

1. **Backup** the original `$ROOT_DIR/kv_offload_lab/backends/labcpu/manager.py`
   to `manager.py.orig`.
2. **Replace** it with the evolved file at
   `$ROOT_DIR/kv-offload-lab/experiments/openevolve/exp9/openevolve_output/fix/manager_freq_init.py`.
3. **Revert** after the run completes — copies `manager.py.orig` back to
   `manager.py` and removes the backup.

This ensures the working tree is left clean regardless of whether the
validation passes or fails.

### Output separation

| Script | Result file | Logs dir |
| --- | --- | --- |
| `run_baseline.sh` | `validation_result_baseline.json` | `logs/baseline/` |
| `run_change_validation.sh` | `validation_result_change.json` | `logs/change/` |

---

## Running via `run_validation.sh` (single combined run)

`run_validation.sh` is the original LSF-submittable wrapper
that activates the venv, exports the HuggingFace / vLLM environment, and
launches `python -m spotlights_validation.cli run` against the reference
`kvoffload` plan. It runs **all** entries (including all three benchmark
policies) in a single pass. Each input can be supplied as a CLI flag, in
the `.env` file, or as an exported environment variable in the calling
shell.

**Precedence (highest first): CLI flag > `.env` file > calling shell
environment.** The script captures CLI values aside, sources `.env` (so
its assignments override anything already exported), then applies the
captured CLI values last — so a `--root-dir` flag always wins over both
a `ROOT_DIR=…` line in `.env` and a `ROOT_DIR=…` exported in the shell
that launched the script.

### Required inputs

| Env var | Flag | Description |
| --- | --- | --- |
| `ROOT_DIR` | `--root-dir` | Parent directory of the `vllm/` and `kv-offload-lab/` clones. Exported so plan entries that shell out (e.g. the multi-turn KV offload benchmark) can resolve `$ROOT_DIR/kv-offload-lab`. |
| `VENV` | `--venv` | Absolute path to the virtualenv to activate. The venv must contain vLLM (editable), its test deps, `spotlights-validation`, and `pytest-json-report` — see [Installing harness runtime dependencies](#installing-harness-runtime-dependencies). |
| `VLLM_NVME_OFFLOAD_PATH` | `--nvme-offload-path` | Host-side NVMe scratch directory used by vLLM's KV offload connector. Must be writable and on a fast local disk. |
| `HF_CACHE_DIR` | `--hf-cache-dir` | HuggingFace cache root. Used as the default for `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` when those aren't explicitly set. |
| `HF_TOKEN` | — (must come from `.env` or the calling shell) | HuggingFace token with access to the gated repos the plan loads (`meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-3-1b-it`). The script fails fast if it isn't set. |

### Optional inputs

| Env var | Flag | Default | Description |
| --- | --- | --- | --- |
| `ARTIFACTS_DIR` | `--artifacts-dir` | Directory containing the script | Where `validation_plan.json`, `validation_result.json`, `validation_job.stdout/stderr`, and `logs/` live. |
| `ENV_FILE` | `--env` | `$ARTIFACTS_DIR/.env` | File sourced into the script's environment before the run. Silently skipped if it doesn't exist *and* the default path was used; if you pass `--env` explicitly and the file is missing, the script aborts. |
| `HF_HOME` | — | `$HF_CACHE_DIR` | HuggingFace cache root used by `huggingface_hub`. |
| `HF_HUB_CACHE` | — | `$HF_CACHE_DIR` | Same, for the hub-cache layout. |
| `TRANSFORMERS_CACHE` | — | `$HF_CACHE_DIR` | Legacy `transformers` cache var; some code paths still consult it. |
| `HF_HUB_OFFLINE` | — | `1` | When `1`, forces all HF lookups to resolve from the local cache. Pre-download every gated repo before flipping this on — see the warning under [Environment preconditions](#environment-preconditions-for-the-reference-kvoffload-plan). |
| `VLLM_LOGGING_LEVEL` | — | `INFO` | Passed through to vLLM. |
| `VLLM_WORKER_MULTIPROC_METHOD` | — | `spawn` | Matches upstream CI; `spawn` lets EngineCore subprocesses release the parent's CUDA context cleanly between parametrized cases under GPU `exclusive_process` mode. |

### Template `.env`

Drop this next to the script as `artifacts/kvoffload/.env` (or point
`--env` at it elsewhere). Replace every `<…>` placeholder with values
appropriate for your host — none of these are committed.

```bash
# --- required ---

# Parent of the vllm/ and kv-offload-lab/ clones
ROOT_DIR=<absolute-path-to-clones-parent>

# Virtualenv with vllm (editable), its test deps, spotlights-validation,
# and pytest-json-report installed into the same interpreter
VENV=<absolute-path-to-venv>

# Host-side NVMe scratch for vLLM KV offload (fast local disk, writable)
VLLM_NVME_OFFLOAD_PATH=<absolute-path-to-nvme-scratch-dir>

# HuggingFace cache root; HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE
# default to this when not set explicitly
HF_CACHE_DIR=<absolute-path-to-hf-cache>

# HuggingFace token with gated-repo access
# (meta-llama/Llama-3.2-1B-Instruct, google/gemma-3-1b-it)
HF_TOKEN=<your-hf-token>

# --- optional overrides (uncomment to change defaults) ---

# ARTIFACTS_DIR=<absolute-path-to-artifacts-dir>
# HF_HOME=<absolute-path>
# HF_HUB_CACHE=<absolute-path>
# TRANSFORMERS_CACHE=<absolute-path>
# HF_HUB_OFFLINE=1
# VLLM_LOGGING_LEVEL=INFO
# VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### Submitting the jobs

The `#BSUB` headers at the top of each script reserve the GPU resources;
submit with `bsub`. The split scripts are submitted independently:

```bash
# Baseline (original vLLM, LRU + ARC benchmarks)
bsub < src/spotlights_validation/examples/kvoffload/run_baseline.sh

# Change validation (evolved policy applied, evolved benchmark)
bsub < src/spotlights_validation/examples/kvoffload/run_change_validation.sh
```

Or submit the combined single-run script if you want all three policies
in one pass (requires the evolved policy to already be applied manually):

```bash
bsub < src/spotlights_validation/examples/kvoffload/run_validation.sh
```

Pick the input style that matches how you prefer
to manage configuration (all three honor the precedence rules above):

#### Option 1 — `.env` file (simplest for repeat runs)

Drop a populated `.env` next to the script (default path:
`<script-dir>/.env`), then submit:

```bash
cd /path/to/spotlights-validation
bsub < src/spotlights_validation/sample/kvoffload/run_validation.sh
```

`bsub` reads the `#BSUB` headers from stdin. The script resolves
`ENV_FILE` to `<script-dir>/.env` regardless of the compute node's `cwd`,
so the file is picked up wherever LSF places the job.

#### Option 2 — CLI flags (override `.env`)

```bash
bsub src/spotlights_validation/sample/kvoffload/run_validation.sh \
    --root-dir            <absolute-path-to-clones-parent> \
    --venv                <absolute-path-to-venv> \
    --nvme-offload-path   <absolute-path-to-nvme-scratch-dir> \
    --hf-cache-dir        <absolute-path-to-hf-cache>
```

`HF_TOKEN` has no flag, so it must still come from `.env` or the
exported shell environment.

> **Caveat:** flags only reach the script when `bsub` invokes it as an
> argument (`bsub script.sh …`), not when the script is redirected on
> stdin (`bsub < script.sh`) — stdin can't carry argv. If your site
> requires the `bsub < …` form, use Option 1 or Option 3 instead.

#### Option 3 — exported shell environment (lowest precedence)

```bash
export ROOT_DIR=…
export VENV=…
export VLLM_NVME_OFFLOAD_PATH=…
export HF_CACHE_DIR=…
export HF_TOKEN=…
bsub < src/spotlights_validation/sample/kvoffload/run_validation.sh
```

LSF forwards the submitting shell's environment to the compute node by
default; both `.env` and CLI flags will still override these if present.

#### Running interactively (no LSF)

The same script works on a GPU host you've grabbed interactively — just
execute it directly. The `#BSUB` lines are comments to bash:

```bash
src/spotlights_validation/sample/kvoffload/run_validation.sh \
    --root-dir            <absolute-path-to-clones-parent> \
    --venv                <absolute-path-to-venv> \
    --nvme-offload-path   <absolute-path-to-nvme-scratch-dir> \
    --hf-cache-dir        <absolute-path-to-hf-cache>
```

#### Watching the job

```bash
bjobs                                                    # job state
bpeek <jobid>                                            # live tail of stdout while queued/running
tail -f src/spotlights_validation/sample/kvoffload/validation_job.stdout        # once the script has redirected
tail -f src/spotlights_validation/sample/kvoffload/validation_job.stderr
```

Stdout and stderr are redirected to
`$ARTIFACTS_DIR/validation_job.stdout` and `validation_job.stderr`; the
full `ValidationResult` lands at `$ARTIFACTS_DIR/validation_result.json`
and per-entry logs under `$ARTIFACTS_DIR/logs/`.

## Run a plan

```bash
python -m spotlights_validation.cli run \
    --plan src/spotlights_validation/sample/kvoffload/validation_plan.json \
    --source-tree ../vllm \
    --change-ref "kv-offload-lab@HEAD" \
    --out result.json
```

### Flags

| Flag | Required | Description |
| --- | --- | --- |
| `--plan` | yes | Path to `validation_plan.json`. |
| `--source-tree` | yes | Root of the target repo. Each entry's `invoke` runs with this as `cwd`. |
| `--change-ref` | no | Free-form label identifying the change under test; stored in the result. |
| `--dry-run` | no | Print the commands that would run without executing them. |
| `--out` | no | Write the full `ValidationResult` JSON to this path. |
| `--timeout-multiplier` | no | Multiplied by each entry's `estimated_duration` to get the subprocess timeout. Default `2.0`. |
| `--logs-dir` | no | Directory for per-entry combined stdout/stderr logs and pytest-json reports. Defaults to `<out-parent>/logs/<UTC-timestamp>` when `--out` is set, else `./logs/<UTC-timestamp>`. |

## What happens during a run

1. Entries are sorted by `priority` (lower runs first).
2. Before each entry the runner prints the resolved `cmd`, `cwd`, and
   `timeout` so you can see exactly what's about to launch.
3. Each entry's `invoke` command is executed via `subprocess.Popen` with
   stdout+stderr merged and **streamed live** to the terminal (each line
   prefixed with `│ `). The combined output is also captured and, if
   `--logs-dir` is set (or its default applies), persisted to
   `<logs-dir>/<harness-id>.log`.
4. For `pytest-json` entries, `--json-report` is appended and the report
   is written to `<logs-dir>/<harness-id>.report.json`; pass/fail/skip
   counts and per-failure detail (nodeid, file, line, message,
   truncated longrepr) are extracted from it. Otherwise the exit code
   determines pass/fail and a single error line is scraped from output.
5. After each entry the runner prints a one-line summary
   (`-> pass/FAIL (passed=… failed=… skipped=… in N.Ns)`) followed by
   each failed test's nodeid, file:line, and first message line.
6. Benchmark entries are run once per workload; their combined output is
   saved to `<logs-dir>/<harness-id>.<workload-id>.log` and capped at
   8 KB in the result JSON.
7. If an entry with `halt_on_failure: true` fails, remaining entries are
   skipped and the run halts.

## Verdicts

The runner emits one of three verdicts:

- **pass** — every test entry passed.
- **fail** — a `halt_on_failure` entry failed, or any `correctness` test
  failed.
- **conditional** — all halt-on-failure checks passed but one or more
  non-critical entries reported failures; `result.conditions` lists the
  scripts to recheck.

Process exit code is `0` for `pass` and `1` otherwise.

## Output

A summary table is printed to stdout, followed by a `Failures:` section
listing each failed test's nodeid, file:line, the saved log path, and the
saved pytest-json report path. When `--out` is provided, a full
`ValidationResult` JSON is written, including:

- `change_ref`, `verdict`, `verdict_reasoning`, `conditions`
- `test_results[]` — per-script `passed`/`failed`/`skipped`/`duration_seconds`,
  plus `log_path` and `json_report_path` pointing at the durable artifacts.
  `errors[]` is a list of structured `TestError` objects:
  `nodeid`, `phase` (setup|call|teardown), `message`, `file`, `lineno`,
  and a truncated `longrepr` (≤ 4 KB) — enough to diagnose a failure
  without re-running.
- `benchmark_results[]` — per-workload captured output (capped at 8 KB;
  full output lives in the per-entry log file)
- `notes` — e.g. skipped entries after a halt, or `dry-run` marker

Per-entry artifacts under `--logs-dir` (default `<out-parent>/logs/<UTC>`):

```
logs/20260524T123407Z/
  correctness-basic.log                    # combined stdout/stderr
  correctness-basic.report.json            # pytest-json-report
  unit-cache-kernels.log
  unit-cache-kernels.report.json
  unit-kv-connector-offloading.log
  unit-kv-connector-offloading.report.json
  ...
```

These survive the run, so post-mortem grepping (`grep -n Traceback …log`)
works even days later without re-executing the plan.

## Example: dry run

To preview what would run without executing anything:

```bash
python -m spotlights_validation.cli run \
    --plan src/spotlights_validation/sample/kvoffload/validation_plan.json \
    --source-tree ../vllm \
    --dry-run
```
