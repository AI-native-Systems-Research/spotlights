# Running Spotlights on DAM

This is a guide for using a **DAM sandbox that has [Spotlights](../README.md)
pre-installed** — pointing it at a repo, running the pipeline, and getting to a
ranked list of code regions worth optimizing.

If you're a maintainer looking to rebuild or push the image, jump to
[Rebuilding the image](#rebuilding-the-image) at the bottom.

## What the sandbox gives you

When your DAM sandbox is launched from the Spotlights custom image, you get two
tabs and a persistent workspace:

- **Chat tab** — an agent (Claude Code, over ACP) already primed on how to drive
  Spotlights. This is the intended way in: describe what you want to optimize
  and it will run the engine for you, help you shape the objective, and walk
  you through the results.
- **Terminal tab** — a login shell with the CLIs on `$PATH` for when you'd
  rather drive things yourself:
  - `spotlights-engine` — the main pipeline. Subcommands: `doctor`, `init`,
    `prep-evolve`, `apply`.
  - `spotlights-objectives` — objective-setting helper.
  - `claude`, `codex` — the agent CLIs Spotlights orchestrates as subprocesses.
    Don't call model APIs directly; the engine drives these.
  - `uv` — Python package manager used by the engine.
- **Working directory** — `/app/working-dir/` (a.k.a. `~` for the `agent` user)
  is persistent across sandbox restarts. Clone target repos here, put run
  outputs here.

API credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are injected on the wire
by the DAM platform when a credential is attached to your sandbox. You don't
need to configure them; if a run fails with an auth error, the credential
attachment is what to check.

## How to create a Spotlights sandbox

You create the sandbox from the DAM UI's create-sandbox wizard, pointing it at
the Spotlights custom image. The wizard's three steps map onto:

1. **Starting point → Custom image.** Pick the *Custom image* card (tagged
   *Advanced*), and paste the image reference into *Image address*:

   ```
   quay.io/oritp/spotlights-agent:0.1.4
   ```

   That's the current published tag; see
   [Rebuilding the image](#rebuilding-the-image) if you're rolling your own.
2. **Setup your sandbox → Name, Size, Provider, Network.** Give the sandbox
   a name and pick the **largest CPU / memory tier** the wizard offers —
   Spotlights orchestrates `claude` and `codex` subprocesses in parallel and
   the extractor is CPU-hungry, so a small tier throttles the whole pipeline.
   Then for **Provider** pick **IBM LiteLLM ETE Proxy** — the image ships with
   a `~/.claude/settings.json` and `~/.codex/config.toml` that route `claude`
   and `codex` through IBM's LiteLLM endpoint.
3. **Grant connections → credentials.** Attach at minimum `ANTHROPIC_API_KEY`
   and `OPENAI_API_KEY` — these are wire-injected into the sandbox and are
   what the `claude` and `codex` CLIs read at run time. If you already know
   you'll be cloning a private target repo, attach a
   [GitHub personal access token](https://github.com/settings/tokens) here
   too (fine-grained, `Contents: Read`, scoped to just the repo you need);
   otherwise you can add it later from the sandbox's connection settings.

Click *Create sandbox*. First boot takes a minute or two while DAM pulls the
image and seeds `/app/working-dir/` from the image's seed workspace. When it
lands, you get the Chat + Terminal tabs described above and can move on to
the first-boot steps below.

## First moves in a fresh sandbox

Do these once per sandbox, in the terminal tab (or ask the chat agent to do
them for you):

```bash
# 1. Sanity-check the environment.
spotlights-engine doctor

# 2. Install the bundled slash commands into the working directory (project scope).
spotlights-engine init
```

`init` gives you `/spotlights-objective-setting`,
`/spotlights-sort-candidates`, `/spotlights-apply-candidate`, and `/spotlights-share-candidates` inside the
chat tab — they're the main way to interact with results.

Then bring the target repo. Nothing is bundled by default, and how you get it
into the sandbox depends on whether it's public.

**Public repo — plain clone works:**

```bash
git clone <target-repo-url> ~/work/<target-repo>
```

**Private repo — attach GitHub auth to the sandbox.** The sandbox has no
credentials by default. Don't paste tokens into the terminal — they leak into
shell history and can end up embedded in git remote URLs on disk. Instead,
attach a [GitHub personal access token](https://github.com/settings/tokens)
(fine-grained, scoped to just the repo you need, with `Contents: Read`) as a
credential on the sandbox's **connection settings** in the DAM UI — the same
wire-injection mechanism used for `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

Once the credential is attached and the sandbox is restarted, the token is
available to `git` in-environment, and:

```bash
git clone https://github.com/<org>/<target-repo>.git ~/work/<target-repo>
```

works without prompting or embedding secrets.

Treat the token as disposable — narrow its scope, and revoke it when you're
done with the sandbox.

**Or just upload the repo directly.** If cloning is awkward (very private
codebases, no ability to mint a token, air-gapped setups), skip GitHub
entirely and push the repo into the sandbox from your machine. In the DAM UI,
open the **Files** tab on the sandbox and click **Add** to upload the repo's
root folder; drop it under `~/work/<target-repo>` and Spotlights won't know
the difference.

## Shape the objective

Before running the engine, nail down what you're optimizing for. In the chat
tab, run:

```
/spotlights-objective-setting
```

The agent interviews you about the workload, the metric you care about, and
any known constraints, and prints ready-to-paste `--objective` and `--hint`
flags for the engine command below. **Don't skip this** — a vague objective
gives fuzzy candidates, and a run is 30–45 minutes and a few dollars per
module. Sharpen the goal first; run once.

## Running the engine

Minimum viable invocation, using the `--objective` and `--hint` you just
generated. **Launch it as a background daemon** with `nohup … &` so a terminal
disconnect (closed tab, dropped SSH, Chrome refresh) can't SIGHUP the engine
mid-run — a single-module run is long enough that this happens in practice, and
the engine dies with `manifest.json` still reading `"status": "RUNNING"`.

Open a terminal session, and run:

```bash
cd /home/agent/work
mkdir -p ./spotlights-out
nohup spotlights-engine \
  --repo ~/work/<target-repo> \
  --include <target-repo>/<path/to/module> \
  --objective "<paste from /spotlights-objective-setting>" \
  --hint "<paste from /spotlights-objective-setting>" \
  --output-folder ./spotlights-out \
  --artifacts-dir ./artifacts \
  --max-parallel <your choice> \
  --max-parallel-candidates <your choice> \
  --max-cost <your choice> \
  --log-file ./spotlights-out/run.log \
  > ./spotlights-out/nohup.out 2>&1 &
```

Watch every line end in `\` (except the last) — a missing backslash silently
truncates the command at that line, and any flag below it (including
`--max-cost` and `--log-file`) is dropped without warning.

Then:

```bash
tail -f ./spotlights-out/run.log   # follow progress
pgrep -af spotlights-engine        # confirm it's still alive
kill $(pgrep -f spotlights-engine) # stop it (checkpoint is preserved)
```

Or just ask the **chat tab** to monitor it for you — e.g. *"watch the spotlights
run and tell me when it's done or if anything breaks"* — the agent has the same
terminal access and will tail the log, report progress, and flag failures
without you having to keep the terminal open.

- `--include` takes slash-form leaf qualified names — repeat the flag or pass
  multiple values to widen scope. Without `--include` the engine covers the
  whole repo, which is rarely what you want.
- A single-module run takes roughly **30–45 minutes** and a few dollars in API
  cost. The full 8-module vLLM example runs around **$26** at default rates.
- Runs are **resumable** — the on-disk checkpoint tree under `--artifacts-dir`
  is the source of truth. Rerunning the same command picks up where you left
  off; pass `--no-resume` to force a cold run. If a run *was* interrupted, the
  stale `"status": "RUNNING"` in `manifest.json` is expected — resume anyway.

## After a run

Results land under `--output-folder` (e.g. `./spotlights-out/`):

```
spotlights-out/
  index.md                          # repo-level summary
  result.json                       # full structured output
  modules/
    <module>.md                     # module page — candidates table
    <module>/
      <symbol-slug>__cand-*.md      # per-candidate write-up + proposals
```

From the chat tab:

- **`/spotlights-sort-candidates`** — pointed at `./spotlights-out/result.json`,
  ranks candidates by estimated impact and writes
  `./spotlights-out/sorted/sorted_candidates.{md,json}`.
- **`/spotlights-share-candidates`** — pointed at `./spotlights-out/sorted/`,
  packages the top-N candidates as a self-contained ZIP to share.
- **`/spotlights-apply-candidate`** — Implements one candidate as a reviewable patch, in-session, in a throwaway git worktree

From the terminal:

- **`spotlights-engine prep-evolve`** — turns one candidate into a launchable
  "evolve bundle" for external evolvers (skydiscover, coral, nous). Emits
  native config + evaluator scaffold with the perf oracle left as a marked
  `# TODO`; does not run the evolve itself.

### Downloading the results

To pull results off the sandbox, ask the chat agent to package the output
folder — a prompt that reliably works:

> Zip spotlights-out and publish it to the artifact library as a private
> artifact so I can download it from the Files view.

Then, in the DAM UI's **Files** tab, click the ⋯ (three-dot) menu next to the
resulting `.zip` and select **Download**.

## Operating notes

- **A vague objective wastes money.** Runs are expensive; a fuzzy goal gives
  fuzzy candidates. When in doubt, use `/spotlights-objective-setting` to
  sharpen it *before* kicking off the engine.
- **Costs come from the engine's own rate table**
  (`costing/rates.json`), not from the CLIs' reported `total_cost_usd` — that
  reports list price, not your LiteLLM contract rate. `run_manifest.json` in
  the artifacts dir has the authoritative number.
- **When something fails, check the underlying CLI first.** Spotlights just
  orchestrates `claude` and `codex` as subprocesses. Their auth, model
  selection, and LiteLLM routing all live in the CLIs' own config
  (`~/.claude/settings.json`, `~/.codex/config.toml`). A step failure is
  usually a CLI-side issue, not a Spotlights bug.
- **Don't put real secrets in files that ride the workspace.** The workspace
  is persistent but not private in the same way the wire-injected credentials
  are.

## Rebuilding the image

*Only relevant if you're maintaining the DAM image itself — regular sandbox
users can skip this section.*

The image lives at `quay.io/oritp/spotlights-agent:0.1.4` and is built from
[`spotlights-agent/Dockerfile`](spotlights-agent/Dockerfile). Because the
Spotlights repo is private, the Dockerfile installs from the **local working
tree** (`docker build` context = repo root) rather than a `git+https://…` URL.

```bash
cd DAM/spotlights-agent
./build-and-push.sh                       # uses defaults from the script
TAG=0.1.3 ./build-and-push.sh             # override tag
QUAY_USER=my-org TAG=0.1.3 ./build-and-push.sh
```

Auth for the push: either put `QUAY_USER` + `QUAY_ENCRYPTED_PASS` in a repo-root
`.env` (generate the encrypted password in Quay: *Account Settings → Generate
Encrypted Password → Docker Login tab*), or run `docker login quay.io`
interactively first. The script prints the exact `quay.io/…:tag` ref to paste
into DAM's "Custom image" field at the end.

The seed workspace at [`spotlights-agent/workspace/`](spotlights-agent/workspace/)
is copied into `/app/working-dir/` on **first boot only** of a new sandbox.
Edits made inside an existing sandbox stick; changes to the seed only reach
sandboxes launched from a rebuilt image.

## References

- Repo-level [README](../README.md) and [INSTALL](../docs/INSTALL.md) — Spotlights itself.
- [Building a Custom Image - DAM](https://pages.github.ibm.com/dam-agents/docs/guides/custom-image/?h=custom) — DAM platform's own docs for the custom-image flow.
- [spotlights-agent/Dockerfile](spotlights-agent/Dockerfile) — annotated stage-by-stage.
- [spotlights-agent/workspace/CLAUDE.md](spotlights-agent/workspace/CLAUDE.md) — the agent prompt that runs in the chat tab.
