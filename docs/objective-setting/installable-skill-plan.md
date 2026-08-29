# Plan: Ship `/spotlights-objective-setting` as an installable skill

## Goal

Make the objective-setting skill — currently usable only when working inside this checkout — installable for any user who runs `pip install spotlights-engine`. Mirror the spec-kit approach: canonical skill markdown lives in the repo's `templates/commands/`, gets bundled into the wheel, and is copied into the user's `.claude/commands/` by an installer subcommand. Rename the skill from `/objective-setting` to `/spotlights-objective-setting` in the process.

## Reference: how spec-kit does it

- Canonical skills live at [`templates/commands/*.md`](https://github.com/github/spec-kit/tree/main/templates/commands), one file per command, named **without** the `speckit-` prefix (`plan.md`, `specify.md`, etc.).
- The `speckit-` prefix is applied at install time by the integration layer in [`src/specify_cli/integrations/`](https://github.com/github/spec-kit/tree/main/src/specify_cli/integrations).
- Users install via `uv tool install specify-cli ...` and then run `specify init my-project --integration claude`. `specify init` calls the chosen integration's `setup()`, which copies `templates/commands/*.md` into `.claude/commands/speckit-<name>.md`.
- A `speckit.manifest.json` is written so subsequent refresh commands can update files without clobbering user edits.
- Spec-kit's own repo does **not** check in a `.claude/commands/` directory. Contributors dogfood by running `specify init` against their working copy.

## Target end-user flow

```bash
uv tool install spotlights-engine            # or pip install
cd <their-project>
spotlights-engine init                        # one-time, copies skills into .claude/commands/
claude                                        # /spotlights-objective-setting now exists
```

## Target contributor flow (this repo)

```bash
uv sync
spotlights-engine init                        # generates .claude/commands/spotlights-objective-setting.md
                                              # locally — file is gitignored
claude                                        # iterate on the skill via /spotlights-objective-setting
# edit templates/commands/objective-setting.md, then `spotlights-engine init --force` to refresh
```

## Required changes

### 1. Repo layout — `templates/commands/` at repo root (mirrors spec-kit)

- Create `templates/commands/objective-setting.md` as the canonical source. Move the body of the dev-mode `.claude/commands/objective-setting.md` here (without the `spotlights-` prefix in the filename — the prefix is applied at install).
- Delete the dev-mode `.claude/commands/objective-setting.md`.
- Add `.claude/` and `.spotlights/` to [.gitignore](../../.gitignore) so contributor-generated copies don't leak back into git.

### 2. Skill body fixes

In `templates/commands/objective-setting.md`, replace the dev-mode invocation:

```bash
PYTHONPATH=src .venv/bin/python -m spotlights_engine.objectives.cli build-intent '<json>'
```

with a stable console-script invocation:

```bash
spotlights-objectives build-intent '<json>'
```

Also drop any other references to `.venv/` or `PYTHONPATH=src` from the skill body.

### 3. New console script for the objectives CLI

In [pyproject.toml](../../pyproject.toml), add to `[project.scripts]`:

```toml
spotlights-objectives = "spotlights_engine.objectives.cli:main"
```

Verify [src/spotlights_engine/objectives/cli.py](../../src/spotlights_engine/objectives/cli.py) exposes a zero-arg `main()` suitable as an entry point. If it currently uses `if __name__ == "__main__":` only, factor the entry into a `main()` function.

### 4. Bundle templates for both wheel and editable installs

Keep templates at repo root (`templates/commands/`, mirrors spec-kit) and use a two-pronged lookup:

**Wheels** — bundle `templates/` into the package via `[tool.hatch.build.targets.wheel.force-include]`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"templates" = "spotlights_engine/_templates"
```

The wheel ships `spotlights_engine/_templates/commands/objective-setting.md` and the installer resolves it via `importlib.resources.files("spotlights_engine") / "_templates" / "commands"`.

**Editable installs (`uv sync` / `pip install -e`)** — `force-include` doesn't fire here, so `_bundled_templates()` falls back to walking up from `__file__` until it finds `<parent>/templates/commands/`. From `src/spotlights_engine/init_skills.py` that's the repo root.

This gives spec-kit-style repo layout without breaking the contributor's editable-install workflow. Verified end-to-end with both `uv run spotlights-engine init` (editable) and a `uv build` + clean-venv install (wheel).

### 5. New `init` subcommand on `spotlights-engine`

Extend [src/spotlights_engine/cli.py](../../src/spotlights_engine/cli.py) (or add a new sub-CLI) with an `init` subcommand:

```bash
spotlights-engine init [--scope project|user] [--force]
```

Behavior:
- Locate bundled templates via `importlib.resources.files("spotlights_engine") / "_templates" / "commands"`.
- Resolve destination:
  - `--scope user` (default): `~/.claude/commands/`
  - `--scope project`: `<cwd>/.claude/commands/`
- For each `<name>.md` in the bundle, write `<dest>/spotlights-<name>.md` (apply prefix at install time, like spec-kit).
- After writing each file, compute its sha256 and record it in the manifest.
- Write `<scope-root>/.spotlights/manifest.json` with the shape below.
- Default behavior: skip files that already exist (don't overwrite).
- `--force` behavior: overwrite every file the bundle ships, discarding local edits
  to those files. This is spec-kit's `force=True` mode, and the manifest is not
  consulted — "force install this version" means exactly that. (Spec-kit's
  hash-compare-and-preserve behavior lives on a *separate* `refresh_managed` mode
  used by its migration paths, not on `--force`. Conflating the two is what made an
  early version of our installer treat `--force` as a no-op on any edited file.)
- Files the bundle does not ship are never written, at any force level.
- Print a one-line summary: `installed: spotlights-objective-setting`.

**Manifest shape** (`.spotlights/manifest.json`, modeled on spec-kit's `speckit.manifest.json`):

```json
{
  "integration": "spotlights",
  "version": "0.1.0",
  "installed_at": "2026-06-07T12:34:56+00:00",
  "files": {
    ".claude/commands/spotlights-objective-setting.md": "<sha256-hex>"
  }
}
```

`.spotlights/manifest.json` is created by `spotlights-engine init` and only by `spotlights-engine init`. It never appears as a side effect of `pip install` and never silently — if the user never runs `init`, the directory never exists. Same model as spec-kit's `.specify/`.

### 6. Update [.gitignore](../../.gitignore)

Add:

```
.claude/
.spotlights/
```

(or, if we want to allow project-specific Claude config in this repo later, scope to `.claude/commands/spotlights-*.md` and `.spotlights/manifest.json`.)

### 7. Tests

In [tests/](../../tests/):
- Unit test for `spotlights-engine init`: writes to a tmp dir, applies the `spotlights-` prefix, is idempotent without `--force`, overwrites with `--force`, writes a manifest.
- Build test: after `hatch build`, the wheel contains `spotlights_engine/_templates/commands/objective-setting.md`. (Can be a tox/pytest job that runs `python -m build` in a tmp dir and inspects the wheel.)
- Smoke test for the new console script: `spotlights-objectives build-intent '<minimal valid json>'` exits 0 and emits valid JSON.

### 8. Docs and README updates

- [README.md](../../README.md): add an "Install the Spotlights skill" section under the existing claude-CLI instructions, showing the `uv tool install` → `spotlights-engine init` → `claude` flow.
- [docs/objective_setting.md](../objective_setting.md): replace the `/objective-setting` reference with `/spotlights-objective-setting` and update the "How to use" section to show the install steps for an external user.
- This plan doc itself stays under `docs/objective-setting/` for reference.
- Grep the repo for any remaining `/objective-setting` slash-command references and update them.

## Decisions

1. **Templates location**: repo-root `templates/commands/` (mirrors spec-kit). Wheel builds bundle them into the package via `[tool.hatch.build.targets.wheel.force-include]`. The installer's `_bundled_templates()` checks the package-data path first, then falls back to a `__file__`-relative walk up to `<repo>/templates/commands/` for editable installs. Both flows tested end-to-end.
2. **`init` scope default**: `user` (`~/.claude/`) — the skills are engine-versioned, not project-specific, so installing them once per machine matches how people actually use them. (This originally shipped as `project`; the default was flipped later.) The package install (`pip install spotlights-engine`) is the same regardless; `--scope project` pins the commands to a single checkout.
3. **Manifest format**: spec-kit shape — `{integration, version, installed_at, files: {path: sha256}}`. It is a *record*, never an input to the overwrite decision: it says which files this integration put on disk, at which version, so a future `uninstall-skills` knows what it owns and a `doctor` can tell a pristine install from an edited one. `--force` does not consult it — see Behavior above. (Spec-kit's hash-compare-and-preserve lives on its separate `refresh_managed` mode; reading the hash into `--force` is exactly the conflation that broke an early version of this installer.)

## Out of scope (explicit non-goals)

- Claude Code plugin packaging (`/plugin install spotlights`). Worth doing once we have ≥2 skills to ship; not yet.
- Multi-agent integrations (Gemini, Copilot, etc.). Spec-kit supports many; we ship Claude Code only for v0.
- Auto-update / refresh of installed skills. `init --force` is the manual mechanism for now.

## Suggested execution order

1. Create `src/spotlights_engine/_templates/commands/objective-setting.md` (move + edit body to use `spotlights-objectives` console script).
2. Add `[project.scripts]` entry for `spotlights-objectives`; ensure `objectives/cli.py` has a `main()`. Smoke-test it.
3. Implement `spotlights-engine init` subcommand. Unit-test it.
4. Delete `.claude/commands/objective-setting.md`; add `.claude/` and `.spotlights/` to `.gitignore`. Run `spotlights-engine init` in this repo and confirm `/spotlights-objective-setting` works in `claude`.
5. Update README + `docs/objective_setting.md`. Grep for stale `/objective-setting` references.
6. Open PR.
