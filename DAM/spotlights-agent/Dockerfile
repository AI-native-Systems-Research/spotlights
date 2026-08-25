# syntax=docker/dockerfile:1.7
#
# Custom DAM image for Spotlights.
#
# Build context is the REPO ROOT, not this directory — the upstream
# spotlights repo is private, so we install from the local working tree via
# COPY instead of `git+https://…`. Paths in this file are therefore relative
# to the repo root (`DAM/spotlights-agent/harness-chat.sh`, etc.). Root-level
# `.dockerignore` trims the context to what the wheel actually needs.
#
# Build:
#   ./build-and-push.sh    # from DAM/spotlights-agent/ — wires context + -f
#
# Or by hand from the repo root:
#   docker build --platform linux/amd64 \
#     -f DAM/spotlights-agent/Dockerfile \
#     -t quay.io/<you>/spotlights-agent:0.1.0 .
#
# Push:
#   docker login quay.io           # use encrypted password / robot account
#   docker push quay.io/<you>/spotlights-agent:0.1.0

# Pin a specific platform-base tag before shipping. See
# https://quay.io/repository/dam-agents/platform-base?tab=tags for available tags.
ARG PLATFORM_BASE_TAG=latest
FROM quay.io/dam-agents/platform-base:${PLATFORM_BASE_TAG}

# Codex release to bake in. Bump when you want a newer version. Available
# targets are listed at https://github.com/openai/codex/releases (look for
# `codex-package-<target>.tar.gz`).
ARG CODEX_VERSION=0.149.1
ARG CODEX_TARGET=x86_64-unknown-linux-musl

# Claude Code release to bake in. Empty ⇒ resolve `latest` at build time.
# Pin a specific version (e.g. 2.0.14) for reproducibility.
ARG CLAUDE_VERSION=
ARG CLAUDE_PLATFORM=linux-x64

USER root

# ---------------------------------------------------------------------------
# Claude Code binary + ACP chat server.
#
# The `@anthropic-ai/claude-code` npm package's native postinstall SIGSEGVs
# under Docker Desktop's QEMU x86_64 emulation on Apple Silicon (same root
# cause as @openai/codex). Instead we fetch the self-contained binary
# directly from Anthropic's release CDN — same URL scheme claude.ai/install.sh
# uses — and skip its `claude install` bootstrapping step (which just sets
# up shell integration we don't need in the DAM harness).
#
# The pure-JS ACP shim stays on npm; it installs cleanly under emulation.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.npm \
    set -eux; \
    v="${CLAUDE_VERSION:-$(curl -fsSL https://downloads.claude.ai/claude-code-releases/latest)}"; \
    curl -fsSL -o /usr/local/bin/claude \
        "https://downloads.claude.ai/claude-code-releases/${v}/${CLAUDE_PLATFORM}/claude"; \
    chmod 0755 /usr/local/bin/claude; \
    npm install -g @agentclientprotocol/claude-agent-acp; \
    ln -sf "$(npm config get prefix)/bin/claude-agent-acp" /usr/local/bin/claude-agent-acp; \
    chown -R 65532:0 /app; \
    /usr/local/bin/claude --version

# ---------------------------------------------------------------------------
# Codex CLI — manual install.
#
# We can't run the vendor installer (`curl chatgpt.com/codex/install.sh | sh`)
# inside this base under Docker Desktop's QEMU x86_64 emulation on Apple
# Silicon: the installer's internal `tar -xzf codex-package-*.tar.gz` step
# hits ENOSYS on the mkdirat/openat2 syscalls that hardened GNU tar uses,
# and dies with "Cannot mkdir: Function not implemented".
#
# Instead, fetch the same package tarball GitHub publishes and extract it
# with Python's stdlib `tarfile`, which uses the older mkdir/openat syscalls
# QEMU implements. Then replicate the directory layout the installer builds
# under $STANDALONE_ROOT so codex finds its bundled rg, bwrap, and zsh at
# runtime. Both `curl` and `python3` are shipped in the base image.
# ---------------------------------------------------------------------------
RUN set -eux; \
    codex_home=/opt/codex; \
    releases_dir="${codex_home}/packages/standalone/releases"; \
    release_dir="${releases_dir}/${CODEX_VERSION}-${CODEX_TARGET}"; \
    current_link="${codex_home}/packages/standalone/current"; \
    mkdir -p "${release_dir}"; \
    curl -fsSL -o /tmp/codex-package.tar.gz \
        "https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/codex-package-${CODEX_TARGET}.tar.gz"; \
    python3 -c "import sys, tarfile; tf=tarfile.open('/tmp/codex-package.tar.gz'); tf.extractall(sys.argv[1], filter='data'); tf.close()" "${release_dir}"; \
    ln -s bin/codex "${release_dir}/codex"; \
    for f in bin/codex bin/codex-code-mode-host codex-path/rg codex-resources/bwrap; do \
      [ -f "${release_dir}/${f}" ] && chmod 0755 "${release_dir}/${f}" || true; \
    done; \
    ln -sfn "${release_dir}" "${current_link}"; \
    ln -sf "${current_link}/bin/codex" /usr/local/bin/codex; \
    rm -f /tmp/codex-package.tar.gz; \
    chown -R 65532:0 "${codex_home}"; \
    /usr/local/bin/codex --version

# ---------------------------------------------------------------------------
# Spotlights engine.
#
# The upstream repo is private, so we install from the LOCAL working tree in
# the build context (repo root) instead of `git+https://…`. The build context
# is the repo root — see build-and-push.sh — and `.dockerignore` at the root
# limits what actually ships. `uv tool install <path>` builds the wheel and
# copies its entry points under UV_TOOL_BIN_DIR; we symlink them into
# /usr/local/bin (already on the agent user's PATH). The source tree is
# removed after install so it doesn't bloat the final image.
#
# If `uv` isn't already resolvable at build time (the base ships it as a lazy
# mise shim and mise may point at the pod's runtime egress gateway), uncomment
# the bootstrap line below.
# ---------------------------------------------------------------------------
ENV UV_TOOL_DIR=/opt/uv-tool \
    UV_TOOL_BIN_DIR=/opt/uv-tool/bin

# RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
COPY --chown=65532:0 . /tmp/spotlights-src
RUN uv tool install --force /tmp/spotlights-src \
    && for bin in spotlights-engine signal-pipeline spotlights-objectives; do \
         [ -x "${UV_TOOL_BIN_DIR}/${bin}" ] && ln -sf "${UV_TOOL_BIN_DIR}/${bin}" "/usr/local/bin/${bin}" || true; \
       done \
    && chown -R 65532:0 /opt/uv-tool \
    && rm -rf /tmp/spotlights-src

# ---------------------------------------------------------------------------
# Harness scripts — the platform's contract. Two scripts, always.
# Paths are relative to the build context (repo root); see build-and-push.sh.
# ---------------------------------------------------------------------------
COPY DAM/spotlights-agent/harness-chat.sh     /usr/local/bin/harness-chat
COPY DAM/spotlights-agent/harness-terminal.sh /usr/local/bin/harness-terminal
RUN chmod +x /usr/local/bin/harness-chat /usr/local/bin/harness-terminal

# ---------------------------------------------------------------------------
# Seed workspace. Copied once into /home/agent on first boot (persistent PVC);
# do NOT COPY straight into /home/agent — the volume would hide it.
# ---------------------------------------------------------------------------
COPY --chown=65532:0 DAM/spotlights-agent/workspace/ /app/working-dir/

# ---------------------------------------------------------------------------
# Placeholder env vars. Real values are injected on the wire by the platform
# when a credential is attached; the harness only needs the names to exist.
# ---------------------------------------------------------------------------
ENV ANTHROPIC_API_KEY=dummy-placeholder \
    OPENAI_API_KEY=dummy-placeholder

USER 65532
