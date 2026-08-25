# syntax=docker/dockerfile:1.7
#
# Custom DAM image for Spotlights.
#
# Layout expected under this Dockerfile:
#   Dockerfile
#   harness-chat.sh
#   harness-terminal.sh
#   workspace/
#     CLAUDE.md
#
# Build (from this directory):
#   docker build --platform linux/amd64 -t quay.io/<you>/spotlights-agent:0.1.0 .
#
# Push:
#   docker login quay.io           # use encrypted password / robot account
#   docker push quay.io/<you>/spotlights-agent:0.1.0

# Pin a specific platform-base tag before shipping. See
# https://quay.io/repository/dam-agents/platform-base?tab=tags for available tags.
ARG PLATFORM_BASE_TAG=latest
FROM quay.io/dam-agents/platform-base:${PLATFORM_BASE_TAG}

# Which Spotlights ref to install. Any tag / branch / commit works.
ARG SPOTLIGHTS_REF=main

# Codex release to bake in. Bump when you want a newer version. Available
# targets are listed at https://github.com/openai/codex/releases (look for
# `codex-package-<target>.tar.gz`).
ARG CODEX_VERSION=0.149.1
ARG CODEX_TARGET=x86_64-unknown-linux-musl

USER root

# ---------------------------------------------------------------------------
# Claude Code + ACP chat server.
#
# The Anthropic install script drops a single self-contained binary at
# /root/.local/bin/claude — that step survives Docker's QEMU x86_64 emulation
# on Apple Silicon. The pure-JS ACP shim comes from npm.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.npm \
    set -eux; \
    curl -fsSL https://claude.ai/install.sh | bash; \
    npm install -g @agentclientprotocol/claude-agent-acp; \
    export PATH="/root/.local/bin:${PATH}"; \
    install -m 0755 "$(command -v claude)" /usr/local/bin/claude; \
    ln -sf "$(npm config get prefix)/bin/claude-agent-acp" /usr/local/bin/claude-agent-acp; \
    chown -R 65532:0 /app

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
# Follows README quickstart step 1: `uv tool install` from the public repo.
# We redirect uv's tool root to /opt so the install survives, then symlink the
# entry-point wrappers into /usr/local/bin (already on the agent user's PATH).
#
# If `uv` isn't already resolvable at build time (the base ships it as a lazy
# mise shim and mise may point at the pod's runtime egress gateway), uncomment
# the bootstrap line below.
# ---------------------------------------------------------------------------
ENV UV_TOOL_DIR=/opt/uv-tool \
    UV_TOOL_BIN_DIR=/opt/uv-tool/bin

# RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
RUN uv tool install --force \
        "git+https://github.com/AI-native-Systems-Research/spotlights.git@${SPOTLIGHTS_REF}" \
    && for bin in spotlights-engine signal-pipeline spotlights-objectives; do \
         [ -x "${UV_TOOL_BIN_DIR}/${bin}" ] && ln -sf "${UV_TOOL_BIN_DIR}/${bin}" "/usr/local/bin/${bin}" || true; \
       done \
    && chown -R 65532:0 /opt/uv-tool

# ---------------------------------------------------------------------------
# Harness scripts — the platform's contract. Two scripts, always.
# ---------------------------------------------------------------------------
COPY harness-chat.sh     /usr/local/bin/harness-chat
COPY harness-terminal.sh /usr/local/bin/harness-terminal
RUN chmod +x /usr/local/bin/harness-chat /usr/local/bin/harness-terminal

# ---------------------------------------------------------------------------
# Seed workspace. Copied once into /home/agent on first boot (persistent PVC);
# do NOT COPY straight into /home/agent — the volume would hide it.
# ---------------------------------------------------------------------------
COPY --chown=65532:0 workspace/ /app/working-dir/

# ---------------------------------------------------------------------------
# Placeholder env vars. Real values are injected on the wire by the platform
# when a credential is attached; the harness only needs the names to exist.
# ---------------------------------------------------------------------------
ENV ANTHROPIC_API_KEY=dummy-placeholder \
    OPENAI_API_KEY=dummy-placeholder

USER 65532
