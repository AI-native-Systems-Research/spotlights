#!/usr/bin/env bash
# Build and push the Spotlights DAM image to Quay.
#
# Auth: if the repo root `.env` file (or your ambient env) sets
# QUAY_ENCRIPTED_PASS, this script does a non-interactive
# `docker login quay.io` for you before push. Generate that value in the
# Quay UI: username → Account Settings → Generate Encrypted Password →
# "Docker Login" tab (or use a robot-account token — same variable).
# Otherwise, `docker login quay.io` beforehand still works.
#
# Usage:
#   ./build-and-push.sh              # uses defaults below
#   TAG=0.1.1 ./build-and-push.sh    # override tag
#
# Overridable env vars: QUAY_USER, IMAGE, TAG, PLATFORM_BASE_TAG,
#                       QUAY_ENCRIPTED_PASS.
#
# The image installs Spotlights from the local checkout (the repo root, which
# is the docker build context). The upstream repo is private, so cloning it
# during `docker build` doesn't work — bake the local tree in instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Build context = repo root (two levels up from DAM/spotlights-agent/).
CTX="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

# Load ${CTX}/.env so credentials (QUAY_ENCRIPTED_PASS, etc.) don't have
# to live in the interactive shell. Anything set in the environment
# already wins; .env just fills in the blanks.
ENV_FILE="${CTX}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

QUAY_USER="${QUAY_USER:?set QUAY_USER to your quay.io username or org}"
IMAGE="${IMAGE:-spotlights-agent}"
TAG="${TAG:-0.1.3}"
PLATFORM_BASE_TAG="${PLATFORM_BASE_TAG:-latest}"

REF="quay.io/${QUAY_USER}/${IMAGE}:${TAG}"

# Log in early so both the base-image pull (private DAM registry) and the
# push work. `--password-stdin` keeps the secret out of `ps`.
if [[ -n "${QUAY_ENCRIPTED_PASS:-}" ]]; then
  echo "Logging in to quay.io as ${QUAY_USER}"
  printf '%s' "${QUAY_ENCRIPTED_PASS}" \
    | docker login quay.io -u "${QUAY_USER}" --password-stdin
fi

echo "Building ${REF}"
echo "  platform-base: ${PLATFORM_BASE_TAG}"
echo "  build context: ${CTX}"

docker build \
  --platform linux/amd64 \
  --build-arg "PLATFORM_BASE_TAG=${PLATFORM_BASE_TAG}" \
  -f "${DOCKERFILE}" \
  -t "${REF}" \
  "${CTX}"

echo "Pushing ${REF}"
docker push "${REF}"

echo
echo "Done. In the DAM create-sandbox wizard, pick 'Custom image' and paste:"
echo "  ${REF}"
