#!/usr/bin/env bash
# Build and push the Spotlights DAM image to Quay.
#
# Prereqs (one-time per machine):
#   docker login quay.io    # use a robot account or encrypted CLI password,
#                           # NOT your web password.
#
# Usage:
#   ./build-and-push.sh              # uses defaults below
#   TAG=0.1.1 ./build-and-push.sh    # override tag
#
# Overridable env vars: QUAY_USER, IMAGE, TAG, PLATFORM_BASE_TAG.
#
# The image installs Spotlights from the local checkout (the repo root, which
# is the docker build context). The upstream repo is private, so cloning it
# during `docker build` doesn't work — bake the local tree in instead.
set -euo pipefail

QUAY_USER="${QUAY_USER:?set QUAY_USER to your quay.io username or org}"
IMAGE="${IMAGE:-spotlights-agent}"
TAG="${TAG:-0.1.0}"
PLATFORM_BASE_TAG="${PLATFORM_BASE_TAG:-latest}"

REF="quay.io/${QUAY_USER}/${IMAGE}:${TAG}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Build context = repo root (two levels up from DAM/spotlights-agent/).
CTX="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

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
