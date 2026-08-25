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
# Overridable env vars: QUAY_USER, IMAGE, TAG, PLATFORM_BASE_TAG, SPOTLIGHTS_REF.
set -euo pipefail

QUAY_USER="${QUAY_USER:?set QUAY_USER to your quay.io username or org}"
IMAGE="${IMAGE:-spotlights-agent}"
TAG="${TAG:-0.1.0}"
PLATFORM_BASE_TAG="${PLATFORM_BASE_TAG:-latest}"
SPOTLIGHTS_REF="${SPOTLIGHTS_REF:-main}"

REF="quay.io/${QUAY_USER}/${IMAGE}:${TAG}"
CTX="$(cd "$(dirname "$0")" && pwd)"

echo "Building ${REF}"
echo "  platform-base: ${PLATFORM_BASE_TAG}"
echo "  spotlights ref: ${SPOTLIGHTS_REF}"

docker build \
  --platform linux/amd64 \
  --build-arg "PLATFORM_BASE_TAG=${PLATFORM_BASE_TAG}" \
  --build-arg "SPOTLIGHTS_REF=${SPOTLIGHTS_REF}" \
  -t "${REF}" \
  "${CTX}"

echo "Pushing ${REF}"
docker push "${REF}"

echo
echo "Done. In the DAM create-sandbox wizard, pick 'Custom image' and paste:"
echo "  ${REF}"
