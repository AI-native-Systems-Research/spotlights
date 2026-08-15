"""Coordinates the storage tiers (fixture)."""

TIERS = ["fs", "obj", "p2p", "cache"]


def offload(block: bytes) -> str:
    return TIERS[0]
