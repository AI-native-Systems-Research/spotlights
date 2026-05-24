"""CLI entry points for the objective-setting skill to call."""

from __future__ import annotations

import json
import sys

from spotlights_engine.objectives.core import (
    assemble_proposal,
    build_intent,
    finalize_objective,
)
from spotlights_engine.objectives.schemas import ObjectiveIntent, ObjectiveProposal


def _cmd_build_intent(data: dict) -> str:
    intent = build_intent(
        target_metric=data["target_metric"],
        target_direction=data["target_direction"],
        workload_classes=data["workload_classes"],
        priorities=data.get("priorities", []),
        notes=data.get("notes", ""),
    )
    return intent.model_dump_json(indent=2)


def _cmd_assemble_proposal(data: dict) -> str:
    intent = ObjectiveIntent.model_validate(data["intent"])
    proposal = assemble_proposal(intent)
    return proposal.model_dump_json(indent=2)


def _cmd_finalize(data: dict) -> str:
    proposal = ObjectiveProposal.model_validate(data["proposal"])
    objective = finalize_objective(
        proposal,
        session_id=data["session_id"],
        approved_by=data["approved_by"],
    )
    json_str = objective.model_dump_json(indent=2)
    output_path = data.get("output_path", "objective.json")
    with open(output_path, "w") as f:
        f.write(json_str + "\n")
    return json_str


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python -m spotlights_engine.objectives.cli <command> '<json>'",
            file=sys.stderr,
        )
        print("Commands: build-intent, assemble-proposal, finalize", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    data = json.loads(sys.argv[2])

    commands = {
        "build-intent": _cmd_build_intent,
        "assemble-proposal": _cmd_assemble_proposal,
        "finalize": _cmd_finalize,
    }

    if command not in commands:
        print(f"Unknown command: {command}. Use: {', '.join(commands)}", file=sys.stderr)
        sys.exit(1)

    result = commands[command](data)
    print(result)


if __name__ == "__main__":
    main()
