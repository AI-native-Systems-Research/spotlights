"""The `Candidate` and `Candidates` schemas.

Stage 1 of the candidate research proposer emits a `Candidates` JSON object as
its final artifact. The shape constraints captured here are the contract: they
are exported via `model_json_schema()` and handed to both subprocess agents so
the same validation runs model-side and orchestrator-side.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Candidate(BaseModel):
    id: str = Field(pattern=r"^cand-\d{4}$")
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    rationale: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _check_range(self) -> Candidate:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class Candidates(BaseModel):
    module_qualified_name: str
    candidates: list[Candidate] = Field(min_length=1)


__all__ = ["Candidate", "Candidates"]
