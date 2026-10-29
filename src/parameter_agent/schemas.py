from typing import Annotated, Literal

from pydantic import AfterValidator, Field, field_validator

from .domain import Identifier, ProcessContext, StrictModel


def _nonblank(value):
    if not value.strip():
        raise ValueError("A nonblank note is required")
    return value


Note = Annotated[str, Field(min_length=1, max_length=2000), AfterValidator(_nonblank)]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ContextRequest(StrictModel):
    action_id: Identifier
    context: ProcessContext


class CreateTaskRequest(StrictModel):
    action_id: Identifier
    context_id: Identifier


class ActionRequest(StrictModel):
    action_id: Identifier


class SupplementRequest(ActionRequest):
    fields: dict[str, float]
    note: Note

    @field_validator("fields")
    @classmethod
    def permitted_measurements(cls, value):
        bounds = {"material_viscosity": (0.3, 2.5), "ambient_c": (5.0, 45.0)}
        if not value or set(value) - set(bounds):
            raise ValueError("Supply only missing measurement fields")
        if any(not bounds[key][0] <= number <= bounds[key][1] for key, number in value.items()):
            raise ValueError("Supplement is outside the permitted range")
        return value


class ReviewRequest(ActionRequest):
    recommendation_hash: Hash
    decision: Literal["approved", "rejected"]
    note: Note


class ExecuteRequest(ActionRequest):
    recommendation_hash: Hash
    confirmed: Literal[True]

    @field_validator("confirmed", mode="before")
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError("Explicit boolean confirmation is required")
        return value


class ReconcileRequest(ActionRequest):
    call_id: Identifier
    no_result: Literal[True]
    tokens: int | None = Field(default=None, ge=0, le=100000)
    note: Note

    @field_validator("no_result", mode="before")
    @classmethod
    def explicit_attestation(cls, value):
        if value is not True:
            raise ValueError("Explicit boolean attestation is required")
        return value
