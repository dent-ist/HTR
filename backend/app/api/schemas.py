"""Request bodies. Geometry is validated against the page in the service layer (it needs the page size)."""

from __future__ import annotations

import unicodedata
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Coord = Annotated[float, Field(allow_inf_nan=False)]
Point = tuple[Coord, Coord]
MAX_TEXT = 2000


def _check_text(v: str) -> str:
    if len(v) > MAX_TEXT:
        raise ValueError(f"text longer than {MAX_TEXT} characters")
    if any(unicodedata.category(c) == "Cc" for c in v):
        raise ValueError("text must be a single line without control characters")
    return v


class DocumentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class LinePatch(BaseModel):
    """Partial update: only the fields present in the JSON are changed (null is not allowed)."""

    model_config = ConfigDict(extra="forbid")
    text: str | None = None
    polygon: list[Point] | None = Field(default=None, max_length=500)
    baseline: list[Point] | None = Field(default=None, max_length=200)
    ignored: bool | None = None
    is_ground_truth: bool | None = None
    index: int | None = Field(default=None, ge=0)

    @field_validator("text")
    @classmethod
    def _text(cls, v: str | None) -> str | None:
        return None if v is None else _check_text(v)

    @model_validator(mode="after")
    def _no_nulls(self):
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be null")
        return self

    def changes(self) -> dict:
        return {name: getattr(self, name) for name in self.model_fields_set}


class LineCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline: list[Point] = Field(max_length=200)
    polygon: list[Point] = Field(max_length=500)
    text: str = ""
    ignored: bool = False
    index: int | None = Field(default=None, ge=0)

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        return _check_text(v)


class SegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    force: bool = False
