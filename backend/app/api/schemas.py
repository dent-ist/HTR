"""Request bodies. Geometry is validated against the page in the service layer (it needs the page size)."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal

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
    regenerate_polygon: bool | None = None

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
    polygon: list[Point] | None = Field(default=None, max_length=500)  # omitted: generated from the baseline
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


class MergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    line_ids: list[int] = Field(min_length=2, max_length=200)


class SplitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: Coord
    y: Coord


class IgnoreRegionRequest(BaseModel):
    """A rectangle (x, y, w, h) or a polygon (>= 3 points), in working-copy pixels."""

    model_config = ConfigDict(extra="forbid")
    x: Coord | None = None
    y: Coord | None = None
    w: Coord | None = Field(default=None, gt=0)
    h: Coord | None = Field(default=None, gt=0)
    polygon: list[Point] | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _one_form(self):
        rect = [self.x, self.y, self.w, self.h]
        if self.polygon is not None:
            if any(v is not None for v in rect):
                raise ValueError("give either x/y/w/h or polygon, not both")
        elif any(v is None for v in rect):
            raise ValueError("give x, y, w and h, or a polygon")
        return self


class ReorderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    line_ids: list[int] = Field(max_length=2000)


class RestoreLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int | None = None
    baseline: list[Point] = Field(max_length=200)
    polygon: list[Point] = Field(max_length=500)
    text: str = ""
    text_source: Literal["manual", "predicted"] | None = None
    confidence: float | None = None
    # Which model predicted this line and what it predicted (the snapshot review effort is measured against); both are
    # given back exactly as the line had them before the operation that is being undone.
    predicted_by_model_id: int | None = None
    predicted_text: str | None = None
    is_ground_truth: bool = False
    ignored: bool = False

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        return _check_text(v)

    @field_validator("predicted_text")
    @classmethod
    def _predicted(cls, v: str | None) -> str | None:
        return None if v is None else _check_text(v)


class RestoreRequest(BaseModel):
    """Replace the page's lines by a snapshot (undo). Refused with 409 if that would discard typed text, unless
    force is true."""

    model_config = ConfigDict(extra="forbid")
    lines: list[RestoreLine] = Field(max_length=2000)
    force: bool = False


class TrainRequest(BaseModel):
    """Fine-tune job options. Every field is optional; defaults are sized for a 4 GB GPU (see README)."""

    model_config = ConfigDict(extra="forbid")
    epochs: int | None = Field(default=None, ge=1, le=500, description="maximum epochs (early stopping may end sooner)")
    batch_size: int | None = Field(
        default=None, ge=1, le=256, description="lines per step; lower it if the GPU runs out of memory"
    )
    learning_rate: float | None = Field(default=None, gt=0, le=0.1, allow_inf_nan=False)
    base_model_id: int | None = Field(default=None, description="model to start from; default: the active model")
    device: Literal["auto", "cpu", "cuda:0"] | None = None
    patience: int | None = Field(default=None, ge=1, le=100, description="early stopping: epochs without improvement")
    augment: bool | None = None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: int | None = Field(default=None, description="default: the active model")
    page_ids: list[int] | None = Field(default=None, max_length=5000, description="limit to these pages")
    batch_size: int | None = Field(default=None, ge=1, le=256)
    device: Literal["auto", "cpu", "cuda:0"] | None = None


# ---- M6: abbreviations ------------------------------------------------------------------------------------------


class AbbreviationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    written: str = Field(min_length=1, max_length=200)
    expansion: str = Field(min_length=1, max_length=2000)
    notes: str | None = Field(default=None, max_length=2000)
    case_sensitive: bool = False


class AbbreviationPatch(BaseModel):
    """Partial update; `notes: null` clears the notes, the other fields must not be null."""

    model_config = ConfigDict(extra="forbid")
    written: str | None = Field(default=None, max_length=200)
    expansion: str | None = Field(default=None, max_length=2000)
    notes: str | None = Field(default=None, max_length=2000)
    case_sensitive: bool | None = None

    @model_validator(mode="after")
    def _no_nulls(self):
        for name in self.model_fields_set - {"notes"}:
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be null")
        if not self.model_fields_set:
            raise ValueError("nothing to change")
        return self

    def changes(self) -> dict:
        return {k: getattr(self, k) for k in self.model_fields_set}


class ExpandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=20000)
    ambiguous: Literal["raw", "mark"] = "raw"
