from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

def nonblank(value):
    if not value.strip():
        raise ValueError("Blank text")
    return value


# Preserve source text and IDs exactly; normalization is confined to retrieval.
Text = Annotated[str, StringConstraints(min_length=1), AfterValidator(nonblank)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Block(StrictModel):
    block_id: Text
    page: Annotated[int, Field(ge=1)]
    section: Text
    text: Text


class Option(StrictModel):
    id: Text
    text: Text


class Question(StrictModel):
    id: Text
    question: Text
    options: Annotated[list[Option], Field(min_length=4, max_length=6)]

    @model_validator(mode="after")
    def unique_options(self):
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("Duplicate option IDs")
        return self


class AnswerBody(StrictModel):
    answer_ids: list[Text]
    abstain: bool
    rationale: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]
    citations: Annotated[list[Text], Field(max_length=4)]

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.answer_ids)) != len(self.answer_ids):
            raise ValueError("Duplicate answer IDs")
        if len(set(self.citations)) != len(self.citations):
            raise ValueError("Duplicate citations")
        if self.abstain and (self.answer_ids or self.citations):
            raise ValueError("Abstention requires empty answer_ids and citations")
        if not self.abstain and (not self.answer_ids or not self.citations):
            raise ValueError("An evidence-grounded answer requires answers and citations")
        return self


class Answer(AnswerBody):
    id: Text


class ClosedBookBody(StrictModel):
    answer_ids: list[Text]
    abstain: bool
    rationale: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]
    citations: list[Text]

    @model_validator(mode="after")
    def consistent(self):
        if self.citations or len(set(self.answer_ids)) != len(self.answer_ids):
            raise ValueError("Closed-book answers require no citations and unique options")
        if self.abstain != (not self.answer_ids):
            raise ValueError("Inconsistent abstention")
        return self


class ClosedBookAnswer(ClosedBookBody):
    id: Text


CATEGORIES = ("conditions", "comparison", "calculation", "insufficient")


class Gold(StrictModel):
    id: Text
    category: Literal["conditions", "comparison", "calculation", "insufficient"]
    answer_ids: list[Text]
    abstain: bool
    evidence_sets: list[list[Text]]
    rationale: Text
    missing_information: str = ""

    @model_validator(mode="after")
    def consistent(self):
        if self.abstain:
            if self.answer_ids or self.evidence_sets or not self.missing_information:
                raise ValueError("Abstention gold needs missing_information and empty lists")
            if self.category != "insufficient":
                raise ValueError("Abstention category must be insufficient")
        elif not self.answer_ids or not self.evidence_sets:
            raise ValueError("Solvable gold requires answers and evidence sets")
        elif self.category == "insufficient":
            raise ValueError("Insufficient category must abstain")
        if len(set(self.answer_ids)) != len(self.answer_ids):
            raise ValueError("Duplicate gold options")
        for evidence in self.evidence_sets:
            if not 1 <= len(evidence) <= 4 or len(set(evidence)) != len(evidence):
                raise ValueError("Gold evidence sets must have 1..4 unique source IDs")
        return self


def validate_references(body, question: Question, shown_ids: set[str]) -> None:
    if not set(body.answer_ids) <= {o.id for o in question.options}:
        raise ValueError("Unknown option ID")
    if not set(body.citations) <= shown_ids:
        raise ValueError("Citation refers to a block not shown to the model")

