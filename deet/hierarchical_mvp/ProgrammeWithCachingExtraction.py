"""Three-pass programme extraction with a shared report-containing prompt prefix."""

# ruff: noqa: N999

from __future__ import annotations

from types import GenericAlias
from typing import TYPE_CHECKING, Literal

import dspy
from pydantic import BaseModel, Field, create_model

from .programme_framework import clean_outcome_classifications, load_indicator_framework
from .ProgrammeWithCachingModel import (
    PROGRAMME_WITH_CACHING_INSTRUCTIONS,
    ProgrammeWithCaching,
    ProgrammeWithCachingStep,
)

if TYPE_CHECKING:
    from pathlib import Path

ExtractionStage = Literal["programme", "quantitative", "qualitative"]


class ExtractProgrammeWithCaching(dspy.Signature):
    """Use a fixed schema and place changing inputs after the report and framework."""

    __doc__ = PROGRAMME_WITH_CACHING_INSTRUCTIONS

    context: str = dspy.InputField(desc="UN programme evaluation report text.")
    classification_framework: str = dspy.InputField(
        desc="CSV category IDs, groups, statements and inclusion/exclusion criteria."
    )
    extraction_stage: ExtractionStage = dspy.InputField(
        desc="The single extraction stage to perform."
    )
    previous_extraction: ProgrammeWithCachingStep = dspy.InputField(
        desc="Previously extracted information; all fields empty in the first stage."
    )
    extraction: ProgrammeWithCachingStep = dspy.OutputField(
        desc="Only the requested stage's results in the shared response envelope."
    )


class ProgrammeWithCachingExtractionPipeline(dspy.Module):
    """Keep instructions, schemas, report and framework identical across stages."""

    programme_model: type[BaseModel] = ProgrammeWithCaching
    step_model: type[ProgrammeWithCachingStep] = ProgrammeWithCachingStep
    extraction_signature: type[dspy.Signature] = ExtractProgrammeWithCaching
    extract_programme_info: dspy.Predict
    extract_quantitative: dspy.Predict
    extract_qualitative: dspy.Predict

    def __init__(self, classification_framework_path: Path | None = None) -> None:
        """Configure the same signature for each of the three named predictors."""
        super().__init__()
        self.classification_framework = load_indicator_framework(
            classification_framework_path
        )
        # Distinct names preserve stage attribution without changing rendered prompts.
        for attribute, signature_name in (
            ("extract_programme_info", "ExtractProgrammeWithCachingInfo"),
            ("extract_quantitative", "ExtractProgrammeWithCachingQuantitative"),
            ("extract_qualitative", "ExtractProgrammeWithCachingQualitative"),
        ):
            signature = type(
                signature_name,
                (self.extraction_signature,),
                {
                    "__doc__": self.extraction_signature.instructions,
                    "__module__": __name__,
                },
            )
            setattr(self, attribute, dspy.Predict(signature))

    def _extract_stage(
        self,
        predictor: dspy.Predict,
        stage: ExtractionStage,
        context: str,
        previous: ProgrammeWithCachingStep,
    ) -> ProgrammeWithCachingStep:
        prediction = predictor(
            context=context,
            classification_framework=self.classification_framework,
            extraction_stage=stage,
            previous_extraction=previous,
        )
        result = self.step_model.model_validate(prediction.extraction)
        for field in self.step_model.model_fields:
            expected = (
                field in {"programme_characteristics", "interventions"}
                if stage == "programme"
                else field == f"{stage}_outcomes"
            )
            if not expected and getattr(result, field):
                message = f"The {stage} stage returned unexpected {field}."
                raise ValueError(message)
        if stage == "programme" and result.programme_characteristics is None:
            message = "The programme stage must return programme_characteristics."
            raise ValueError(message)
        return result

    def forward(self, context: str) -> BaseModel:
        """Extract each stage, cleaning numerical results before qualitative use."""
        empty = self.step_model(
            programme_characteristics=None,
            interventions=[],
            quantitative_outcomes=[],
            qualitative_outcomes=[],
        )
        programme = self._extract_stage(
            self.extract_programme_info, "programme", context, empty
        )
        quantitative = self._extract_stage(
            self.extract_quantitative, "quantitative", context, programme
        )
        clean_outcome_classifications(
            quantitative.quantitative_outcomes, self.classification_framework
        )
        previous = programme.model_copy(
            update={"quantitative_outcomes": quantitative.quantitative_outcomes}
        )
        qualitative = self._extract_stage(
            self.extract_qualitative, "qualitative", context, previous
        )
        clean_outcome_classifications(
            qualitative.qualitative_outcomes, self.classification_framework
        )
        return self.programme_model(
            programme_characteristics=programme.programme_characteristics,
            interventions=programme.interventions,
            quantitative_outcomes=quantitative.quantitative_outcomes,
            qualitative_outcomes=qualitative.qualitative_outcomes,
        )


def build_programme_with_caching_pipeline(
    runtime_models: dict[str, type[BaseModel]],
) -> type[ProgrammeWithCachingExtractionPipeline]:
    """Use custom CSV field descriptions without changing schemas between stages."""
    fields = ProgrammeWithCachingStep.model_fields
    step_model = create_model(
        "DynamicProgrammeWithCachingStep",
        __base__=ProgrammeWithCachingStep,
        programme_characteristics=(
            runtime_models["Programme_Characteristics"] | None,
            Field(description=fields["programme_characteristics"].description),
        ),
        interventions=(
            GenericAlias(list, runtime_models["Intervention"]),
            Field(description=fields["interventions"].description),
        ),
        quantitative_outcomes=(
            GenericAlias(list, runtime_models["Quantitative_Outcome"]),
            Field(description=fields["quantitative_outcomes"].description),
        ),
        qualitative_outcomes=(
            GenericAlias(list, runtime_models["Qualitative_Outcome"]),
            Field(description=fields["qualitative_outcomes"].description),
        ),
    )
    signature = ExtractProgrammeWithCaching.with_updated_fields(
        "previous_extraction", type_=step_model
    ).with_updated_fields("extraction", type_=step_model)
    return type(
        "DynamicProgrammeWithCachingExtractionPipeline",
        (ProgrammeWithCachingExtractionPipeline,),
        {
            "programme_model": runtime_models["Programme"],
            "step_model": step_model,
            "extraction_signature": signature,
        },
    )
