"""Controlled vocabularies for the behavioural-intervention extraction.

Values follow the codebook (resources/Codebook-2025-02-24.docx). Where the
human-coded CSV uses a different convention, the codebook wording is used here
and the mapping is left to the gold-data loader.
"""

from __future__ import annotations

from typing import Literal

YesNo = Literal["Yes", "No"]
YesNoUnclear = Literal["Yes", "No", "Unclear"]

DocumentType = Literal[
    "Peer-Reviewed Paper",
    "Unpublished Academic Paper",
    "Conference Paper",
]

ExclusionReason = Literal[
    "",
    "No relevant effect sizes",
    "Effect size information incomplete or irretrievable",
    "Document not relevant",
    "Other",
]

GeographicScope = Literal[
    "Smaller than municipality",
    "Municipality",
    "City",
    "County",
    "State",
    "Utility Area",
    "Country",
    "Other",
]

RandomisationLevel = Literal["household", "group", "not randomised"]

InterventionType = Literal[
    "Information",
    "Feedback",
    "Monetary Incentives",
    "Social Comparison",
    "Motivation",
]

# Sub-types per broad type (codebook "Intervention types and sub-types").
# Sub-types that appear as columns in the human-coded data but not in the
# codebook list are included under their closest type.
SUBTYPES_BY_TYPE: dict[str, list[str]] = {
    "Information": ["Home audits", "Tips", "Awareness Campaign"],
    "Feedback": ["Historical", "Bills", "In home display", "Smart Metering"],
    "Social Comparison": [
        "Peer comparison",
        "Home Energy Report",
        "Injunctive Norm / Social Norm",
    ],
    "Monetary Incentives": [
        "Rewards",
        "ToU",
        "Real-time Pricing",
        "Dynamic Pricing",
        "Peak Load Pricing",
        "Critical Peak Pricing",
    ],
    "Motivation": [
        "Gamification",
        "Commitment strategies",
        "Goal Setting",
        "Competition",
        "Moral suasion",
    ],
}

InterventionSubtype = Literal[
    "Home audits",
    "Tips",
    "Awareness Campaign",
    "Historical",
    "Bills",
    "In home display",
    "Smart Metering",
    "Peer comparison",
    "Home Energy Report",
    "Injunctive Norm / Social Norm",
    "Rewards",
    "ToU",
    "Real-time Pricing",
    "Dynamic Pricing",
    "Peak Load Pricing",
    "Critical Peak Pricing",
    "Gamification",
    "Commitment strategies",
    "Goal Setting",
    "Competition",
    "Moral suasion",
]

FramingUnit = Literal[
    "Energy (kWh)",
    "Savings",
    "CO2",
    "Other efficiency (R-value)",
    "Other ecological",
    "Relative consumption (%)",
    "Not applicable",
]

InformationFrequency = Literal[
    "continuous (on-demand)",
    "real-time",
    "daily",
    "weekly",
    "bi-weekly",
    "monthly",
    "bi-monthly",
    "one time only",
    "other - more frequent",
    "other - less frequent",
    "Not applicable",
]

StudyDesign = Literal[
    "Pre-test/post-test",
    "Control-treatment",
    "Difference in Difference",
    "NA",
]

# Never "Difference in Difference": that is the study design.
StatisticalTechnique = Literal[
    "NA",
    "Difference of means",
    "ANOVA",
    "One-step GMM",
    "Household fixed effects",
    "Time fixed effects",
    "Household and time fixed effects",
    "Random effects regression",
    "OLS",
    "Other",
]

EstimateKind = Literal["coefficient", "difference_of_means"]

OutcomeScale = Literal["absolute", "log", "percent"]

EffectDirection = Literal["Increase", "Decrease", "No change", "NA"]

CoefficientSdType = Literal[
    "NA",
    "Standard error",
    "Standard deviation",
    "Robust standard errors",
    "Clustered standard errors",
    "Bootstrapped clustered standard errors",
    "Pooled standard errors",
    "Corrected variance matrix",
    "Not provided",
]

SignificanceReporting = Literal["actual", "lower bound", "upper bound", "none"]

SignificanceTest = Literal["t-test", "F-test", "z-test", "H-test", "Other", "NA"]

BaselineSource = Literal[
    "all households, baseline period",
    "weighted treatment and control, baseline period",
    "all households, treatment period",
    "weighted treatment and control, treatment period",
    "control group",
    "not reported",
]

ControlVariable = Literal[
    "Weather controls",
    "Seasonal controls",
    "Energy prices",
    "Residence controls",
    "HH controls (demographics)",
    "Electricity use",
    "Environmental attitudes",
]
