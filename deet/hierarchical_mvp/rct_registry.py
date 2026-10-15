"""Model modules for single-arm and preserved pairwise RCT study types."""

from . import (
    AnimalRCTmodel,
    AnimalRCTmodel_pairwise,
    CochraneRCTmodel,
    CochraneRCTmodel_pairwise,
    ObesityRCTmodel,
    ObesityRCTmodel_pairwise,
    RCTmodel,
    RCTmodel_pairwise,
)

RCT_MODEL_MODULES = {
    "RCT": RCTmodel,
    "CochraneRCT": CochraneRCTmodel,
    "ObesityRCT": ObesityRCTmodel,
    "AnimalRCT": AnimalRCTmodel,
    "RCT_pairwise": RCTmodel_pairwise,
    "CochraneRCT_pairwise": CochraneRCTmodel_pairwise,
    "ObesityRCT_pairwise": ObesityRCTmodel_pairwise,
    "AnimalRCT_pairwise": AnimalRCTmodel_pairwise,
}
