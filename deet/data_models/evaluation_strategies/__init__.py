"""Data models for evaluation strategies, inheriting from base."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from deet.data_models.enums import EvaluationStrategyName
from deet.data_models.evaluation_strategies.base import (
    BaseEvaluationStrategy,
)
from deet.data_models.evaluation_strategies.dev_val_test import (
    DevValTestEvaluationStrategy,
)
from deet.data_models.evaluation_strategies.null import (
    NullEvaluationStrategy,
)

if TYPE_CHECKING:
    from deet.data_models.project import DeetProject


_STRATEGY_REGISTRY: dict[
    EvaluationStrategyName,
    Callable[["DeetProject"], BaseEvaluationStrategy],
] = {
    EvaluationStrategyName.NONE: NullEvaluationStrategy,
    EvaluationStrategyName.DEV_VAL_TEST: DevValTestEvaluationStrategy,
}
