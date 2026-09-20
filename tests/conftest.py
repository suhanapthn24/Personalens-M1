import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personalens import HashEmbedder, IngestionPipeline, Settings, VectorMemory  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def memory(settings):
    return VectorMemory(settings.data_dir, HashEmbedder())


@pytest.fixture
def pipeline(memory, settings):
    return IngestionPipeline(memory, settings)


BAYES = (
    "Bayes Theorem relates conditional probabilities. It states that the posterior probability of a hypothesis "
    "equals the likelihood times the prior divided by the evidence. Conditional probability is a prerequisite "
    "for understanding Bayes Theorem. Naive Bayes is a classifier built on Bayes Theorem with independence assumptions."
)
GRADIENT = (
    "Gradient descent minimises a loss function by stepping against the gradient. The learning rate controls the "
    "step size. Too large a learning rate makes the optimiser diverge, while too small makes convergence slow."
)
