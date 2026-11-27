"""Shared test fixtures."""

import pytest

from src.embeddings.embedder import get_embedder


@pytest.fixture(scope="session")
def embedder():
    return get_embedder()  # loads the model once for all tests
