"""Pytest configuration and shared fixtures."""

import os

import pytest


@pytest.fixture(scope="session")
def db_url():
    """Use in-memory SQLite for tests by default."""
    return os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")
