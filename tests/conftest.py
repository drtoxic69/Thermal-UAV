"""
Pytest configuration file.

This file automatically loads environment variables
from a .env file before any tests are run.
"""

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """
    A session-wide fixture to automatically load environment
    variables from a .env file at the start of the test run.
    """
    load_dotenv()
