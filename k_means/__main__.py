"""Runs program with default parameters."""
from k_means.core.main import run
from k_means.resources.input import default_message
from tests.unit.resources.test_input import test_default_message

if __name__ == "__main__":
    if test_default_message():
        run(default_message)
