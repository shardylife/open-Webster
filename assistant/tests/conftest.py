"""Import ``assistant`` without a Pi: stub the wake-word and audio modules and
point HOME at a temp dir holding a minimal config.env."""
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "stubs"))
sys.path.insert(0, os.path.dirname(HERE))


@pytest.fixture(scope="session")
def assistant(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    (home / "assistant").mkdir()
    (home / "assistant" / "config.env").write_text(
        'OPENROUTER_API_KEY="sk-test"\nexport MODEL=test/model\n# comment\n')
    os.environ["HOME"] = str(home)
    import assistant as module
    return module
