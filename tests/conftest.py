"""Test-wide setup: the Voice Room (app/agents) keeps its database in a throwaway folder."""

import os
import tempfile

_AGENTS_DIR = tempfile.mkdtemp(prefix="voice-room-test-")
os.environ.setdefault("AGENTS_DATA_DIR", _AGENTS_DIR)
