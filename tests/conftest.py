import os
import sys
from pathlib import Path

# Offline + deterministic: never hit the network or download models in tests.
os.environ["ARENA_EMBEDDER"] = "fuzzy"
os.environ.pop("GROQ_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
