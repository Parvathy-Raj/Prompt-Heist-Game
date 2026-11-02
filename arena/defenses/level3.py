"""Level 3: Level 2 + input filter (regex blocklist + embedding similarity)."""
from __future__ import annotations

from .input_filter import RegexFilter, SimilarityFilter, get_scorer, load_phrases
from .level2 import Level2


class Level3(Level2):
    level = 3
    title = "Level 3 - Input filter"
    description = "Regex blocklist + similarity to known jailbreak phrases, before the LLM."

    def __init__(self, phrases_path=None, threshold: float | None = None, embedder: str | None = None,
                 cache_dir=None):
        from config import get_settings
        st = get_settings()
        self.phrases_path = phrases_path or st.jailbreak_phrases_path
        self.threshold = st.sim_threshold if threshold is None else threshold
        self.embedder = embedder
        self.cache_dir = cache_dir or st.cache_dir
        self._layers = None

    def input_layers(self):
        if self._layers is None:
            scorer = get_scorer(self.phrases_path, self.embedder, self.cache_dir)
            self._layers = [RegexFilter(),
                            SimilarityFilter(scorer, load_phrases(self.phrases_path), self.threshold)]
        return super().input_layers() + self._layers
