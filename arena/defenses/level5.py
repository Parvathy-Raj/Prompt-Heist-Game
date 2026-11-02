"""Level 5: Level 4 + a second-LLM guard on both input and output."""
from .guard import Guard
from .level4 import Level4


class Level5(Level4):
    level = 5
    title = "Level 5 - LLM guard"
    description = "A separate guard model classifies the input and the output."

    def input_layers(self):
        return super().input_layers() + [Guard("input")]

    def output_layers(self):
        return super().output_layers() + [Guard("output")]
