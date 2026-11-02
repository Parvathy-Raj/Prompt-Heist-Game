"""Level 4: Level 3 + output filter that blocks responses containing the secret."""
from .level3 import Level3
from .output_filter import OutputFilter


class Level4(Level3):
    level = 4
    title = "Level 4 - Output filter"
    description = "Scans the reply for the password (plain, encoded, spaced, fuzzy...)."

    def output_layers(self):
        return super().output_layers() + [OutputFilter()]
