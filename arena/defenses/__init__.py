from .level1 import Level1
from .level2 import Level2
from .level3 import Level3
from .level4 import Level4
from .level5 import Level5

LEVELS = {1: Level1, 2: Level2, 3: Level3, 4: Level4, 5: Level5}


def make_defense(level: int, **kwargs):
    cls = LEVELS[int(level)]
    return cls(**kwargs) if level >= 3 else cls()
