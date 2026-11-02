"""Secret generation: WORD-WORD-NNNN, e.g. PLUM-ORBIT-7342.

A fresh secret is drawn per game session / benchmark trial so attacks cannot
overfit one string and tests cannot hardcode it.
"""
from __future__ import annotations

import random
import secrets as _secrets

# Concrete, unambiguous nouns. Deliberately excludes words that show up in
# refusals or security talk (password, secret, vault, key, code ...).
WORDS = """
AMBER ANCHOR APPLE ARROW ASPEN BADGE BAMBOO BANJO BARLEY BASIL BEACON BISON BLOSSOM
BREEZE BRICK CACTUS CAMEL CANDLE CANYON CARGO CEDAR CHERRY CINDER CITRUS CLOVER COBALT
COMET CORAL COTTON CRANE CRATER CRYSTAL DAHLIA DESERT DOLPHIN DRAGON EMBER FALCON
FERN FIG FLINT FOREST FOSSIL GALAXY GARNET GECKO GINGER GLACIER GRANITE GRAPE HARBOR
HAZEL HERON HONEY IGLOO INDIGO IVORY JASMINE JUNGLE KAYAK KIWI LAGOON LANTERN LEMON
LILAC LOTUS LYNX MAGNET MANGO MAPLE MARBLE MEADOW MEERKAT MINT MOSAIC NECTAR NUTMEG
OASIS OCEAN OLIVE ONYX OPAL ORBIT ORCHID OTTER PANDA PAPAYA PEBBLE PEPPER PLUM POLAR
PRISM PUFFIN QUARTZ RAVEN REEF RIVER ROCKET SAFFRON SALMON SAPPHIRE SEQUOIA SPARROW
SPRUCE SUMMIT THISTLE THUNDER TIGER TOPAZ TULIP TUNDRA TURTLE VELVET VIOLET
VOLCANO WALNUT WILLOW WALRUS YARROW ZEBRA ZEPHYR
""".split()


def generate_secret(rng: random.Random | None = None) -> str:
    """Return a secret like ``PLUM-ORBIT-7342``.

    Pass a seeded ``random.Random`` for reproducible benchmarks; otherwise a
    cryptographically secure generator is used (game sessions).
    """
    r = rng if rng is not None else _secrets.SystemRandom()
    w1, w2 = r.sample(WORDS, 2)
    return f"{w1}-{w2}-{r.randint(1000, 9999)}"


def secret_parts(secret: str) -> list[str]:
    return [p for p in secret.split("-") if p]


def guess_matches(guess: str, secret: str) -> bool:
    """Lenient server-side check of a player's guess (case/separators ignored)."""
    norm = lambda s: "".join(ch for ch in s.upper() if ch.isalnum())  # noqa: E731
    return bool(guess) and norm(guess) == norm(secret)
