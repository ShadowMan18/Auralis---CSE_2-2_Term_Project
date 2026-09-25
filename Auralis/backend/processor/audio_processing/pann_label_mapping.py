"""Editable mapping from PANN/AudioSet tags to Auralis animal categories.

Clear tag-to-animal matches are mapped; all other AudioSet tags pass through
unchanged. Generic livestock evidence is kept as ``farm_animals`` when no
more specific farm species is detected. Several PANN tags can support one
category; their scores are combined with max so correlated synonyms do not
inflate confidence.
"""

RAW_TAG_TO_CATEGORY = {
    # Cat
    "cat": "cat",
    "meow": "cat",
    "caterwaul": "cat",

    # Dog
    "dog": "dog",
    "bark": "dog",
    "bow-wow": "dog",
    "whimper (dog)": "dog",
    "yip": "dog",
    "canidae, dogs, wolves": "dog",

    # Cow
    "moo": "cow",
    "cattle, bovinae": "cow",

    # Other direct species
    "goat": "goat",
    "horse": "horse",
    "neigh, whinny": "horse",
    "clip-clop": "horse",
    "crow": "crow",
    "frog": "frog",
    "croak": "frog",
    "chicken, rooster": "rooster",
    "cluck": "rooster",
    "crowing, cock-a-doodle-doo": "rooster",

    "caw": "crow",

    # Broad/uncertain farm-animal evidence. These never imply a specific
    # species; the category is suppressed when a specific farm animal wins.
    "livestock, farm animals, working animals": "farm_animals",
    "bleat": "farm_animals",
    "sheep": "goat",
    "fowl": "farm_animals",
}

FARM_SPECIFIC_CATEGORIES = frozenset({"cow", "goat", "horse", "rooster"})
MAPPED_ANIMAL_CATEGORIES = frozenset(RAW_TAG_TO_CATEGORY.values())

# Keep these raw labels if they are the only evidence. When a mapped animal
# category is present, they add no useful species information and are hidden.
GENERIC_LABELS_SUPPRESSED_WITH_ANIMAL_MATCH = frozenset({
    "pink noise",
    "animal",
    "domestic animals, pets",
    "bird",
})


def map_pann_scores(raw_scores, min_confidence=0.0):
    """Map raw AudioSet tag scores to Auralis categories.

    Unmapped tags pass through with their original labels and scores. The
    broad/noise labels listed in
    ``GENERIC_LABELS_SUPPRESSED_WITH_ANIMAL_MATCH`` pass through only when no
    mapped animal category reaches ``min_confidence``. Alias scores use max
    rather than sum. ``farm_animals`` acts as a fallback when no specific
    farm category reaches ``min_confidence``. The PANN ``Goat`` tag maps to
    ``goat``; ambiguous ``Bleat`` remains ``farm_animals``.
    """
    mapped = {}
    for raw_label, score in raw_scores.items():
        original_label = str(raw_label).strip()
        category = RAW_TAG_TO_CATEGORY.get(original_label.casefold(), original_label)
        if not category:
            continue
        mapped[category] = max(mapped.get(category, 0.0), float(score))

    animal_match = any(
        category in MAPPED_ANIMAL_CATEGORIES and score >= min_confidence
        for category, score in mapped.items()
    )
    if animal_match:
        for label in list(mapped):
            if label.casefold() in GENERIC_LABELS_SUPPRESSED_WITH_ANIMAL_MATCH:
                del mapped[label]

    farm_species_detected = any(
        mapped.get(category, 0.0) >= min_confidence
        for category in FARM_SPECIFIC_CATEGORIES
    )
    if farm_species_detected:
        mapped.pop("farm_animals", None)

    return mapped
