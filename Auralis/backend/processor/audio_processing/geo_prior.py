"""
geo_prior.py

Inference-time geo-temporal candidate filtering (workflow summary,
"next steps" #6). This does NOT retrain anything -- it reshapes a
per-location/per-time species-probability signal into a per-class prior
over the SAME class space the CNN was trained on (final_label slugs from
build_taxonomy_classes.py / inatsounds_route.py), then predict_v2.py
combines that prior with the CNN's own output.

Two ways to get that per-class prior, pick whichever matches what you
actually have:

1. JsonGeoPrior -- use this if "your separate region/season
   species-probability dataset" is your own lookup table (the common
   case: a gazetteer / range-map / checklist-derived table you already
   built). Expected format, one JSON file:

       [
         {"lat_min": -10, "lat_max": 10, "lon_min": 90, "lon_max": 120,
          "months": [1,2,3,12],
          "species": {"Jaguar": 0.9, "Ocelot": 0.6, "Rodentia (order)": 0.3}},
         ...
       ]

   Keys under "species" should be the SAME final_label strings
   build_taxonomy_classes.py produced (species common name, or a rollup
   label like "Panthera spp." / "Rodentia (order)") -- they get slugified
   the same way inatsounds_route.py slugifies them, so casing/spacing
   doesn't need to match exactly. Species/rollups not mentioned in a
   matching region+month block default to a neutral prior (no filtering)
   rather than zero, since an incomplete lookup table is a much more
   likely failure mode than a deliberately-exhaustive one, and silently
   zeroing everything not explicitly listed would be a bad default.
   Adjust `default_prior` if your table IS meant to be exhaustive.

2. SinrGeoPrior -- use this if you instead want the SINR geo-prior model
   from the iNatSounds paper itself (their biggest single accuracy lever:
   44.7%->61.6% Top-1 in one config). This is ported directly from
   iNatSounds-main/src/models.py's GeoModel, plus a rollup step this
   project needs that the original doesn't: the official model predicts
   per-SPECIES (per raw iNat category id), but our classifier's label
   space is genus/family/order-collapsed for long-tail species (see
   build_taxonomy_classes.py). RollupGeoPrior max-pools the raw
   per-species SINR output over every species folded into each rolled-up
   class, so a rolled-up "Panthera spp." class gets a sensible prior even
   though SINR itself has no such class. Needs the pretrained SINR
   weights + inat_id2scientific.json, both released alongside the
   iNatSounds paper/repo (not included here -- see the repo's README for
   the weights link).
"""

import json
import re
from pathlib import Path

import numpy as np


def slugify(label: str) -> str:
    """Must match inatsounds_route.py's slugify() exactly, or lookups
    silently miss."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


# --------------------------------------------------------------------------
# 1. Generic JSON lookup-table geo prior
# --------------------------------------------------------------------------

class JsonGeoPrior:
    def __init__(self, path: Path, default_prior: float = 1.0):
        with open(path) as f:
            self.blocks = json.load(f)
        self.default_prior = default_prior
        # pre-slugify each block's species keys once
        for block in self.blocks:
            block["_species_slugged"] = {slugify(k): v for k, v in block.get("species", {}).items()}

    def _matches(self, block, lat, lon, month):
        if not (block["lat_min"] <= lat <= block["lat_max"]):
            return False
        if not (block["lon_min"] <= lon <= block["lon_max"]):
            return False
        months = block.get("months")
        if months is not None and month not in months:
            return False
        return True

    def predict(self, lat: float, lon: float, month: int, class_slugs):
        """Returns {slug: prior_in_[0,1]} for every slug in class_slugs.
        Slugs matched by more than one overlapping block take the max
        (most permissive) -- a region/season lookup is meant to rule
        things OUT, not additionally penalize plausible overlaps."""
        prior = {slug: self.default_prior for slug in class_slugs}
        explicitly_set = set()
        for block in self.blocks:
            if not self._matches(block, lat, lon, month):
                continue
            for slug, p in block["_species_slugged"].items():
                if slug not in prior:
                    continue
                # The first matching block's explicit value REPLACES the
                # neutral default (not max()'d against it -- max()'ing an
                # explicit 0.1 against the neutral default of 1.0 would
                # silently keep the 1.0 and defeat the whole point of
                # having an explicit low value). Only once a slug already
                # has an explicit value from an earlier matching block do
                # further matching blocks get max()'d against each other.
                if slug not in explicitly_set:
                    prior[slug] = p
                    explicitly_set.add(slug)
                else:
                    prior[slug] = max(prior[slug], p)
        return prior


# --------------------------------------------------------------------------
# 2. SINR-style geo-prior model (ported from iNatSounds-main/src/models.py),
#    rolled up onto the genus/family/order-collapsed class space.
# --------------------------------------------------------------------------

class _ResLayer:
    """Lazily-imported torch.nn module -- see _build_sinr_model()."""
    pass


def _build_sinr_torch_modules():
    import torch
    import torch.nn as nn

    class ResLayer(nn.Module):
        def __init__(self, linear_size):
            super().__init__()
            self.nonlin1 = nn.ReLU(inplace=True)
            self.nonlin2 = nn.ReLU(inplace=True)
            self.dropout1 = nn.Dropout()
            self.w1 = nn.Linear(linear_size, linear_size)
            self.w2 = nn.Linear(linear_size, linear_size)

        def forward(self, x):
            y = self.w1(x)
            y = self.nonlin1(y)
            y = self.dropout1(y)
            y = self.w2(y)
            y = self.nonlin2(y)
            return x + y

    class SinrGeoModel(nn.Module):
        """Directly ported from iNatSounds-main/src/models.py (GeoModel),
        minus the val.json/inat2sci_path wiring -- that mapping from raw
        SINR class index to iNat category id is handled by SinrGeoPrior
        below instead, so this stays a plain lat/lon -> per-SINR-class
        sigmoid-probability model."""

        def __init__(self, num_classes, num_inputs=4, num_filts=256, depth=4):
            super().__init__()
            self.class_emb = nn.Linear(num_filts, num_classes, bias=False)
            layers = [nn.Linear(num_inputs, num_filts), nn.ReLU(inplace=True)]
            for _ in range(depth):
                layers.append(ResLayer(num_filts))
            self.feats = nn.Sequential(*layers)

        def forward(self, x):
            # x: B x 2, [lat, lon] normalized to [-1, 1]
            assert ((x > 1) + (x < -1)).sum() == 0
            x = torch.flip(x, [-1])  # [lat,lon] -> [lon,lat], SINR's convention
            x_encode = torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], -1)
            loc_emb = self.feats(x_encode)
            return torch.sigmoid(self.class_emb(loc_emb))

    return torch, SinrGeoModel


class SinrGeoPrior:
    def __init__(self, geo_model_weights: Path, inat_id2scientific_json: Path,
                 species_class_assignments: Path, dataset_categories_json: Path):
        """
        geo_model_weights: SINR checkpoint (torch.load -> {"state_dict":...,
            "params": {"class_to_taxa": [...]}}), released with the paper.
        inat_id2scientific_json: iNatSounds-main/assets/inat_id2scientific.json
        species_class_assignments: species_class_assignments.json from
            build_taxonomy_classes.py (category_id -> final_label/resolution)
        dataset_categories_json: any of train/val/test.json (just need its
            "categories" list, for id -> scientific-name mapping)
        """
        torch, SinrGeoModel = _build_sinr_torch_modules()
        self._torch = torch

        checkpoint = torch.load(geo_model_weights, map_location="cpu")
        class_to_taxa = checkpoint["params"]["class_to_taxa"]
        self.model = SinrGeoModel(num_classes=len(class_to_taxa))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        with open(inat_id2scientific_json) as f:
            inat2sci = json.load(f)
        sci2inat = {v: int(k) for k, v in inat2sci.items()}

        with open(dataset_categories_json) as f:
            categories = json.load(f)["categories"]
        inat2cat = {sci2inat[c["name"]]: c["id"] for c in categories if c["name"] in sci2inat}

        # raw SINR output index -> our category_id, for every taxon SINR covers
        self.sinr_idx_to_category_id = {
            i: inat2cat[taxa] for i, taxa in enumerate(class_to_taxa) if taxa in inat2cat
        }

        with open(species_class_assignments) as f:
            assignments = json.load(f)
        # slug -> [sinr_idx, ...] so we can max-pool raw species scores up
        # to whatever rolled-up class each species belongs to.
        self.slug_to_sinr_idxs = {}
        for sinr_idx, cat_id in self.sinr_idx_to_category_id.items():
            info = assignments.get(str(cat_id))
            if info is None or info.get("resolution") == "excluded_zero_shot_only":
                continue
            slug = slugify(info["final_label"])
            self.slug_to_sinr_idxs.setdefault(slug, []).append(sinr_idx)

    def predict(self, lat: float, lon: float, month: int, class_slugs):
        """month is accepted for interface symmetry with JsonGeoPrior;
        the released SINR model is location-only (no explicit season
        term) -- if you have a season-aware variant, condition here."""
        torch = self._torch
        x = torch.tensor([[lat / 90.0, lon / 180.0]], dtype=torch.float32)
        with torch.no_grad():
            raw = self.model(x).squeeze(0).numpy()  # (num_sinr_classes,)

        prior = {slug: 1.0 for slug in class_slugs}
        for slug, idxs in self.slug_to_sinr_idxs.items():
            if slug in prior:
                prior[slug] = float(np.max(raw[idxs]))
        return prior


# --------------------------------------------------------------------------
# Combining a prior with the CNN's own probabilities
# --------------------------------------------------------------------------

def apply_geo_prior(cnn_probs: dict, geo_prior: dict | None,
                     temperature: float = 1.0, hard_zero_threshold: float = 0.02):
    """cnn_probs, geo_prior: {label_slug: probability}. geo_prior may be
    None (no location/date given, or no geo prior configured) -- in that
    case cnn_probs is returned unchanged.

    A geo prior below hard_zero_threshold hard-masks the class out
    entirely (this also resolves acoustically-similar-but-geographically-
    disjoint confusions like lion vs. jaguar automatically -- see workflow
    summary -- without having to hand-prune the training class list).
    Above that threshold it's a soft reweighting via
    cnn_prob * geo_prior**temperature, renormalized to sum to 1 over the
    surviving classes so the output is still a proper probability
    distribution comparable to cnn-only output."""
    if geo_prior is None:
        return dict(cnn_probs)

    combined = {}
    for label, p in cnn_probs.items():
        g = geo_prior.get(label, 1.0)
        if g < hard_zero_threshold:
            combined[label] = 0.0
        else:
            combined[label] = p * (g ** temperature)

    total = sum(combined.values())
    if total <= 0:
        # geo prior zeroed out everything the CNN was confident about --
        # more likely a bad/mismatched lookup than a real "nothing here"
        # situation, so fall back to the ungated CNN output rather than
        # returning an all-zero result.
        return dict(cnn_probs)
    return {label: p / total for label, p in combined.items()}
