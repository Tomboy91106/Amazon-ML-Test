"""
features.py
Pairwise features for one Source-1 entity and one blocked Source-2 or
Source-3 candidate.

Inputs are the candidate-pair table from blocking.py
(source1_entity_id, candidate_entity_id, rules) and the source frames.
Name and address text come from the existing normalization views. Nothing
here calls an external service, an embedding model, or a geocoder.

Missing text is never an exact match. Two empty names, two empty
addresses, or two empty countries score 0 on every exact-match feature.
Similarity and length features are NaN when either side has no text, so
LightGBM can treat that as missing. The missing-indicator columns still
separate "no text" from "text that does not match", so filling those
NaNs with 0 is safe for estimators that cannot accept NaN.

Character TF-IDF rank is not a column. generate_candidates() records
which passes fired, not the TF-IDF score or rank.
"""
from __future__ import annotations

import math

import pandas as pd
from rapidfuzz import fuzz

from .blocking import (
    RULE_ADDRESS_NUMBER,
    RULE_ADDRESS_POSTAL,
    RULE_ADDRESS_TOKEN,
    RULE_CHAR_TFIDF,
    RULE_EXACT_CORE,
    RULE_EXACT_NAME,
    RULE_EXACT_SORTED,
    RULE_NAME_TOKEN,
    RULE_TRANSLIT,
    ground_truth_map,
)
from .normalization import add_normalized_columns
from .text_utils import is_missing_token, tokenize

# RapidFuzz's default processor strips characters with a \\W regex, which
# deletes Devanagari/Tamil/Kannada vowel signs. These strings are already
# normalized, so every scorer is called with processor=None.
_FUZZ_KW = {"processor": None}

_VIEW_COLUMNS = (
    "business_name_basic",
    "business_name_core",
    "business_name_sorted",
    "business_name_transliterated",
    "business_name_tokens",
    "business_address_basic",
    "business_address_tokens",
    "business_address_transliterated",
    "address_numbers",
    "address_postal_candidates",
)

_RULE_FLAGS = (
    ("rule_exact_name", RULE_EXACT_NAME),
    ("rule_exact_core", RULE_EXACT_CORE),
    ("rule_exact_sorted", RULE_EXACT_SORTED),
    ("rule_translit_name", RULE_TRANSLIT),
    ("rule_name_token", RULE_NAME_TOKEN),
    ("rule_address_token", RULE_ADDRESS_TOKEN),
    ("rule_address_number", RULE_ADDRESS_NUMBER),
    ("rule_address_postal", RULE_ADDRESS_POSTAL),
    ("rule_char_tfidf", RULE_CHAR_TFIDF),
)

FEATURE_COLUMNS = (
    # Name. Exact flags are 0 when either side is empty.
    "name_exact_basic",
    "name_exact_core",
    "name_exact_sorted",
    "name_exact_translit",
    "name_ratio_basic",
    "name_ratio_core",
    "name_ratio_translit",
    "name_token_sort_ratio",
    "name_token_jaccard",
    "name_shared_tokens",
    "name_len_diff",
    "name_len_ratio",
    "name_acronym",
    # Address.
    "addr_ratio_basic",
    "addr_ratio_translit",
    "addr_token_sort_ratio",
    "addr_token_jaccard",
    "addr_shared_tokens",
    "addr_number_agree",
    "addr_number_overlap",
    "addr_postal_agree",
    "addr_len_diff",
    "addr_len_ratio",
    # Country, missingness, source. Not text similarity.
    "country_exact",
    "country_missing",
    "name_s1_missing",
    "name_cand_missing",
    "addr_s1_missing",
    "addr_cand_missing",
    "cand_is_s3",
    # Blocking metadata copied off the pair's `rules` value.
    "n_blocking_rules",
    "rule_exact_name",
    "rule_exact_core",
    "rule_exact_sorted",
    "rule_translit_name",
    "rule_name_token",
    "rule_address_token",
    "rule_address_number",
    "rule_address_postal",
    "rule_char_tfidf",
)

ID_COLUMNS = ("source1_entity_id", "candidate_entity_id")
OUTPUT_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS


class _Entity:
    __slots__ = (
        "country",
        "country_missing",
        "name_basic",
        "name_core",
        "name_sorted",
        "name_trans",
        "name_tokens",
        "core_tokens",
        "addr_basic",
        "addr_tokens",
        "addr_trans",
        "numbers",
        "postals",
        "premises",
        "name_missing",
        "addr_missing",
        "is_s3",
    )

    def __init__(
        self,
        country,
        country_missing,
        name_basic,
        name_core,
        name_sorted,
        name_trans,
        name_tokens,
        core_tokens,
        addr_basic,
        addr_tokens,
        addr_trans,
        numbers,
        postals,
        premises,
        name_missing,
        addr_missing,
        is_s3,
    ):
        self.country = country
        self.country_missing = country_missing
        self.name_basic = name_basic
        self.name_core = name_core
        self.name_sorted = name_sorted
        self.name_trans = name_trans
        self.name_tokens = name_tokens
        self.core_tokens = core_tokens
        self.addr_basic = addr_basic
        self.addr_tokens = addr_tokens
        self.addr_trans = addr_trans
        self.numbers = numbers
        self.postals = postals
        self.premises = premises
        self.name_missing = name_missing
        self.addr_missing = addr_missing
        self.is_s3 = is_s3


def _blank(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_tuple(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, float) and math.isnan(value):
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(item for item in value if item)


def _premises_number(numbers: tuple) -> str:
    """First 1–4 digit run. Postal-length runs stay in the postal feature."""
    for number in numbers:
        if 1 <= len(number) <= 4:
            return number
    return ""


def _empty_entity(entity_id: str) -> _Entity:
    return _Entity(
        country="",
        country_missing=1.0,
        name_basic="",
        name_core="",
        name_sorted="",
        name_trans="",
        name_tokens=frozenset(),
        core_tokens=(),
        addr_basic="",
        addr_tokens=frozenset(),
        addr_trans="",
        numbers=frozenset(),
        postals=frozenset(),
        premises="",
        name_missing=1.0,
        addr_missing=1.0,
        is_s3=1.0 if str(entity_id).startswith("S3-") else 0.0,
    )


def _entity_from_row(row) -> _Entity:
    country_missing = 1.0 if is_missing_token(row["country"]) else 0.0
    country = "" if country_missing else _blank(row["country"])
    name_basic = _blank(row["business_name_basic"])
    name_core = _blank(row["business_name_core"])
    addr_basic = _blank(row["business_address_basic"])
    numbers = _as_tuple(row["address_numbers"])
    entity_id = _blank(row["entity_id"])
    return _Entity(
        country=country,
        country_missing=country_missing,
        name_basic=name_basic,
        name_core=name_core,
        name_sorted=_blank(row["business_name_sorted"]),
        name_trans=_blank(row["business_name_transliterated"]),
        name_tokens=frozenset(_as_tuple(row["business_name_tokens"])),
        core_tokens=tuple(tokenize(name_core)),
        addr_basic=addr_basic,
        addr_tokens=frozenset(_as_tuple(row["business_address_tokens"])),
        addr_trans=_blank(row["business_address_transliterated"]),
        numbers=frozenset(numbers),
        postals=frozenset(_as_tuple(row["address_postal_candidates"])),
        premises=_premises_number(numbers),
        name_missing=0.0 if name_basic else 1.0,
        addr_missing=0.0 if addr_basic else 1.0,
        is_s3=1.0 if entity_id.startswith("S3-") else 0.0,
    )


def _views_ready(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in _VIEW_COLUMNS)


def _frame_for_ids(df: pd.DataFrame, wanted: set) -> pd.DataFrame:
    """Keep only entities that appear in the pair table, then normalize.

    Normalization runs on that subset. A small pair list does not trigger
    a full-file normalize even if the caller passed a large frame.
    """
    if "entity_id" not in df.columns:
        raise ValueError("entity frame is missing entity_id")
    if wanted:
        df = df.loc[df["entity_id"].isin(wanted)]
    if df.empty:
        return df.iloc[0:0]
    if _views_ready(df):
        keep = ["entity_id", "country", *_VIEW_COLUMNS]
        missing = [column for column in keep if column not in df.columns]
        if missing:
            raise ValueError(f"normalized frame is missing columns: {missing}")
        return df.loc[:, keep]
    missing = [column for column in ("business_name", "business_address", "country") if column not in df.columns]
    if missing:
        raise ValueError(f"entity frame is missing columns: {missing}")
    raw = df.loc[:, ["entity_id", "business_name", "business_address", "country"]]
    return add_normalized_columns(raw)


def _index_entities(frames, wanted: set) -> dict:
    indexed = {}
    for frame in frames:
        prepared = _frame_for_ids(frame, wanted)
        if prepared.empty:
            continue
        records = prepared.to_dict("records")
        for record in records:
            indexed[_blank(record["entity_id"])] = _entity_from_row(record)
    return indexed


def _exact(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return 1.0 if left == right else 0.0


def _fuzz_ratio(left: str, right: str) -> float:
    if not left or not right:
        return math.nan
    return float(fuzz.ratio(left, right, **_FUZZ_KW)) / 100.0


def _fuzz_token_sort(left: str, right: str) -> float:
    if not left or not right:
        return math.nan
    return float(fuzz.token_sort_ratio(left, right, **_FUZZ_KW)) / 100.0


def _jaccard(left: frozenset, right: frozenset) -> float:
    if not left or not right:
        return math.nan
    overlap = len(left & right)
    union = len(left) + len(right) - overlap
    if union == 0:
        return math.nan
    return overlap / union


def _shared(left: frozenset, right: frozenset) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right))


def _length_stats(left: str, right: str) -> tuple:
    if not left or not right:
        return math.nan, math.nan
    left_len = len(left)
    right_len = len(right)
    longest = left_len if left_len >= right_len else right_len
    shortest = right_len if left_len >= right_len else left_len
    return float(abs(left_len - right_len)), shortest / longest


def _acronym(left_tokens: tuple, right_tokens: tuple) -> float:
    """1 when one core name is an acronym of the other core name.

    Uses business_name_core, so trailing legal suffixes already removed by
    normalization are not part of the initials. A short single-token name
    matches only a multi-token name whose initials spell it. Two ordinary
    names with the same initials do not fire.
    """
    def acronym_form(tokens):
        if len(tokens) == 1 and tokens[0].isalpha() and 2 <= len(tokens[0]) <= 6:
            return tokens[0]
        if len(tokens) >= 2 and all(len(token) == 1 and token.isalpha() for token in tokens):
            return "".join(tokens)
        return None

    def initials(tokens):
        if len(tokens) < 2:
            return None
        if any((not token) or (not token[0].isalpha()) for token in tokens):
            return None
        return "".join(token[0] for token in tokens)

    left_form = acronym_form(left_tokens)
    right_form = acronym_form(right_tokens)
    left_initials = initials(left_tokens)
    right_initials = initials(right_tokens)
    if left_form and right_initials and left_form == right_initials:
        return 1.0
    if right_form and left_initials and right_form == left_initials:
        return 1.0
    return 0.0


def _rule_features(rules) -> tuple:
    if rules is None or (isinstance(rules, float) and math.isnan(rules)):
        present = set()
    else:
        text = str(rules).strip()
        present = set(text.split("|")) if text else set()
    flags = [1.0 if name in present else 0.0 for _, name in _RULE_FLAGS]
    return float(sum(flags)), flags


def _feature_dict(source, candidate, rules, candidate_id: str) -> dict:
    name_len_diff, name_len_ratio = _length_stats(source.name_basic, candidate.name_basic)
    addr_len_diff, addr_len_ratio = _length_stats(source.addr_basic, candidate.addr_basic)
    n_rules, rule_flags = _rule_features(rules)
    both_countries = source.country_missing == 0.0 and candidate.country_missing == 0.0
    features = {
        "name_exact_basic": _exact(source.name_basic, candidate.name_basic),
        "name_exact_core": _exact(source.name_core, candidate.name_core),
        "name_exact_sorted": _exact(source.name_sorted, candidate.name_sorted),
        "name_exact_translit": _exact(source.name_trans, candidate.name_trans),
        "name_ratio_basic": _fuzz_ratio(source.name_basic, candidate.name_basic),
        "name_ratio_core": _fuzz_ratio(source.name_core, candidate.name_core),
        "name_ratio_translit": _fuzz_ratio(source.name_trans, candidate.name_trans),
        "name_token_sort_ratio": _fuzz_token_sort(source.name_basic, candidate.name_basic),
        "name_token_jaccard": _jaccard(source.name_tokens, candidate.name_tokens),
        "name_shared_tokens": _shared(source.name_tokens, candidate.name_tokens),
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "name_acronym": _acronym(source.core_tokens, candidate.core_tokens),
        "addr_ratio_basic": _fuzz_ratio(source.addr_basic, candidate.addr_basic),
        "addr_ratio_translit": _fuzz_ratio(source.addr_trans, candidate.addr_trans),
        "addr_token_sort_ratio": _fuzz_token_sort(source.addr_basic, candidate.addr_basic),
        "addr_token_jaccard": _jaccard(source.addr_tokens, candidate.addr_tokens),
        "addr_shared_tokens": _shared(source.addr_tokens, candidate.addr_tokens),
        "addr_number_agree": (
            1.0 if source.premises and candidate.premises and source.premises == candidate.premises else 0.0
        ),
        "addr_number_overlap": _shared(source.numbers, candidate.numbers),
        "addr_postal_agree": 1.0 if source.postals and candidate.postals and (source.postals & candidate.postals) else 0.0,
        "addr_len_diff": addr_len_diff,
        "addr_len_ratio": addr_len_ratio,
        "country_exact": 1.0 if both_countries and source.country == candidate.country else 0.0,
        "country_missing": 1.0 if source.country_missing or candidate.country_missing else 0.0,
        "name_s1_missing": source.name_missing,
        "name_cand_missing": candidate.name_missing,
        "addr_s1_missing": source.addr_missing,
        "addr_cand_missing": candidate.addr_missing,
        "cand_is_s3": 1.0 if str(candidate_id).startswith("S3-") else 0.0,
        "n_blocking_rules": n_rules,
    }
    for (column, _), flag in zip(_RULE_FLAGS, rule_flags):
        features[column] = flag
    return features


def featurize_candidate_pairs(
    pairs: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """One feature row per blocking candidate pair, in the same order.

    `pairs` must have source1_entity_id, candidate_entity_id, and rules.
    Source frames need either the raw name/address/country columns or the
    normalized views from add_normalized_columns(). Original columns are
    not modified.
    """
    required = ("source1_entity_id", "candidate_entity_id", "rules")
    missing = [column for column in required if column not in pairs.columns]
    if missing:
        raise ValueError(f"pairs frame is missing columns: {missing}")

    if pairs.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    source_ids = pairs["source1_entity_id"].astype(str).tolist()
    candidate_ids = pairs["candidate_entity_id"].astype(str).tolist()
    rules = pairs["rules"].tolist()

    wanted = set(source_ids)
    wanted.update(candidate_ids)
    entities = _index_entities((source1, source2, source3), wanted)
    unknown = wanted.difference(entities)

    rows = []
    for source_id, candidate_id, rule_text in zip(source_ids, candidate_ids, rules):
        source = entities.get(source_id) or _empty_entity(source_id)
        candidate = entities.get(candidate_id) or _empty_entity(candidate_id)
        features = _feature_dict(source, candidate, rule_text, candidate_id)
        rows.append([features[column] for column in FEATURE_COLUMNS])

    frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    frame.insert(0, "candidate_entity_id", candidate_ids)
    frame.insert(0, "source1_entity_id", source_ids)
    frame.attrs["n_unknown_entities"] = len(unknown)
    frame.attrs["unknown_entity_ids"] = sorted(unknown)
    return frame


def attach_true_match_label(features: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Add is_true_match from train_ground_truth. Not a model feature."""
    labeled = features.copy()
    mapping = ground_truth_map(ground_truth, labeled["source1_entity_id"].tolist())
    labeled["is_true_match"] = [
        1 if candidate_id in mapping.get(source_id, ()) else 0
        for source_id, candidate_id in zip(labeled["source1_entity_id"], labeled["candidate_entity_id"])
    ]
    return labeled


def _self_check() -> None:
    columns = ["entity_id", "business_name", "business_address", "country"]
    source1 = pd.DataFrame([
        ["S1-same", "Orelee Barbershop", "10 Main Street, Austin, TX 78701", "US"],
        ["S1-order", "Blue River Cafe", "9 Lake Ave, Madison, WI", "US"],
        ["S1-suffix", "Foo Incorporated", "1 Oak Road, Dallas, TX", "US"],
        ["S1-acr", "International Business Machines", "1 New Orchard Road, Armonk, NY 10504", "US"],
        ["S1-accent", "École Primaire Sainte", "22 Rue Descartes, Calais 62100", "France"],
        ["S1-empty", "null", "NA", "null"],
        ["S1-na-name", "NA Enterprises", "5 Park Road, Pune 411001", "India"],
        ["S1-postal", "Different Sign", "99 Other Road, San Francisco, CA 94105", "US"],
    ], columns=columns)
    source2 = pd.DataFrame([
        ["S2-same", "Orelee Barbershop", "10 Main St, Austin, TX 78701", "US"],
        ["S2-order", "Cafe Blue River", "9 Lake Avenue, Madison, WI", "US"],
        ["S2-suffix", "Foo Inc", "1 Oak Rd, Dallas, TX", "US"],
        ["S2-acr", "IBM", "1 New Orchard Rd, Armonk, NY", "US"],
        ["S2-accent", "Ecole Primaire Sainte", "22 Rue Descartes, Calais", "France"],
        ["S2-empty", "NA", "none", None],
        ["S2-na-name", "NA Enterprises", "5 Park Road, Pune 411001", "India"],
        ["S2-postal", "Unrelated Shop", "10 Market Street, San Francisco, CA 94105", "US"],
    ], columns=columns)
    source3 = pd.DataFrame([
        ["S3-same", "orelee's barbershop", "10 Main Street Austin TX", "US"],
    ], columns=columns)
    pairs = pd.DataFrame([
        ["S1-same", "S2-same", "exact_name|exact_core|char_tfidf"],
        ["S1-same", "S3-same", "exact_name|name_token"],
        ["S1-order", "S2-order", "exact_sorted|name_token"],
        ["S1-suffix", "S2-suffix", "exact_core"],
        ["S1-acr", "S2-acr", "char_tfidf"],
        ["S1-accent", "S2-accent", "translit_name"],
        ["S1-empty", "S2-empty", ""],
        ["S1-na-name", "S2-na-name", "exact_name"],
        ["S1-postal", "S2-postal", "address_postal"],
        ["S1-same", "S2-not-in-index", "name_token"],
    ], columns=["source1_entity_id", "candidate_entity_id", "rules"])

    features = featurize_candidate_pairs(pairs, source1, source2, source3)
    assert list(features.columns) == list(OUTPUT_COLUMNS)
    assert len(FEATURE_COLUMNS) == 40
    assert features.attrs["n_unknown_entities"] == 1

    def row(candidate_id):
        hit = features.loc[features["candidate_entity_id"] == candidate_id].iloc[0]
        return hit

    same = row("S2-same")
    assert same["name_exact_basic"] == 1.0
    assert same["name_ratio_basic"] == 1.0
    assert same["addr_number_agree"] == 1.0
    assert same["addr_postal_agree"] == 1.0
    assert same["country_exact"] == 1.0
    assert same["country_missing"] == 0.0
    assert same["cand_is_s3"] == 0.0
    assert same["n_blocking_rules"] == 3.0
    assert same["rule_exact_name"] == 1.0
    assert same["rule_char_tfidf"] == 1.0
    assert same["rule_address_postal"] == 0.0
    assert same["name_s1_missing"] == 0.0

    other_source = row("S3-same")
    # Apostrophe becomes a token boundary ("orelee s barbershop"), so this
    # is a high name score and an S3 candidate, not an exact basic match.
    assert other_source["cand_is_s3"] == 1.0
    assert other_source["name_exact_basic"] == 0.0
    assert other_source["name_ratio_basic"] > 0.8
    assert other_source["name_shared_tokens"] >= 2.0

    ordered = row("S2-order")
    assert ordered["name_exact_basic"] == 0.0
    assert ordered["name_exact_sorted"] == 1.0
    assert ordered["name_token_sort_ratio"] == 1.0
    assert ordered["name_token_jaccard"] == 1.0

    suffix = row("S2-suffix")
    assert suffix["name_exact_basic"] == 0.0
    assert suffix["name_exact_core"] == 1.0

    acronym = row("S2-acr")
    assert acronym["name_acronym"] == 1.0
    assert acronym["name_exact_basic"] == 0.0

    france = row("S2-accent")
    assert france["country_exact"] == 1.0
    assert france["name_exact_translit"] == 1.0
    assert france["name_exact_basic"] == 0.0

    empty = row("S2-empty")
    assert empty["name_exact_basic"] == 0.0
    assert empty["name_exact_core"] == 0.0
    assert empty["name_exact_sorted"] == 0.0
    assert empty["name_exact_translit"] == 0.0
    assert math.isnan(empty["name_ratio_basic"])
    assert math.isnan(empty["name_token_jaccard"])
    assert math.isnan(empty["addr_ratio_basic"])
    assert empty["name_shared_tokens"] == 0.0
    assert empty["addr_number_agree"] == 0.0
    assert empty["addr_postal_agree"] == 0.0
    assert empty["name_s1_missing"] == 1.0
    assert empty["name_cand_missing"] == 1.0
    assert empty["addr_s1_missing"] == 1.0
    assert empty["addr_cand_missing"] == 1.0
    assert empty["country_exact"] == 0.0
    assert empty["country_missing"] == 1.0
    assert empty["n_blocking_rules"] == 0.0

    na_name = row("S2-na-name")
    assert na_name["name_s1_missing"] == 0.0
    assert na_name["name_exact_basic"] == 1.0
    assert na_name["country_exact"] == 1.0
    assert na_name["addr_postal_agree"] == 1.0

    postal = row("S2-postal")
    assert postal["addr_postal_agree"] == 1.0
    assert postal["addr_number_agree"] == 0.0
    assert postal["addr_number_overlap"] == 1.0
    assert postal["name_exact_basic"] == 0.0

    missing_candidate = row("S2-not-in-index")
    assert missing_candidate["name_cand_missing"] == 1.0
    assert math.isnan(missing_candidate["name_ratio_basic"])
    assert missing_candidate["name_exact_basic"] == 0.0

    truth = pd.DataFrame({
        "source1_entity_id": ["S1-same", "S1-order"],
        "matched_entity_ids": ["S2-same,S3-same", ""],
    })
    labeled = attach_true_match_label(features, truth)
    flags = dict(zip(labeled["candidate_entity_id"], labeled["is_true_match"]))
    assert flags["S2-same"] == 1
    assert flags["S3-same"] == 1
    assert flags["S2-order"] == 0


def main() -> None:
    _self_check()
    print(f"self-check passed ({len(FEATURE_COLUMNS)} features)")


if __name__ == "__main__":
    main()
