"""Schema invariants.

The channel partition is the load-bearing assumption of the whole experiment:
if the tabular and narrative channels overlap, the text modality becomes a
re-encoding of the tabular one and every multimodal comparison in this project
becomes meaningless. It is therefore tested, not merely documented.
"""

from __future__ import annotations

import pytest

from src.data import schema


def test_channels_are_disjoint():
    schema.assert_disjoint_channels()


def test_no_field_name_is_defined_twice():
    names = [f.name for f in schema.ALL_FIELDS]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate field definitions: {sorted(duplicates)}"


def test_no_source_column_is_mapped_twice():
    sources = [f.source_column for f in schema.ALL_FIELDS if f.source_column]
    duplicates = {s for s in sources if sources.count(s) > 1}
    assert not duplicates, f"source columns mapped more than once: {sorted(duplicates)}"


def test_tabular_and_narrative_feature_lists_match_their_field_groups():
    assert [f.name for f in schema.TABULAR_FIELDS] == schema.TABULAR_FEATURES
    assert [f.name for f in schema.NARRATIVE_FIELDS] == schema.NARRATIVE_FEATURES


def test_identifiers_are_not_features():
    """Policy number is a monotonic row index in the source data.

    Using it as a feature would hand the model a proxy for time, which under a
    temporal split is a direct leak.
    """
    identifier_names = {f.name for f in schema.IDENTIFIERS}
    assert not identifier_names & set(schema.TABULAR_FEATURES)
    assert not identifier_names & set(schema.NARRATIVE_FEATURES)


def test_target_is_not_among_the_features():
    assert schema.TARGET not in schema.TABULAR_FEATURES
    assert schema.TARGET not in schema.NARRATIVE_FEATURES


def test_fairness_attributes_exist_in_the_schema():
    for attribute in schema.FAIRNESS_ATTRIBUTES:
        assert attribute in schema.FIELDS_BY_NAME, attribute


def test_disjointness_assertion_actually_fires(monkeypatch):
    """A guard that cannot fail is not a guard."""
    monkeypatch.setattr(
        schema, "TABULAR_FEATURES", [*schema.TABULAR_FEATURES, "fault"]
    )
    with pytest.raises(AssertionError, match="overlap"):
        schema.assert_disjoint_channels()


def test_damage_fields_are_marked_real():
    """Damage labels come from the image dataset and are never invented."""
    for name in ("damage_location", "damage_condition"):
        assert schema.FIELDS_BY_NAME[name].provenance is schema.Provenance.REAL


def test_claim_text_is_marked_constructed():
    """The narratives are generated, and the schema must say so."""
    assert schema.FIELDS_BY_NAME["claim_text"].provenance is schema.Provenance.CONSTRUCTED
