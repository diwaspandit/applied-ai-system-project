from src.ui_helpers import (
    build_user_profile,
    catalog_options,
    confidence_label,
    status_label,
)


def test_build_user_profile_normalizes_text_and_types():
    profile = build_user_profile(
        favorite_genre=" Pop ",
        favorite_mood=" Happy ",
        target_energy=0.82,
        likes_acoustic=False,
    )

    assert profile == {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.82,
        "likes_acoustic": False,
    }


def test_catalog_options_sorts_values_and_reports_energy_range():
    options = catalog_options(
        [
            {"genre": "rock", "mood": "intense", "energy": 0.9},
            {"genre": "pop", "mood": "happy", "energy": 0.4},
        ]
    )

    assert options.genres == ["pop", "rock"]
    assert options.moods == ["happy", "intense"]
    assert options.energy_range == (0.4, 0.9)


def test_catalog_options_handles_empty_catalog():
    options = catalog_options([])

    assert options.genres == []
    assert options.moods == []
    assert options.energy_range == (0.0, 1.0)


def test_confidence_label_groups_scores():
    assert confidence_label(0.8) == "High"
    assert confidence_label(0.5) == "Medium"
    assert confidence_label(0.2) == "Low"


def test_status_label_normalizes_known_and_unknown_statuses():
    assert status_label("passed") == "PASS"
    assert status_label("warning") == "WARN"
    assert status_label("failed") == "FAIL"
    assert status_label("custom") == "CUSTOM"
