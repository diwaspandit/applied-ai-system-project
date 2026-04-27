from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class CatalogOptions:
    """UI-ready option lists derived from the loaded song catalog."""

    genres: List[str]
    moods: List[str]
    energy_range: Tuple[float, float]


def build_user_profile(
    favorite_genre: str,
    favorite_mood: str,
    target_energy: float,
    likes_acoustic: bool,
) -> Dict:
    """Build the profile dictionary expected by the recommendation pipeline."""
    return {
        "favorite_genre": favorite_genre.strip().lower(),
        "favorite_mood": favorite_mood.strip().lower(),
        "target_energy": float(target_energy),
        "likes_acoustic": bool(likes_acoustic),
    }


def catalog_options(songs: Iterable[Dict]) -> CatalogOptions:
    """Return sorted genres, moods, and energy limits from songs."""
    song_list = list(songs)
    genres = sorted({str(song["genre"]) for song in song_list})
    moods = sorted({str(song["mood"]) for song in song_list})
    energies = [float(song["energy"]) for song in song_list]
    if not energies:
        return CatalogOptions(genres=[], moods=[], energy_range=(0.0, 1.0))
    return CatalogOptions(
        genres=genres,
        moods=moods,
        energy_range=(min(energies), max(energies)),
    )


def confidence_label(confidence: float) -> str:
    """Map numeric confidence into a compact UI label."""
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def status_label(status: str) -> str:
    """Normalize agent step status text for display."""
    normalized = status.strip().lower()
    if normalized == "passed":
        return "PASS"
    if normalized == "warning":
        return "WARN"
    if normalized == "failed":
        return "FAIL"
    return normalized.upper() or "UNKNOWN"
