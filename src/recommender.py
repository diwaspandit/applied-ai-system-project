import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict, dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked for the given user."""
        ranked_songs = recommend_songs(
            asdict(user),
            [asdict(song) for song in self.songs],
            k=k,
        )
        songs_by_id = {song.id: song for song in self.songs}
        return [songs_by_id[song_dict["id"]] for song_dict, _, _ in ranked_songs]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Summarize why a song matched the user's preferences."""
        score, reasons = score_song(asdict(user), asdict(song))
        reason_text = ", ".join(reasons)
        return f"Score {score:.2f}: {reason_text}"

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into typed dictionaries."""
    print(f"Loading songs from {csv_path}...")

    songs: List[Dict] = []
    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            song = dict(row)

            for field in int_fields:
                song[field] = int(song[field])

            for field in float_fields:
                song[field] = float(song[field])

            songs.append(song)

    return songs

def _get_preference(user_prefs: Dict, preferred_key: str, legacy_key: str, default=None):
    """Read a user preference while supporting legacy key names."""
    return user_prefs.get(preferred_key, user_prefs.get(legacy_key, default))

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song and return the score plus explanation snippets."""
    reasons: List[str] = []
    score = 0.0

    favorite_genre = _get_preference(user_prefs, "favorite_genre", "genre")
    favorite_mood = _get_preference(user_prefs, "favorite_mood", "mood")
    target_energy = float(_get_preference(user_prefs, "target_energy", "energy", 0.0))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    if song["mood"] == favorite_mood:
        score += 3.0
        reasons.append("mood match (+3.0)")

    if song["genre"] == favorite_genre:
        score += 1.0
        reasons.append("genre match (+1.0)")

    energy_fit = max(0.0, min(1.0, 1 - abs(float(song["energy"]) - target_energy)))
    energy_points = 6.0 * energy_fit
    score += energy_points
    reasons.append(f"energy close to target (+{energy_points:.1f})")

    acousticness = float(song["acousticness"])
    acoustic_fit = acousticness if likes_acoustic else (1 - acousticness)
    acoustic_fit = max(0.0, min(1.0, acoustic_fit))
    acoustic_points = 2.0 * acoustic_fit
    score += acoustic_points

    if likes_acoustic:
        reasons.append(f"matches acoustic preference (+{acoustic_points:.1f})")
    else:
        reasons.append(f"matches less-acoustic preference (+{acoustic_points:.1f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank the catalog by score and return the top-k recommendations."""
    favorite_mood = _get_preference(user_prefs, "favorite_mood", "mood")
    favorite_genre = _get_preference(user_prefs, "favorite_genre", "genre")

    ranked_songs = sorted(
        (
            (
                song,
                score,
                ", ".join(reasons),
                song["mood"] == favorite_mood,
                song["genre"] == favorite_genre,
            )
            for song in songs
            for score, reasons in [score_song(user_prefs, song)]
        ),
        key=lambda item: (item[1], item[3], item[4]),
        reverse=True,
    )

    return [(song, score, explanation) for song, score, explanation, _, _ in ranked_songs[:k]]
