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
        user_prefs = asdict(user)
        scored_songs = []

        for song in self.songs:
            song_dict = asdict(song)
            score, _ = score_song(user_prefs, song_dict)
            scored_songs.append((song, score))

        scored_songs.sort(key=lambda item: item[1], reverse=True)
        return [song for song, _ in scored_songs[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = score_song(asdict(user), asdict(song))
        reason_text = ", ".join(reasons)
        return f"Score {score:.2f}: {reason_text}"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
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

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores one song against a user's taste profile.
    Returns the numeric score and a list of human-readable reasons.
    """
    reasons: List[str] = []
    score = 0.0

    favorite_genre = user_prefs.get("favorite_genre", user_prefs.get("genre"))
    favorite_mood = user_prefs.get("favorite_mood", user_prefs.get("mood"))
    target_energy = float(user_prefs.get("target_energy", user_prefs.get("energy", 0.0)))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    if song["mood"] == favorite_mood:
        score += 3.0
        reasons.append("mood match (+3.0)")

    if song["genre"] == favorite_genre:
        score += 2.0
        reasons.append("genre match (+2.0)")

    energy_fit = max(0.0, 1 - abs(float(song["energy"]) - target_energy))
    energy_points = 3.0 * energy_fit
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
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored_songs: List[Tuple[Dict, float, str]] = []

    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = ", ".join(reasons)
        scored_songs.append((song, score, explanation))

    scored_songs.sort(key=lambda item: item[1], reverse=True)
    return scored_songs[:k]
