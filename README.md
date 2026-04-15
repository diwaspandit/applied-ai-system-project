# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Real-world recommendation systems usually combine many signals at once: a user's past behavior, patterns from similar users, content features like genre or mood, and a ranking step that decides what to show first. My version is a much smaller, more transparent content-based simulation. Instead of learning from millions of listeners, it will prioritize musical "vibe" by rewarding mood and genre matches, songs whose energy is close to the user's target, and whether the song fits the user's acoustic preference.

---

## How The System Works

This simulation uses a simple content-based recommender that compares each song's features to a small user taste profile and then ranks the best matches.

`Song` features used in the simulation:
- `id`
- `title`
- `artist`
- `genre`
- `mood`
- `energy`
- `tempo_bpm`
- `valence`
- `danceability`
- `acousticness`

`UserProfile` features used in the simulation:
- `favorite_genre`
- `favorite_mood`
- `target_energy`
- `likes_acoustic`

Example taste profile dictionary:

```python
user_prefs = {
    "favorite_genre": "lofi",
    "favorite_mood": "focused",
    "target_energy": 0.40,
    "likes_acoustic": True,
}
```

The recommender gives points when a song matches the user's favorite genre and mood, rewards songs whose `energy` is closer to the user's target energy, and can add a smaller bonus when the song's `acousticness` fits the user's acoustic preference. After every song gets a score, the system sorts the songs from highest to lowest score and recommends the top `k` songs.

Algorithm recipe:
- Read the user's taste profile from `user_prefs`
- Load every song from `data/songs.csv`
- For each song, calculate a score using:
  - `+3` points if the mood matches
  - `+2` points if the genre matches
  - up to `+3` points for energy closeness using `1 - abs(song_energy - target_energy)`
  - up to `+2` points for acoustic fit
- Save each song with its final score
- Sort all songs from highest score to lowest score
- Return the top `k` songs as recommendations

In this design, mood is weighted slightly more than genre because the system is trying to match overall vibe, not just musical category. A song from a different genre can still feel right if it matches the user's mood and energy.

Potential bias note:
- This system might over-prioritize exact mood or genre labels and miss good songs that feel similar in other ways.
- It may also over-recommend acoustic songs for users with `likes_acoustic = True`, even when a less acoustic track matches the mood better.
- Because the catalog is small, underrepresented genres and moods may appear less often in the final recommendations.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"
