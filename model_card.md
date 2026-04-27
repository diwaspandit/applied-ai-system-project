# Model Card: VibeFinder Gemini RAG Music Recommender

## 1. Model Name

**VibeFinder Gemini RAG**

## 2. Intended Use

VibeFinder recommends songs from a small classroom catalog based on a user's preferred genre, mood, energy level, and acoustic preference. It is intended for portfolio and learning use, not for production music discovery or real user profiling. It can be used from either the CLI or the Streamlit browser UI.

The new AI behavior is explanation generation plus an observable agent workflow. The system first plans the recommendation steps, ranks songs with deterministic scoring, retrieves local fictional facts about the recommended songs, and then asks Gemini to produce a grounded JSON explanation. If Gemini is unavailable, a deterministic fallback generator explains the recommendation from the same retrieved context.

## 3. How the System Works

The recommender scores each song using:

- mood match
- genre match
- energy closeness
- acoustic preference fit

The RAG layer then retrieves song and artist facts from `data/music_knowledge.csv`. The generator is instructed to use only those facts, return JSON, include citations, and avoid claiming live web access. The output parser validates confidence, citations, empty answers, malformed JSON, and unsupported web-search claims.

The agent layer records intermediate steps: planning, profile validation, the RAG tool call, grounding self-check, fallback decision when needed, and final answer selection. This makes the workflow auditable instead of hiding the AI process behind one response.

## 4. Data

The catalog contains 18 fictional songs in `data/songs.csv`. The RAG knowledge base contains 22 local facts about the fictional songs and artists. Genres include pop, lofi, rock, ambient, jazz, synthwave, country, hip hop, classical, reggae, metal, folk, EDM, and R&B.

Because the data is small and fictional, it is useful for demonstrating architecture and reliability, but it does not represent real listener behavior or real artist histories.

## 5. Strengths

- The scoring logic is transparent and easy to test.
- The RAG layer gives explanations that are grounded in local facts instead of unsupported general claims.
- The generator boundary supports Gemini in normal use and fake/local generators in tests.
- The agent trace shows which tools were used and whether outputs passed grounding checks.
- The CLI, Streamlit UI, and evaluation script run even without an API key, which makes the project reproducible.

## 6. Limitations and Bias

The system can over-prioritize exact labels. For example, a user who asks for a mood or genre that appears only once in the catalog has very few good options. The recommender also assumes that mood, genre, energy, and acousticness are enough to describe taste, which leaves out lyrics, culture, language, listening history, era, and user context.

The local knowledge base is manually written, so it reflects the author's idea of each fictional song. If this were used on real music, the system could misrepresent artists or over-amplify popular genres unless the data source and retrieval quality were carefully audited.

## 7. Reliability and Evaluation

Reliability mechanisms include:

- unit tests for scoring, retrieval, prompt building, Gemini boundary behavior, fallback generation, guardrail parsing, and end-to-end pipeline behavior
- validation for required profile fields
- clipping for out-of-range energy values
- fallback generation when Gemini is unavailable or returns malformed output
- citation checks and live-web claim cleanup
- observable agent planning/tool/self-check traces
- an evaluation script with three predefined profiles

Current local result:

```text
pytest -q
19 passed

python scripts/evaluate_recommender.py
Passed 3 out of 3 cases
```

The most surprising reliability result was that the deterministic fallback remained useful when Gemini was not available. Instead of failing the demo, the app still ranked songs, retrieved local facts, generated a grounded explanation, and clearly labeled the fallback guardrail.

## 8. Misuse Risks

The main misuse risk is presenting a small classroom demo as a real recommendation engine. Another risk is making generated explanations sound more authoritative than the underlying data supports. To reduce that risk, the system cites local facts, labels guardrail behavior, avoids live-web claims, and keeps confidence visible.

## 9. Future Work

- Add a real source-backed web-search agent for current music releases.
- Track diversity so the top recommendations do not become too similar.
- Add richer user preference fields such as language, era, lyrical theme, or listening context.
- Improve confidence scoring with measured retrieval coverage instead of a simple heuristic.
- Add generated comparison explanations that show why a runner-up was not selected.
- Add a specialization experiment or fine-tuning comparison; this project currently uses structured prompting, not fine-tuning.

## 10. AI Collaboration Reflection

AI was helpful for converting the rubric into a modular build plan: separate scoring, retrieval, generation, guardrails, evaluation, and documentation. It was also useful for identifying test cases around malformed model JSON and missing credentials.

One flawed AI direction would have been to read the Gemini key from `document.md`. That would have made the app easier to run on one machine, but it would be a poor security practice and would couple the code to an uncommitted assignment note. The final version uses `GEMINI_API_KEY` from the environment instead.
