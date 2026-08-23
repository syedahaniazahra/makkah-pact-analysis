# X/Twitter Trend & Correlation Analysis — Project Brief (Handoff Doc)

## Original assignment
- **Problem statement:** Collect Twitter/X data and analyze topics, hashtags, keywords, or trends to identify relationships and patterns.
- **Objective:** Build a system that gathers social media data, processes it, and performs correlation or trend analysis.
- **Expected outcome:** A live, interactive dashboard showing trending topics and correlations between hashtags, keywords, users, or time periods.
- **Suggested workflow:** Data collection → cleaning → text preprocessing → topic/hashtag extraction → correlation analysis → visualization → evaluation.
- **Dataset requirements:** Tweet text, hashtags, timestamps, engagement info.
- **Deliverables:** Source code, cleaned dataset, analysis results, graphs, documentation, final report.

## Professor-approved specifics
- **Topic:** The Makkah/Mecca Joint Defence Pact (signed Aug 7, 2026, between Saudi Arabia, Turkiye, Pakistan) and its regional ripple effects — Iran's response/remarks, US impact, Pakistan's position, and the correlation among these events.
- **Scope:** Open/general, no fixed volume required.
- **Correlation priority:** Cross-actor relationships (Iran, Pakistan, US, Saudi Arabia, Turkiye) and their implications — not just hashtag-to-hashtag.
- **Deliverable format confirmed:** Live interactive dashboard (not a static report).
- **Deadline:** 3 weeks total; targeting 2 weeks as the working goal, with the 3rd week as buffer.

## Data source strategy (updated — X is now PRIMARY)
- X's official API has had no free tier since Feb 2026 (pay-per-use only, no free reads) — ruled out due to zero-cost constraint.
- Old static Kaggle datasets don't work because the topic is a live, days-old current event.
- **Revised priority (based on professor's in-person guidance): X data is the primary source and backbone of the analysis and dashboard. Reddit has been dropped entirely. GDELT is kept only as a small supporting/validation layer, not an equal-weight source.**
  1. **X/Twitter via `twscrape`** (free, open-source library) — PRIMARY source. No free official API exists, so scraping is the approved path. Needs real volume (not just a small curated sample) across all 5 actors (Iran, Pakistan, US, Saudi Arabia, Turkiye) to credibly support "primary source" framing.
     - Risk (elevated now that this is primary, not one of three equal sources): scraping account can get flagged/blocked (same underlying risk as the multi-account issue the professor originally flagged), and requires periodic maintenance as X changes internals every 2-4 weeks.
     - Mitigation: dedicated secondary account (not personal), incremental/append-only saving (never lose prior progress if collection breaks mid-run), front-load heavy collection early while account is fresh, short scoped bursts rather than continuous polling.
  2. **GDELT Project** (free, no signup) — SECONDARY/supporting only. Used to contextualize or validate patterns seen in X data (e.g. cross-checking an X mention spike against a real diplomatic event on GDELT), not as a parallel analytical pillar.
  3. ~~Reddit API~~ — dropped. Was never approved by the professor and the approval wait (2-4 weeks) doesn't fit the compressed timeline anyway.

## Phased plan (target: compress to ~2 weeks, 3-week hard deadline as buffer)
0. Environment setup (Python, VS Code, GitHub) — ~1 day
1. GDELT collection script — ~2 days
2. Reddit — apply now, build once approved (parallel, don't block on it)
3. X scraping via twscrape (dedicated account, test small, then scoped collection) — ~3-4 days, includes buffer for breakage
4. Data cleaning (pandas: timestamps, duplicates, nulls) — ~2 days
5. Text preprocessing (NLTK/spaCy: clean text, extract hashtags) — ~2 days
6. Hashtag/topic extraction + actor tagging (Iran/Pakistan/US/Saudi/Turkiye keyword matching) — ~2 days
7. Correlation analysis (cross-actor event correlation, sentiment vs engagement) — ~3 days
8. Dashboard build (Streamlit, live refresh) — ~4-5 days
9. Evaluation, documentation, final report — ~3-4 days

## Tools/stack (all free)
- Python, pandas, NLTK/spaCy, networkx, matplotlib/seaborn/plotly, Streamlit (+ Streamlit Community Cloud for free hosting), GitHub (free code hosting).

## Current status
Plan finalized, professor has approved topic and data-source approach. About to begin **Phase 0: environment setup**. No code written yet.
