# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-09-13

### Added
* Groq API integration using `openai/gpt-oss-20b` for Swahili and Sheng dialect processing.
* Client-side Markdown rendering via `Marked.js` in `src/web.py`.
* Enforcement of bilingual (English & Swahili/Sheng) model system prompts.
* Automated benchmark history tracking in `benchmarks/history.json` and auto-syncing metrics in `README.md`.

### Fixed
* Vector dimension mismatch guard in `SautiEngine.compute_cosine_similarity`.
* UTF-8 charset encoding headers in web server responses.
