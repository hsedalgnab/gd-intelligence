# GD Intelligence
AI-powered legal document intelligence system for Gibson, Dunn & Crutcher LLP.
Built on Retrieval-Augmented Generation (RAG) technology.

## Features
- Multi-document PDF indexing
- Natural language search across case files
- Source document references with page numbers
- Streaming responses

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file:
   - `API_KEY`
   - `BASE_URL`
   - `MODEL`
4. Add documents to `documents/` folder
5. Run indexer: `python indexer.py`
6. Run app: `streamlit run app.py`

## Project
Part of [Project Legal Intelligence](https://github.com/users/hsedalgnab/projects/1) — a large-scale RAG initiative for law firms.