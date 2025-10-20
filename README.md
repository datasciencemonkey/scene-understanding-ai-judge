# 🎬 Scene Understanding AI Judge

AI-powered video analysis tool that evaluates generated video frames against ground truth descriptions using Google's Gemini 2.5 Flash model via Databricks.

## 🚀 What It Does

This tool provides automated quality assessment for AI-generated videos by:

1. **Extracting frames** from video files at configurable FPS rates
2. **Analyzing each frame** using Gemini 2.5 Flash through Databricks endpoints
3. **Scoring frames** across three dimensions:
   - **Correctness**: How well the frame matches the ground truth
   - **Coherence**: How well the frame fits the expected sequence
   - **Faithfulness**: How accurate the frame is to ground truth elements
4. **Generating reports** with aggregate scores and detailed per-frame analysis

## 📋 Prerequisites

- Python 3.12+
- Databricks account with Personal Access Token
- `uv` package manager

## ⚙️ Setup

```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Configure your Databricks token
echo "DATABRICKS_TOKEN=your_token_here" > .env
```

## 🎯 Usage

### Option 1: Web UI (Recommended)

```bash
uv run streamlit run streamlit_app.py
```

Launch the interactive Streamlit web interface featuring:
- **Two-tab layout**: Extract frames and analyze in separate tabs
- **Video selection**: Choose from multiple test videos
- **Configurable FPS**: Select extraction rate (0.5-3 fps)
- **Real-time analysis**: Watch frames being analyzed with live summaries
- **Interactive results**: View scores, descriptions, and download JSON reports

### Option 2: CLI Tools

#### Step 1: Extract Frames

```bash
uv run 01-frame-split.py
```

Extracts frames from `veo3-generations.mp4` at your chosen FPS rate (0.5, 1, 2, or custom).

#### Step 2: Analyze Frames

```bash
uv run 02-app.py
```

Analyzes all extracted frames and generates:
- Real-time progress display with per-frame scores
- Final aggregate report with averages
- JSON export (`frame_analysis_results.json`)

## 📊 Output

The analysis provides:
- **Per-frame scores** (1-10 scale) for correctness, coherence, and faithfulness
- **Frame descriptions** detailing what the AI sees
- **Running summary** that builds context across frames
- **Aggregate metrics** averaging all dimensions
- **Overall quality rating** with visual indicators

## 🏗️ Architecture

- **streamlit_app.py**: Interactive web UI with two-tab interface
- **helpers.py**: Shared utilities for frame extraction and analysis (uses MoviePy)
- **01-frame-split.py**: CLI tool for video preprocessing using MoviePy
- **02-app.py**: CLI analysis engine with rich terminal UI
- Uses base64 encoding for image transmission
- Pydantic models for structured AI responses

## 📝 Notes

Ground truth comparison and scoring criteria are customizable in `02-app.py`.
