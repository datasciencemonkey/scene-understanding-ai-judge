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
- FFmpeg (for frame extraction)
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

### Step 1: Extract Frames

```bash
uv run 01-frame-split.py
```

Extracts frames from `veo3-generations.mp4` at your chosen FPS rate (0.5, 1, 2, or custom).

### Step 2: Analyze Frames

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

- **01-frame-split.py**: Video preprocessing using FFmpeg
- **02-app.py**: Main analysis engine using OpenAI-compatible API
- Uses base64 encoding for image transmission
- Rich terminal UI with progress tracking and color-coded scores

## 📝 Notes

Ground truth comparison and scoring criteria are customizable in `02-app.py`.
