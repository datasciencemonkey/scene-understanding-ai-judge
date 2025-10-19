# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based video analysis application that uses Databricks endpoints to analyze video content through the Gemini 2.5 Flash model via OpenAI-compatible API. The application encodes videos to base64 and sends them to a hosted AI model for content analysis.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main video analysis script
uv run 02-app.py
```

## Architecture

### Core Components
- **app.py**: Main application file containing video encoding and AI analysis logic
- **requirements.txt**: Python dependencies (openai, python-dotenv)
- **.env**: Environment configuration (contains DATABRICKS_TOKEN)
- **.venv/**: Python virtual environment with installed packages

### Key Functions
- `encode_video(video_path)`: Converts video files to base64 encoding for API transmission
- Video analysis using Databricks-hosted Gemini 2.5 Flash model through OpenAI compatibility layer

### Configuration
- Uses Databricks Personal Access Token for authentication
- Databricks endpoint: `https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints`
- Target video file: `veo3-generations.mp4`

## Environment Variables
- `DATABRICKS_TOKEN`: Required for API authentication (stored in .env file)
- always use uv run anything or install. its the recommended package and dependency manager. install with uv pip install -r requirements.txt