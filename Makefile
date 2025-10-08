# Video Review Analysis Pipeline
# Usage: make judge

.PHONY: judge split analyze clean help

# Default target
help:
	@echo "🎬 Video Review Analysis Pipeline"
	@echo "=================================="
	@echo "Available targets:"
	@echo "  judge     - Run complete pipeline (split frames + analyze)"
	@echo "  split     - Extract frames from video only"
	@echo "  analyze   - Analyze existing frames only"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"

# Main target: run complete pipeline
judge: split analyze
	@echo ""
	@echo "🎉 Pipeline completed successfully!"
	@echo "📊 Check frame_analysis_results.json for detailed results"

# Extract frames from video
split:
	@echo "🎬 Starting frame extraction..."
	@uv run 01-frame-split.py
	@echo "✅ Frame extraction completed"

# Analyze extracted frames
analyze:
	@echo ""
	@echo "🔍 Starting frame analysis..."
	@if [ ! -d "data" ] || [ -z "$$(ls -A data 2>/dev/null)" ]; then \
		echo "❌ Error: No frames found in data/ directory"; \
		echo "   Please run 'make split' first or check if frames were extracted"; \
		exit 1; \
	fi
	@uv run 02-app.py
	@echo "✅ Frame analysis completed"

# Clean up generated files
clean:
	@echo "🧹 Cleaning up generated files..."
	@if [ -d "data" ]; then \
		rm -rf data/; \
		echo "  ✓ Removed data/ directory"; \
	fi
	@if [ -f "frame_analysis_results.json" ]; then \
		rm -f frame_analysis_results.json; \
		echo "  ✓ Removed frame_analysis_results.json"; \
	fi
	@echo "✅ Cleanup completed"

# Check prerequisites
check-deps:
	@echo "🔍 Checking dependencies..."
	@command -v uv >/dev/null 2>&1 || { echo "❌ uv is required but not installed. Please install uv first."; exit 1; }
	@command -v ffmpeg >/dev/null 2>&1 || { echo "❌ ffmpeg is required but not installed. Please install with: brew install ffmpeg"; exit 1; }
	@if [ ! -f "veo3-generations.mp4" ]; then \
		echo "❌ Video file 'veo3-generations.mp4' not found"; \
		exit 1; \
	fi
	@if [ ! -f ".env" ]; then \
		echo "❌ .env file not found. Please create .env with DATABRICKS_TOKEN"; \
		exit 1; \
	fi
	@echo "✅ All dependencies are ready"

# Enhanced judge target with dependency checks
judge-safe: check-deps split analyze
	@echo ""
	@echo "🎉 Pipeline completed successfully!"
	@echo "📊 Check frame_analysis_results.json for detailed results"