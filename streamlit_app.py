#!/usr/bin/env python3
# ABOUTME: Streamlit web app for video frame extraction and AI-based quality analysis
# ABOUTME: Allows users to select videos, extract frames at custom FPS, and score them against ground truth prompts

import base64
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from urllib.parse import urljoin

# Load environment variables
load_dotenv()
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
base_url = urljoin(DATABRICKS_HOST, "serving-endpoints")

current_dir = Path(__file__).parent


# Pydantic model for structured analysis output
class FrameAnalysis(BaseModel):
    """Structured output for frame analysis"""

    correctness_score: int = Field(
        ..., ge=1, le=10, description="How well the frame matches the ground truth"
    )
    coherence_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="How coherent the frame is with the video sequence",
    )
    faithfulness_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="How faithful the frame is to ground truth elements",
    )
    frame_description: str = Field(
        ..., description="Detailed description of what is visible in the frame"
    )
    updated_summary: str = Field(
        ...,
        description="Comprehensive summary incorporating this and all previous frames",
    )


# Initialize OpenAI client for Databricks
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=base_url,
)


def load_video_prompts():
    """Load video prompts from JSON file"""
    with open(current_dir / "videos/video_prompts.json", "r") as f:
        return json.load(f)


def clear_data_folder(output_dir):
    """Clear all contents of the data folder"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from video at specified fps using FFmpeg

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)

    Returns:
        tuple: (success: bool, frame_count: int, error_message: str)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        f"{output_dir}/frame_%03d.png",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            frame_files = [
                f
                for f in os.listdir(output_dir)
                if f.startswith("frame_") and f.endswith(".png")
            ]
            return True, len(frame_files), None
        else:
            return False, 0, result.stderr

    except FileNotFoundError:
        return (
            False,
            0,
            "FFmpeg not found. Please install FFmpeg first (brew install ffmpeg)",
        )
    except Exception as e:
        return False, 0, str(e)


def encode_image(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_frame(
    image_path: str, frame_number: int, ground_truth: str, previous_summary: str = ""
) -> FrameAnalysis:
    """Analyze a single frame and return scoring and description"""

    base64_image = encode_image(image_path)

    analysis_prompt = f"""
    You are a judge evaluating the quality and coherence of generated video frames.

    Ground Truth: "{ground_truth}"

    Previous Summary (if any): {previous_summary}

    Please analyze this frame (Frame #{frame_number}) and provide:

    1. CORRECTNESS SCORE (1-10): How well does this frame match the ground truth description?
    2. COHERENCE SCORE (1-10): How coherent is this frame with the expected video sequence?
    3. FAITHFULNESS SCORE (1-10): How faithful is this frame to the ground truth elements?
    4. FRAME DESCRIPTION: Detailed description of what you see in this frame
    5. UPDATED SUMMARY: Based on this frame and previous frames, provide an updated summary of the overall video content
    """

    try:
        response = client.beta.chat.completions.parse(
            model="o4-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            response_format=FrameAnalysis,
        )

        return response.choices[0].message.parsed

    except Exception as e:
        st.error(f"Error analyzing frame {frame_number}: {e}")
        return None


def analyze_all_frames(ground_truth: str, summary_placeholder=None) -> Dict:
    """Analyze all frames in the data directory"""
    frame_files = sorted(glob.glob("data/frame_*.png"))

    if not frame_files:
        return {}

    results = []
    current_summary = ""

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, frame_path in enumerate(frame_files):
        frame_number = idx + 1
        status_text.text(f"Analyzing frame {frame_number}/{len(frame_files)}...")

        frame_result = analyze_frame(frame_path, frame_number, ground_truth, current_summary)

        if frame_result:
            frame_dict = frame_result.model_dump()
            frame_dict["frame_number"] = frame_number
            frame_dict["frame_path"] = frame_path
            results.append(frame_dict)
            current_summary = frame_dict["updated_summary"]

            # Update summary in real-time if placeholder provided
            if summary_placeholder:
                summary_placeholder.info(f"**Current Summary (after frame {frame_number}):**\n\n{current_summary}")

        progress_bar.progress((idx + 1) / len(frame_files))

    status_text.text("Analysis complete!")

    # Calculate aggregate scores
    total_frames = len(results)
    if total_frames == 0:
        return {}

    avg_correctness = sum(r["correctness_score"] for r in results) / total_frames
    avg_coherence = sum(r["coherence_score"] for r in results) / total_frames
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / total_frames
    overall_score = (avg_correctness + avg_coherence + avg_faithfulness) / 3

    analysis_summary = {
        "ground_truth": ground_truth,
        "total_frames": total_frames,
        "aggregate_scores": {
            "average_correctness": round(avg_correctness, 2),
            "average_coherence": round(avg_coherence, 2),
            "average_faithfulness": round(avg_faithfulness, 2),
            "overall_score": round(overall_score, 2),
        },
        "final_summary": current_summary if current_summary else "No summary generated",
        "frame_results": results,
    }

    return analysis_summary


def main():
    st.set_page_config(
        page_title="Video Quality Analyzer",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Video Quality Analyzer")
    st.markdown("---")

    # Load video prompts
    video_prompts = load_video_prompts()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Video selection
        video_name = st.selectbox(
            "Select Video",
            options=list(video_prompts.keys()),
            format_func=lambda x: x.title()
        )

        # FPS selection
        fps_options = {
            "0.5 fps (1 frame every 2 seconds) - Very sparse": 0.5,
            "1 fps (1 frame per second) - Default": 1.0,
            "2 fps (2 frames per second) - Dense": 2.0,
            "3 fps (3 frames per second) - Very dense": 3.0,
        }

        fps_label = st.selectbox(
            "Frame Extraction Rate",
            options=list(fps_options.keys())
        )
        fps = fps_options[fps_label]

        st.markdown("---")

        # Display selected configuration
        st.subheader("📋 Selected Configuration")
        st.write(f"**Video:** {video_name.title()}")
        st.write(f"**FPS:** {fps}")

    # Check if frames exist
    output_dir = current_dir / "data"
    frame_files = sorted(glob.glob(str(output_dir / "frame_*.png")))
    frames_exist = len(frame_files) > 0

    # Initialize session state for extraction success message
    if "extraction_success" not in st.session_state:
        st.session_state.extraction_success = None

    # Initialize session state for analysis results
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    # Create tabs FIRST
    tab1, tab2 = st.tabs(["🎞️ Extract Frames", "🔍 Analyze Frames"])

    # TAB 1: Frame Extraction
    with tab1:
        # Define video path first (needed in both columns)
        video_path = current_dir / "videos" / video_prompts[video_name]["save_path"]

        # Video info and prompt at top
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📝 Video Prompt (Ground Truth)")
            st.info(video_prompts[video_name]["prompt"])

            # Extract button right below prompt
            extract_button = st.button("🎞️ Extract Frames", use_container_width=True, type="primary", key="extract_btn")

            # Show extraction results or status right below button
            if extract_button:
                with st.spinner("Clearing previous frames..."):
                    clear_data_folder(output_dir)

                with st.spinner(f"Extracting frames at {fps} fps..."):
                    success, frame_count, error_msg = extract_frames(
                        str(video_path), str(output_dir), fps
                    )

                if success:
                    st.session_state.extraction_success = {
                        "success": True,
                        "frame_count": frame_count
                    }
                    st.rerun()  # Rerun to update frames_exist state
                else:
                    st.session_state.extraction_success = {
                        "success": False,
                        "error_msg": error_msg
                    }

            # Display success/error messages from session state
            if st.session_state.extraction_success:
                if st.session_state.extraction_success["success"]:
                    st.success(f"✅ Successfully extracted {st.session_state.extraction_success['frame_count']} frames!")
                    st.info("👉 Switch to 'Analyze Frames' tab to start the AI analysis")
                else:
                    st.error(f"❌ Frame extraction failed: {st.session_state.extraction_success['error_msg']}")

        with col2:
            st.subheader("🎥 Video Info")

            if os.path.exists(video_path):
                video_size = os.path.getsize(video_path) / (1024 * 1024)
                st.success(f"✅ Video file found ({video_size:.1f} MB)")

                # Display video preview
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"❌ Video file not found: {video_path}")

        # Show extracted frames section
        if frames_exist:
            st.markdown("---")
            st.subheader(f"📸 Extracted Frames ({len(frame_files)} total)")

            # Display all frames in a grid
            cols_per_row = 4
            for i in range(0, len(frame_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(frame_files):
                        with col:
                            st.image(frame_files[idx], caption=f"Frame {idx + 1}", use_container_width=True)

    # TAB 2: Frame Analysis
    with tab2:
        # Prompt at top (no video preview)
        st.subheader("📝 Video Prompt (Ground Truth)")
        st.info(video_prompts[video_name]["prompt"])

        if not frames_exist:
            st.warning("⚠️ No frames found. Please extract frames first.")
            st.info("👈 Go to the 'Extract Frames' tab to extract frames before analyzing")
        else:
            # Analyze button right below prompt
            analyze_button = st.button("🔍 Analyze Frames", use_container_width=True, type="primary", key="analyze_btn")

            # Run analysis when button is clicked
            if analyze_button:
                st.info(f"Found {len(frame_files)} frames to analyze")

                # Create placeholder for real-time summary updates
                summary_placeholder = st.empty()

                with st.spinner("Analyzing frames with AI..."):
                    analysis_results = analyze_all_frames(
                        video_prompts[video_name]["prompt"],
                        summary_placeholder=summary_placeholder
                    )

                if analysis_results:
                    # Store results in session state
                    st.session_state.analysis_results = analysis_results
                    st.rerun()

            # Display results from session state (persists across reruns)
            if st.session_state.analysis_results:
                analysis_results = st.session_state.analysis_results

                st.success("✅ Analysis complete!")

                # Display aggregate scores
                st.subheader("📊 Aggregate Scores")

                scores = analysis_results["aggregate_scores"]

                # Create metrics in columns
                metric_cols = st.columns(4)

                with metric_cols[0]:
                    st.metric(
                        "Correctness",
                        f"{scores['average_correctness']}/10",
                        delta=None
                    )

                with metric_cols[1]:
                    st.metric(
                        "Coherence",
                        f"{scores['average_coherence']}/10",
                        delta=None
                    )

                with metric_cols[2]:
                    st.metric(
                        "Faithfulness",
                        f"{scores['average_faithfulness']}/10",
                        delta=None
                    )

                with metric_cols[3]:
                    st.metric(
                        "Overall Score",
                        f"{scores['overall_score']}/10",
                        delta=None
                    )

                # Display final summary
                st.subheader("📝 Final Summary")
                st.write(analysis_results["final_summary"])

                # Display frame-by-frame results
                st.subheader("🎞️ Frame-by-Frame Analysis")

                for result in analysis_results["frame_results"]:
                    with st.expander(f"Frame {result['frame_number']} - Scores: C:{result['correctness_score']} | Co:{result['coherence_score']} | F:{result['faithfulness_score']}"):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.image(result["frame_path"], use_container_width=True)

                        with col2:
                            st.write("**Description:**")
                            st.write(result["frame_description"])

                            st.write("**Scores:**")
                            score_col1, score_col2, score_col3 = st.columns(3)
                            with score_col1:
                                st.metric("Correctness", f"{result['correctness_score']}/10")
                            with score_col2:
                                st.metric("Coherence", f"{result['coherence_score']}/10")
                            with score_col3:
                                st.metric("Faithfulness", f"{result['faithfulness_score']}/10")

                # Save results
                results_file = "frame_analysis_results.json"
                with open(results_file, "w") as f:
                    json.dump(analysis_results, f, indent=2)

                st.success(f"💾 Results saved to {results_file}")

                # Download button
                st.download_button(
                    label="⬇️ Download Results JSON",
                    data=json.dumps(analysis_results, indent=2),
                    file_name=results_file,
                    mime="application/json",
                    key="download_results_btn"
                )

            # Show all extracted frames at the bottom
            st.markdown("---")
            st.subheader(f"📸 Extracted Frames ({len(frame_files)} frames)")

            # Display all frames in a grid
            cols_per_row = 4
            for i in range(0, len(frame_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(frame_files):
                        with col:
                            st.image(frame_files[idx], caption=f"Frame {idx + 1}", use_container_width=True)


if __name__ == "__main__":
    main()
