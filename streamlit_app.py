#!/usr/bin/env python3
# ABOUTME: Streamlit web app for video frame extraction and AI-based quality analysis
# ABOUTME: Allows users to select videos, extract frames at custom FPS, and score them against ground truth prompts

import glob
import json
import os
from pathlib import Path
from typing import Dict

import streamlit as st

from helpers import (analyze_frame, clear_data_folder, extract_frames,
                     get_openai_client, load_video_prompts)

current_dir = Path(__file__).parent

# Initialize OpenAI client for Databricks
client = get_openai_client()


def analyze_all_frames(ground_truth: str, summary_placeholder=None, image_placeholder=None) -> Dict:
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

        # Update image in real-time if placeholder provided
        if image_placeholder:
            image_placeholder.image(frame_path, caption=f"Analyzing Frame {frame_number}/{len(frame_files)}", width=400)

        frame_result = analyze_frame(
            client,
            frame_path,
            frame_number,
            ground_truth,
            current_summary,
            error_handler=st.error,
        )

        if frame_result:
            frame_dict = frame_result.model_dump()
            frame_dict["frame_number"] = frame_number
            frame_dict["frame_path"] = frame_path
            results.append(frame_dict)
            current_summary = frame_dict["updated_summary"]

            # Update summary in real-time if placeholder provided
            if summary_placeholder:
                summary_placeholder.info(
                    f"**Current Summary (after frame {frame_number}):**\n\n{current_summary}"
                )

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
        page_title="Video Quality Analyzer", page_icon="🎬", layout="wide"
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
            format_func=lambda x: x.title(),
        )

        # FPS selection
        fps_options = {
            "0.5 fps (1 frame every 2 seconds) - Very sparse": 0.5,
            "1 fps (1 frame per second) - Default": 1.0,
            "2 fps (2 frames per second) - Dense": 2.0,
            "3 fps (3 frames per second) - Very dense": 3.0,
        }

        fps_label = st.selectbox(
            "Frame Extraction Rate", options=list(fps_options.keys())
        )
        fps = fps_options[fps_label]

        st.markdown("---")

        # Display selected configuration
        st.subheader("📋 Selected Configuration")
        st.write(f"**Video:** {video_name.title()}")
        st.write(f"**FPS:** {fps}")

        st.markdown("---")

        # What is this? expander
        with st.expander("❓ What is this?"):
            st.write("""
            This application analyzes AI-generated videos by extracting frames and evaluating them against their original generation prompts.

            **How it works:**
            1. Extract frames from a video at your chosen rate (FPS)
            2. AI analyzes each frame for correctness, coherence, and faithfulness to the prompt
            3. Get detailed scores and descriptions for quality assessment

            **Key Feature - Temporal Understanding:**
            The summary builds over time by using the previous summary as context for analyzing the next frame. This allows the AI to track changes and understand the video's progression through time, not just individual static frames.

            **About the videos:**
            I generated the videos using **Sora** (OpenAI's video generation model), I used the same prompts as the samples on their website.
            """)

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
            extract_button = st.button(
                "🎞️ Extract Frames",
                width='stretch',
                type="primary",
                key="extract_btn",
            )

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
                        "frame_count": frame_count,
                    }
                    st.rerun()  # Rerun to update frames_exist state
                else:
                    st.session_state.extraction_success = {
                        "success": False,
                        "error_msg": error_msg,
                    }

            # Display success/error messages from session state
            if st.session_state.extraction_success:
                if st.session_state.extraction_success["success"]:
                    st.success(
                        f"✅ Successfully extracted {st.session_state.extraction_success['frame_count']} frames!"
                    )
                    st.info(
                        "👉 Switch to 'Analyze Frames' tab to start the AI analysis"
                    )
                else:
                    st.error(
                        f"❌ Frame extraction failed: {st.session_state.extraction_success['error_msg']}"
                    )

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
                            st.image(
                                frame_files[idx],
                                caption=f"Frame {idx + 1}",
                                width="stretch",
                            )

    # TAB 2: Frame Analysis
    with tab2:
        # Prompt at top (no video preview)
        st.subheader("📝 Video Prompt (Ground Truth)")
        st.info(video_prompts[video_name]["prompt"])

        if not frames_exist:
            st.warning("⚠️ No frames found. Please extract frames first.")
            st.info(
                "👈 Go to the 'Extract Frames' tab to extract frames before analyzing"
            )
        else:
            # Analyze button right below prompt
            analyze_button = st.button(
                "🔍 Analyze Frames",
                width="stretch",
                type="primary",
                key="analyze_btn",
            )

            # Run analysis when button is clicked
            if analyze_button:
                st.info(f"Found {len(frame_files)} frames to analyze")

                # Create placeholders for real-time updates
                image_placeholder = st.empty()
                summary_placeholder = st.empty()

                with st.spinner("Analyzing frames with AI..."):
                    analysis_results = analyze_all_frames(
                        video_prompts[video_name]["prompt"],
                        summary_placeholder=summary_placeholder,
                        image_placeholder=image_placeholder,
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
                        "Correctness", f"{scores['average_correctness']}/10", delta=None
                    )

                with metric_cols[1]:
                    st.metric(
                        "Coherence", f"{scores['average_coherence']}/10", delta=None
                    )

                with metric_cols[2]:
                    st.metric(
                        "Faithfulness",
                        f"{scores['average_faithfulness']}/10",
                        delta=None,
                    )

                with metric_cols[3]:
                    st.metric(
                        "Overall Score", f"{scores['overall_score']}/10", delta=None
                    )

                # Display final summary
                st.subheader("📝 Final Summary")
                st.write(analysis_results["final_summary"])

                # Display frame-by-frame results
                st.subheader("🎞️ Frame-by-Frame Analysis")

                for result in analysis_results["frame_results"]:
                    with st.expander(
                        f"Frame {result['frame_number']} - Scores: C:{result['correctness_score']} | Co:{result['coherence_score']} | F:{result['faithfulness_score']}"
                    ):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.image(result["frame_path"], width="stretch")

                        with col2:
                            st.write("**Description:**")
                            st.write(result["frame_description"])

                            st.write("**Scores:**")
                            score_col1, score_col2, score_col3 = st.columns(3)
                            with score_col1:
                                st.metric(
                                    "Correctness", f"{result['correctness_score']}/10"
                                )
                            with score_col2:
                                st.metric(
                                    "Coherence", f"{result['coherence_score']}/10"
                                )
                            with score_col3:
                                st.metric(
                                    "Faithfulness", f"{result['faithfulness_score']}/10"
                                )

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
                    key="download_results_btn",
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
                            st.image(
                                frame_files[idx],
                                caption=f"Frame {idx + 1}",
                                width="stretch",
                            )


if __name__ == "__main__":
    main()
