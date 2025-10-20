#!/usr/bin/env python3
# ABOUTME: Shared helper functions and classes for video frame extraction and analysis
# ABOUTME: Contains common utilities used across streamlit_app.py, 01-frame-split.py, and 02-app.py

import base64
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin

import mlflow
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

if "https" not in os.environ.get("DATABRICKS_HOST", ""):
    os.environ["DATABRICKS_HOST"] = "https://" + os.environ.get("DATABRICKS_HOST", "")

# Enable automatic tracing for OpenAI - that's it!
mlflow.openai.autolog()

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID"))

# Get current directory
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


def get_openai_client() -> OpenAI:
    """
    Initialize and return OpenAI client for Databricks

    Returns:
        OpenAI: Configured OpenAI client instance
    """
    databricks_host: Optional[str] = os.getenv("DATABRICKS_HOST")
    databricks_token: Optional[str] = os.environ.get("DATABRICKS_TOKEN")
    base_url: str = urljoin(databricks_host, "serving-endpoints")

    return OpenAI(
        api_key=databricks_token,
        base_url=base_url,
    )


def load_video_prompts() -> dict:
    """Load video prompts from JSON file

    Returns:
        dict: Dictionary containing video prompts and metadata
    """
    with open(current_dir / "videos/video_prompts.json", "r") as f:
        return json.load(f)


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_video(video_path: str) -> str:
    """Encode video file to base64 string

    Args:
        video_path: Path to the video file

    Returns:
        str: Base64 encoded string of the video
    """
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def clear_data_folder(output_dir: str) -> None:
    """Clear all contents of the data folder

    Args:
        output_dir: Path to the output directory to clear
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def analyze_frame(
    client: OpenAI,
    image_path: str,
    frame_number: int,
    ground_truth: str,
    previous_summary: str = "",
    error_handler: Optional[Callable[[str], None]] = None,
) -> Optional[FrameAnalysis]:
    """
    Analyze a single frame and return scoring and description

    Args:
        client: OpenAI client instance
        image_path: Path to the image file
        frame_number: Frame number being analyzed
        ground_truth: Ground truth prompt to compare against
        previous_summary: Summary from previous frames
        error_handler: Optional function to call on error (e.g., st.error or print)

    Returns:
        FrameAnalysis: Parsed analysis result or None on error
    """
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
    5. UPDATED SUMMARY: Based on this frame and previous frames, provide an updated summary of the overall video content. That should include new elements *and* what has been observed previously.
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
        error_msg = f"Error analyzing frame {frame_number}: {e}"
        if error_handler:
            error_handler(error_msg)
        else:
            print(error_msg)
        return None


def extract_frames(
    video_path: str, output_dir: str, fps: float = 1
) -> tuple[bool, int, Optional[str]]:
    """
    Extract frames from video at specified fps using MoviePy

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)

    Returns:
        tuple: (success: bool, frame_count: int, error_message: str)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Load video using MoviePy
        video = VideoFileClip(video_path)

        # Calculate the interval between frames based on fps
        interval = 1.0 / fps

        # Extract frames at specified intervals
        frame_count = 0
        current_time = 0

        while current_time < video.duration:
            # Save frame at current time with zero-padded numbering
            frame_filename = f"{output_dir}/frame_{frame_count + 1:03d}.png"
            video.save_frame(frame_filename, t=current_time)
            frame_count += 1
            current_time += interval

        # Clean up video resource
        video.close()

        return True, frame_count, None

    except FileNotFoundError:
        return (
            False,
            0,
            f"Video file not found: {video_path}",
        )
    except Exception as e:
        return False, 0, f"Error extracting frames with MoviePy: {str(e)}"
