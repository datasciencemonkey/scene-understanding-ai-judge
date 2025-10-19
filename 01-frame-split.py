#!/usr/bin/env python3

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

current_dir = Path(__file__).parent

def clear_data_folder(output_dir):
    """
    Clear all contents of the data folder

    Args:
        output_dir: Directory to clear
    """
    if os.path.exists(output_dir):
        print(f"Clearing existing files in {output_dir}/...")
        shutil.rmtree(output_dir)
        print(f"✓ Cleared {output_dir}/")

    # Recreate empty directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Created clean {output_dir}/ directory")


def get_fps_from_user():
    """
    Prompt user for desired FPS sampling rate

    Returns:
        float: FPS value selected by user
    """
    print("\n=== Frame Extraction Settings ===")
    print("Choose sampling rate (frames per second):")
    print("  1. 0.5 fps (1 frame every 2 seconds) - Very sparse")
    print("  2. 1 fps (1 frame per second) - Default")
    print("  3. 2 fps (2 frames per second) - Dense")
    print("  4. Custom fps value")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                return 0.5
            elif choice == "2":
                return 1.0
            elif choice == "3":
                return 2.0
            elif choice == "4":
                while True:
                    try:
                        custom_fps = float(
                            input("Enter custom fps value (e.g., 0.25, 1.5, 3): ")
                        )
                        if custom_fps > 0:
                            return custom_fps
                        else:
                            print("FPS must be greater than 0. Please try again.")
                    except ValueError:
                        print("Invalid number. Please enter a valid fps value.")
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}. Please try again.")


def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from video at specified fps using FFmpeg

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
    """
    # Output directory should already be created and cleared
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        f"{output_dir}/frame_%03d.png",
    ]

    try:
        print("\n=== Starting Frame Extraction ===")
        print(f"Video file: {video_path}")
        print(f"Output directory: {output_dir}")
        print(f"Sampling rate: {fps} fps")
        print(f"FFmpeg command: {' '.join(cmd)}")
        print("\nExtracting frames...")

        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Frame extraction completed successfully!")

            # Count extracted frames
            frame_files = [
                f
                for f in os.listdir(output_dir)
                if f.startswith("frame_") and f.endswith(".png")
            ]
            print(f"✓ Extracted {len(frame_files)} frames")

            # Show file sizes
            total_size = 0
            for frame_file in frame_files:
                file_path = os.path.join(output_dir, frame_file)
                size = os.path.getsize(file_path)
                total_size += size

            total_size_mb = total_size / (1024 * 1024)
            print(f"✓ Total size: {total_size_mb:.1f} MB")

        else:
            print(f"Error extracting frames: {result.stderr}")
            return False

    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg first.")
        print("On macOS: brew install ffmpeg")
        print("On Ubuntu: sudo apt install ffmpeg")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

    return True


if __name__ == "__main__":
    # Default values


    video_name = "hunt"

    with open(current_dir / "videos/video_prompts.json", "r") as f:
        prompts = json.load(f)
    video_file = prompts[video_name]["save_path"]
    video_file = str(current_dir / "videos" / video_file)

    output_directory = current_dir / "data"

    print("🎬 Video Frame Extractor")
    print("=" * 40)

    # Check if video file exists
    if not os.path.exists(video_file):
        print(f"❌ Error: Video file '{video_file}' not found")
        sys.exit(1)

    # Show video file info
    video_size = os.path.getsize(video_file) / (1024 * 1024)
    print(f"📹 Video file: {video_file} ({video_size:.1f} MB)")

    # Clear data folder first
    print("\n=== Preparing Output Directory ===")
    clear_data_folder(output_directory)

    # Get FPS from user
    fps = get_fps_from_user()

    # Confirm before processing
    print("\n=== Confirmation ===")
    print(f"Video: {video_file}")
    print(f"Output: {output_directory}/")
    print(f"Image rate: {fps} fps")

    try:
        confirm = input("\nProceed with frame extraction? (y/N): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("Operation cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)

    # Extract frames
    success = extract_frames(video_file, output_directory, fps)

    if success:
        print("\n🎉 Success! Frames extracted to {}/".format(output_directory))
        print("You can now run the analysis with: uv run app.py")
    else:
        print("\n❌ Frame extraction failed.")
        sys.exit(1)
