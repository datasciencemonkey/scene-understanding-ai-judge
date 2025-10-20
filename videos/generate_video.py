import json
import sys
import time
from pathlib import Path

from openai import OpenAI

openai = OpenAI()


current_dir = Path(__file__).parent

# Make sure this is one of the keys in video_prompts.json
video_name = "hunt"
with open(current_dir / "video_prompts.json", "r") as f:
    prompts = json.load(f)

video_file_name = prompts[video_name]["save_path"]
PROMPT = prompts[video_name]["prompt"]

video_path = current_dir / video_file_name


def main(client: OpenAI, prompt: str, save_path: str) -> None:
    video = client.videos.create(model="sora-2", prompt=prompt, seconds="8")

    print("Video generation started:", video)

    progress = getattr(video, "progress", 0)
    bar_length = 30

    while video.status in ("in_progress", "queued"):
        # Refresh status
        video = openai.videos.retrieve(video.id)
        progress = getattr(video, "progress", 0)

        filled_length = int((progress / 100) * bar_length)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)
        status_text = "Queued" if video.status == "queued" else "Processing"

        sys.stdout.write(f"\r{status_text}: [{bar}] {progress:.1f}%")
        sys.stdout.flush()
        time.sleep(2)
    # Move to next line after progress loop
    sys.stdout.write("\n")

    if video.status == "failed":
        message = getattr(
            getattr(video, "error", None), "message", "Video generation failed"
        )
        print(message)
    else:
        print("Downloading video content...")

        content = openai.videos.download_content(video.id, variant="video")
        content.write_to_file(save_path)

        print("Wrote video to ", save_path)


if __name__ == "__main__":
    main(openai, PROMPT, save_path=video_path)
