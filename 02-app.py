import glob
import json
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text

from helpers import analyze_frame, get_openai_client, load_video_prompts

current_dir = Path(__file__).parent

# Make sure this is one of the keys in video_prompts.json
video_name = "hunt"

# Load prompts and get ground truth
prompts = load_video_prompts()
GROUND_TRUTH = prompts[video_name]["prompt"]

# Initialize OpenAI client
client = get_openai_client()


def analyze_all_frames() -> Dict:
    """Analyze all frames in the data directory"""
    console = Console()

    # Get all frame files sorted by name
    frame_files = sorted(glob.glob("data/frame_*.png"))

    if not frame_files:
        console.print("[red]No frame files found in data/ directory[/red]")
        return {}

    # Display header
    console.print(
        Panel.fit(
            "[bold blue]Video Frame Analysis[/bold blue]\n"
            + f"[cyan]Found {len(frame_files)} frames to analyze[/cyan]\n"
            + f"[yellow]Ground Truth:[/yellow] {GROUND_TRUTH}",
            border_style="blue",
        )
    )

    results = []
    current_summary = ""

    # Use rich progress tracking
    for frame_path in track(frame_files, description="[green]Analyzing frames..."):
        frame_number = len(results) + 1

        frame_result = analyze_frame(
            client, frame_path, frame_number, GROUND_TRUTH, current_summary
        ).model_dump()
        frame_result["frame_number"] = frame_number
        frame_result["frame_path"] = frame_path

        results.append(frame_result)
        current_summary = frame_result["updated_summary"]

        # Create a table for frame results
        frame_table = Table(show_header=True, header_style="bold magenta")
        frame_table.add_column("Metric", style="cyan")
        frame_table.add_column("Score", justify="center")

        # Color code scores
        def get_score_color(score):
            if score >= 8:
                return "[green]"
            elif score >= 6:
                return "[yellow]"
            else:
                return "[red]"

        correctness = frame_result["correctness_score"]
        coherence = frame_result["coherence_score"]
        faithfulness = frame_result["faithfulness_score"]

        frame_table.add_row(
            "Correctness", f"{get_score_color(correctness)}{correctness}/10[/]"
        )
        frame_table.add_row(
            "Coherence", f"{get_score_color(coherence)}{coherence}/10[/]"
        )
        frame_table.add_row(
            "Faithfulness", f"{get_score_color(faithfulness)}{faithfulness}/10[/]"
        )

        # Display frame analysis
        console.print(f"\n[bold]Frame {frame_number}:[/bold] [dim]{frame_path}[/dim]")
        console.print(frame_table)

        # Display description in a panel
        description_text = Text(frame_result["frame_description"])
        description_text.wrap(console, console.width - 4)
        console.print(
            Panel(
                description_text,
                title="[bold]Frame Description[/bold]",
                border_style="dim",
            )
        )

        console.print("[dim]" + "─" * console.width + "[/dim]")

    # Calculate aggregate scores
    total_frames = len(results)
    avg_correctness = sum(r["correctness_score"] for r in results) / total_frames
    avg_coherence = sum(r["coherence_score"] for r in results) / total_frames
    avg_faithfulness = sum(r["faithfulness_score"] for r in results) / total_frames
    overall_score = (avg_correctness + avg_coherence + avg_faithfulness) / 3

    final_summary = current_summary if current_summary else "No summary generated"

    analysis_summary = {
        "ground_truth": GROUND_TRUTH,
        "total_frames": total_frames,
        "aggregate_scores": {
            "average_correctness": round(avg_correctness, 2),
            "average_coherence": round(avg_coherence, 2),
            "average_faithfulness": round(avg_faithfulness, 2),
            "overall_score": round(overall_score, 2),
        },
        "final_summary": final_summary,
        "frame_results": results,
    }

    return analysis_summary


def print_final_report(analysis: Dict):
    """Print a comprehensive final report using rich formatting"""
    console = Console()

    # Header
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold white]🎬 FINAL ANALYSIS REPORT 🎬[/bold white]",
            style="bold blue",
            padding=(1, 2),
        )
    )

    # Ground truth and basic info
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column("Label", style="bold cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Ground Truth:", analysis["ground_truth"])
    info_table.add_row("Total Frames:", str(analysis["total_frames"]))

    console.print(
        Panel(info_table, title="[bold]Analysis Details[/bold]", border_style="cyan")
    )

    # Aggregate scores table
    scores = analysis["aggregate_scores"]
    score_table = Table(show_header=True, header_style="bold magenta")
    score_table.add_column("Metric", style="cyan")
    score_table.add_column("Average Score", justify="center")
    score_table.add_column("Rating", justify="center")

    def get_rating(score):
        if score >= 9:
            return "[green]Excellent ⭐⭐⭐[/green]"
        elif score >= 8:
            return "[bright_green]Very Good ⭐⭐[/bright_green]"
        elif score >= 7:
            return "[yellow]Good ⭐[/yellow]"
        elif score >= 6:
            return "[orange3]Fair[/orange3]"
        else:
            return "[red]Poor[/red]"

    def get_score_style(score):
        if score >= 8:
            return "[bold green]"
        elif score >= 6:
            return "[bold yellow]"
        else:
            return "[bold red]"

    correctness = scores["average_correctness"]
    coherence = scores["average_coherence"]
    faithfulness = scores["average_faithfulness"]
    overall = scores["overall_score"]

    score_table.add_row(
        "Correctness",
        f"{get_score_style(correctness)}{correctness}/10[/]",
        get_rating(correctness),
    )
    score_table.add_row(
        "Coherence",
        f"{get_score_style(coherence)}{coherence}/10[/]",
        get_rating(coherence),
    )
    score_table.add_row(
        "Faithfulness",
        f"{get_score_style(faithfulness)}{faithfulness}/10[/]",
        get_rating(faithfulness),
    )
    score_table.add_row(
        "[bold]Overall Score[/bold]",
        f"{get_score_style(overall)}{overall}/10[/]",
        f"[bold]{get_rating(overall)}[/bold]",
    )

    console.print(
        Panel(
            score_table,
            title="[bold]📊 Aggregate Scores[/bold]",
            border_style="magenta",
        )
    )

    # Final summary
    summary_text = Text(analysis["final_summary"])
    summary_text.wrap(console, console.width - 4)
    console.print(
        Panel(summary_text, title="[bold]📝 Final Summary[/bold]", border_style="green")
    )


if __name__ == "__main__":
    console = Console()

    # Run frame-by-frame analysis
    analysis_results = analyze_all_frames()

    if analysis_results:
        # Print final report
        print_final_report(analysis_results)

        # Save results to JSON file
        with open("frame_analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2)

        console.print(
            Panel.fit(
                "[bold green]✅ Analysis Complete![/bold green]\n"
                + "[cyan]Detailed results saved to:[/cyan] [yellow]frame_analysis_results.json[/yellow]",
                border_style="green",
            )
        )
    else:
        console.print("[red]No analysis results to display[/red]")
