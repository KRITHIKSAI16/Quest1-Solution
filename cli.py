"""thin CLI wrapper over core/. no pipeline logic here, just wiring user
input to locate() and printing the result.
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

# has to happen before Rich gets imported. its Windows console writer talks
# to the Win32 API directly, so fixing sys.stdout later doesn't help.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import questionary
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

sys.path.insert(0, str(Path(__file__).parent))

from core.acquire import (
    AcquisitionError,
    InvalidSourceError,
    MissingDependencyError,
    NetworkUnreachableError,
    acquire,
)
from core.config import DEFAULT, Config
from core.fuse import Outcome, resolve_strict
from core.report import build_result, print_report, save_frame, write_result_json
from core.report_html import generate_html_report
from core.video import VideoSource

app = typer.Typer(add_completion=False)
console = Console()

MODE_CHOICES = ("audio", "visual", "auto")


def _ask_nonempty(label: str) -> str:
    while True:
        value = Prompt.ask(label).strip()
        if value:
            return value
        console.print("[red]This can't be empty.[/red]")


_MODE_MENU_CHOICES = [
    questionary.Choice(title="Audio  [For spoken dialogue - Fastest response]", value="audio"),
    questionary.Choice(title="Visual [For burned-in text on screen]", value="visual"),
    questionary.Choice(title="Auto   [If unsure or need both - Dual-channel]", value="auto"),
]


def _ask_mode() -> str:
    """arrow-key menu, can't return anything outside MODE_CHOICES so no
    retry loop needed here.
    """
    console.print()
    choice = questionary.select(
        "Select search mode:",
        choices=_MODE_MENU_CHOICES,
        default=_MODE_MENU_CHOICES[2],
    ).ask()
    if choice is None:
        raise typer.Exit(1)  # user hit ctrl+c / esc
    return choice


def _run_interactive_prompt(url: str | None, text: str | None, mode: str | None) -> tuple[str, str, str]:
    """only runs when --url or --text is missing, and only asks for the
    fields that weren't already given.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Frame & Dialogue Extractor[/bold cyan]\n"
            "[dim]Find the exact frame where a line of dialogue first appears[/dim]",
            border_style="cyan",
        )
    )
    if url is None:
        url = _ask_nonempty("[bold]Video URL[/bold] (or local file path)")
    if text is None:
        text = _ask_nonempty("[bold]Target dialogue text[/bold] to search for")
    if mode is None:
        mode = _ask_mode()
    console.print()
    return url, text, mode


@app.command()
def main(
    url: str = typer.Option(None, "--url", help="Video URL (yt-dlp-supported site, direct media URL) or local file path"),
    text: str = typer.Option(None, "--text", help="Target dialogue text to search for"),
    out: str = typer.Option("output", "--out", help="Output directory for result.json and the saved frame"),
    mode: str = typer.Option(None, "--mode", help="auto | visual | audio (prompted interactively if omitted along with --url/--text)"),
    roi: float = typer.Option(DEFAULT.roi_bottom_fraction, "--roi", help="ROI band as a fraction of frame height, from the bottom"),
    full_frame: bool = typer.Option(False, "--full-frame", help="Skip the ROI fast path; sweep the whole frame"),
    sample_fps: float = typer.Option(DEFAULT.sample_fps, "--sample-fps", help="Visual channel sampling rate"),
    strict: bool = typer.Option(False, "--strict", help="Force a single answer even on ambiguous/uncertain results"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip generating report.html (result.json is still written)"),
    no_face_check: bool = typer.Option(False, "--no-face-check", help="Skip the on-screen face check for audio matches"),
    face_check_window: float = typer.Option(DEFAULT.face_check_window_s, "--face-check-window", help="Seconds around the audio onset to look for a face"),
    no_mouth_motion_check: bool = typer.Option(False, "--no-mouth-motion-check", help="Skip the mouth-motion signal (report-only, never affects the outcome)"),
):
    interactive = url is None or text is None  # only prompt if one of these is actually missing
    if url is None or text is None:
        url, text, mode = _run_interactive_prompt(url, text, mode)
    if mode is None:
        mode = "auto"

    if mode not in MODE_CHOICES:
        console.print(f"[red]Invalid --mode '{mode}'. Must be one of: {', '.join(MODE_CHOICES)}.[/red]")
        raise typer.Exit(1)

    cfg = Config(
        sample_fps=sample_fps,
        roi_bottom_fraction=roi,
        force_full_frame=full_frame,
        face_check_enabled=not no_face_check,
        face_check_window_s=face_check_window,
        mouth_motion_check_enabled=not no_mouth_motion_check,
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        t = progress.add_task("Acquiring video...", total=None)
        try:
            video_path = acquire(url, cache_dir=cfg.cache_dir)
        except InvalidSourceError as e:
            progress.stop()
            console.print(f"[red]This doesn't look like a usable video source:[/red] {e}")
            console.print(
                "[yellow]Expected:[/yellow] a link to a video (any site yt-dlp "
                "supports — YouTube, Vimeo, ok.ru, ~1800 others), a direct "
                "media file URL, or a path to a local video file."
            )
            raise typer.Exit(1)
        except MissingDependencyError as e:
            progress.stop()
            console.print(f"[red]Missing dependency:[/red] {e}")
            raise typer.Exit(1)
        except NetworkUnreachableError as e:
            # acquire() already retried before giving up on this one
            progress.stop()
            console.print(f"[red]Could not reach the video source after retrying:[/red] {e}")
            console.print(
                "[yellow]This network may be blocking or unable to reach the "
                "source.[/yellow] Try a different network, or pass a local "
                "file path to --url instead of the source URL."
            )
            raise typer.Exit(1)
        except AcquisitionError as e:
            progress.stop()
            console.print(f"[red]Could not obtain a usable video file:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            # whatever this is, still don't dump a raw traceback on the user
            progress.stop()
            console.print(f"[red]Unexpected error while acquiring the video:[/red] {e}")
            raise typer.Exit(1)

        progress.update(t, description=f"Running {mode} channel(s)...")
        from core.locate import locate
        fusion, primary, diagnostics = locate(video_path, text, cfg, mode=mode)

        if strict and fusion.outcome in (Outcome.AMBIGUOUS,) and primary is None:
            primary = resolve_strict(fusion)

        result = build_result(fusion, primary, target=text, diagnostics=diagnostics)

        if primary is not None:
            progress.update(t, description="Saving frame...")
            with VideoSource(video_path) as video:
                frame = video.frame_at_index(primary.frame_index) if primary.frame_index is not None \
                    else video.frame_at_index(round(primary.timestamp_s * video.meta.fps))
                result.image_path = save_frame(frame.image, out, primary.frame_index, primary.timestamp_s)

        write_result_json(result, out)
        progress.update(t, description="Done.")

    console.print()
    print_report(result)

    if not no_report:
        report_path = generate_html_report(result, target_text=text, source=url, out_dir=out)
        if interactive:
            webbrowser.open(Path(report_path).resolve().as_uri())

    if result.timestamp is None:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
