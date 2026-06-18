import os
import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_YOUTUBE_DOMAINS = ('youtube.com', 'youtu.be', 'www.youtube.com', 'youtube')


def is_youtube_url(source: str) -> bool:
    return any(d in source for d in _YOUTUBE_DOMAINS)


def download_youtube_video(url: str, output_dir: str | None = None, max_height: int = 720) -> str:
    """Download a YouTube video with yt-dlp and return the local file path."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='construction_detector_')

    output_template = os.path.join(output_dir, '%(title).60s.%(ext)s')

    cmd = [
        'yt-dlp',
        '-f', (
            f'bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]'
            f'/best[height<={max_height}][ext=mp4]'
            f'/best[height<={max_height}]'
        ),
        '--merge-output-format', 'mp4',
        '-o', output_template,
        '--no-playlist',
        '--quiet',
        '--progress',
        url,
    ]

    logger.info(f"Downloading: {url}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (exit {result.returncode}). Make sure yt-dlp is installed.")

    mp4_files = sorted(Path(output_dir).glob('*.mp4'), key=os.path.getmtime, reverse=True)
    if not mp4_files:
        raise FileNotFoundError(f"Downloaded file not found in {output_dir}")

    return str(mp4_files[0])
