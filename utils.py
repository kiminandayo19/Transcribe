import sys
import subprocess
import json
from typing import NamedTuple, Generator, List, Any
from faster_whisper import WhisperModel

class TranscribeConfig(NamedTuple):
    model_size: str
    device: str
    compute_type: str
    lang: str

def format_time(seconds: float) -> str:
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

def create_srt_block(index: int, start: float, end: float, text: str) -> str:
    """Formats a single SRT block."""
    return f"{index}\n{format_time(start)} --> {format_time(end)}\n{text.strip()}\n"

def format_srt(segments) -> Generator[str, None, None]:
    """Yields formatted SRT blocks from segments."""
    for i, seg in enumerate(segments, start=1):
        yield create_srt_block(i, seg.start, seg.end, seg.text)

def get_model_instance(config: TranscribeConfig) -> WhisperModel:
    """Initializes and returns the WhisperModel."""
    return WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type
    )

def get_audio_duration(audio_path: str) -> float:
    """Gets audio duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        audio_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        sys.stderr.write(f"Error getting duration: {e}\n")
        return 0.0

def write_srt_file(audio_path: str, srt_content: List[str]) -> str:
    """Writes the list of SRT blocks to a file."""
    outpath = f"{audio_path}.srt"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))
    return outpath
