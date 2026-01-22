import subprocess, json
from tqdm import tqdm
from google.colab import files
from faster_whisper import WhisperModel

def file_upload():
    uploaded = files.upload()
    return list(uploaded.keys())[0] # Get only first item from uploaded

def get_model(config):
    return WhisperModel(
        config["model_size"],
        device=config["device"],
        compute_type=config["compute_type"]
    )

def get_audio_duration(audio):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        audio
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

def write_output(audio, srt_lines):
    outpath = audio + ".srt"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print("Srt stored:", outpath)

def transcribe(audio_duration, args):
    model = get_model(args)
    
    # Processing
    segments, _ = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,
        language=args.lang
    )

    progress_bar = tqdm(
        total=audio_duration,
        unit="sec",
        desc="Transcribing to SRT"
    )

    srt_lines = list()
    last_time = 0.0

    for i, seg in enumerate(segments, start=1):
        # Srt contents
        srt_lines.append(str(i))
        srt_lines.append(
            f"{format_time(seg.start)} --> {format_time(seg.end)}"
        )
        srt_lines.append(seg.text.strip())
        srt_lines.append("")

        # Progress update
        delta = max(seg.end - last_time, 0)
        progress_bar.update(delta)
        last_time = seg.end

        progress_bar.set_postfix({
            "at": f"{seg.end:.1f}s",
            "text": seg.text[:25] + "..."
        })

    progress_bar.close()

    return srt_lines
