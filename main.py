import os
import argparse
from tqdm import tqdm
from utils import (
    TranscribeConfig, 
    get_model_instance, 
    get_audio_duration, 
    write_srt_file, 
    create_srt_block
)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Your audio (in .m4a)")
    parser.add_argument("--hf_key", help="Your HuggingFace API KEY", default=None)
    parser.add_argument("--model_size", help="Adjust based on your computing power.", default="large-v3")
    parser.add_argument("--device", help="Use CPU or GPU. Can either be 'cuda' or 'cpu'", default="cuda")
    parser.add_argument("--compute_type", help="Float bytes you use", default="float16")
    parser.add_argument("--lang", help="Target language", default="id")
    return parser.parse_args()

def process_audio(args: argparse.Namespace) -> None:
    if args.hf_key:
        os.environ["HF_API_TOKEN"] = args.hf_key

    config = TranscribeConfig(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        lang=args.lang
    )

    print(f"Using Model Size: {config.model_size}")
    print(f"Running On: {config.device}")
    
    model = get_model_instance(config)
    audio_duration = get_audio_duration(args.audio_path)

    segments, _ = model.transcribe(
        args.audio_path,
        beam_size=5,
        vad_filter=True,
        language=config.lang
    )

    srt_lines = []
    
    with tqdm(total=audio_duration, unit="sec", desc="Transcribing to SRT") as pbar:
        last_time = 0.0
        for i, seg in enumerate(segments, start=1):
            block = create_srt_block(i, seg.start, seg.end, seg.text)
            srt_lines.append(block)

            delta = max(seg.end - last_time, 0)
            pbar.update(delta)
            pbar.set_postfix({
                "at": f"{seg.end:.1f}s",
                "text": (seg.text[:20] + "...") if len(seg.text) > 20 else seg.text
            })
            last_time = seg.end

    outpath = write_srt_file(args.audio_path, srt_lines)
    print(f"Srt stored: {outpath}")

if __name__ == "__main__":
    args = get_args()
    process_audio(args)
