import os
import argparse
from utils import file_upload, get_model, get_audio_duration, format_time, write_output, transcribe

def get_args():
    parser = argparse.AugmentParser()
    parser.add_argument("HF_KEY", help="Your HuggingFace API KEY")
    parser.add_argument("--model_size", help="Adjust based on your computing power.", default="large-v3")
    parser.add_argument("--device", help="Use CPU or GPU. Can either be 'cuda' or 'cpu'", default="cuda")
    parser.add_argument("--compute_type", help="Float bytes you use", default="float16")
    parser.add_argument("--lang", help="Target language", default="id")
    return parser.parse_arg()

def main(args):
    os.environ["HF_API_TOKEN"] = args.HF_KEY
    audio = file_upload()
    audio_duration = get_audio_duration(audio)

    print("Using Model Size:", args.model_size)
    print("Running On:", args.device)
    print("Using Compute Type:", args.compute_type)

    model = get_model(args)
    
    srt_lines = transcribe(audio_duration, args)
    write_output(audio, srt_lines)

if __name__ == "__main__":
    args = get_args()
    main(args)
