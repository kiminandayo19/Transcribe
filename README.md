# ⚡️ Whisper-to-SRT
> **Blazing fast, GPU-accelerated audio transcription CLI tool.**

Turn your audio files into subtitle files (`.srt`) in seconds using the power of [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). 

Faster-Whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which is **up to 4x faster** than the original OpenAI implementation while using less memory.

## ✨ Features

- 🚀 **Incredible Speed**: Leverages `faster-whisper` for optimized inference.
- 🎯 **High Accuracy**: Supports all standard Whisper models (`large-v3`, `medium`, `base`, etc.).
- 🎞️ **Direct to SRT**: Automatically generates formatted `.srt` files ready for video players or YouTube.
- 📊 **Visual Feedback**: Real-time progress bar tracking transcription status.
- 🌍 **Multi-language Support**: Auto-detection or force specific languages (e.g., English, Indonesian, Japanese).
- 🔧 **Hardware Flexible**: Runs on NVIDIA GPUs (CUDA) or CPU.

## 🛠️ Installation

### 1. Prerequisites
You need **FFmpeg** installed on your system to process audio files.

**Quick Install (Script included):**
```bash
chmod +x script.sh
./script.sh
```

### 2. Install Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

> **Note**: For GPU acceleration, ensure you have the appropriate NVIDIA drivers and CUDA toolkit installed for `torch` and `ctranslate2`.

## 🚀 Usage

Basic usage is simple. Just pass your audio file. For standard models (like `large-v3`), you typically **do not** need an API key.

```bash
python3 main.py <path_to_audio_file>
```

### Example
```bash
python3 main.py podcast_episode.m4a --model_size medium --lang en
```

This will generate `podcast_episode.m4a.srt` in the same directory.

## ⚙️ Configuration Options

Customize reliability and performance with these flags:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `audio_path` | **Required** | Path to your input audio file (e.g., `.m4a`, `.mp3`, `.wav`). |
| `--hf_key` | `None` | Your HuggingFace API Key (only needed for gated/private models). |
| `--model_size` | `large-v3` | Balance speed vs. accuracy. Options: `tiny`, `base`, `small`, `medium`, `large-v3`. |
| `--device` | `cuda` | Compute device. Use `cuda` for GPU (recommended) or `cpu`. |
| `--compute_type` | `float16` | Precision type. `int8` is faster/smaller, `float16` is standard for GPUs. |
| `--lang` | `id` | Target language ISO code (e.g., `en`, `id`, `ja`). Use `auto` for detection. |

## 📦 Under the Hood

This tool utilizes:
- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)**: For the heavy lifting of transcription.
- **FFmpeg**: For extracting audio duration and metadata.
- **tqdm**: For the beautiful progress bar.

---
*Built for speed and simplicity.*
