## AUDIO TRANSCRIBING WITH PYTHON AND FASTER WHISPER

#### Installation Setup

1.  Run `script.sh` to install ffmpeg
    <details>
        ```
            chmod +x script.sh
            ./script.sh

        ```

    </details>

2.  Install all the requirements in `requirements.txt`

#### Usage

Run the following command

```
python3 <path_to_audio_file> <your_hf_api_key>
```

###### Optional Args

| Args         | Values                                                                                         | Default  | Description                                                         |
| ------------ | ---------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------- |
| model_size   | tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3 | large-v3 | Whisper model checkpoint. Larger = better accuracy, more VRAM.      |
| device       | cuda, cpu                                                                                      | cuda     | Compute device. `cuda` requires NVIDIA GPU, `cpu` works everywhere. |
| compute_type | float16, int8_float16, int8, float32                                                           | float16  | Numeric precision. Controls speed, memory usage, and accuracy.      |
| lang         | ISO 639-1 codes (e.g. en, id, ja, zh, fr, de, es, ar, ru, pt) or auto-detect                   | id       | Target language. Omit or set to auto for detection.                 |
