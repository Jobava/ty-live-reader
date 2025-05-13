# Romanian Live Chat TTS

Scripts for Romanian TTS from YouTube live chat. Tested only on macOS (Apple Silicon M2+).

## Requirements
- macOS (Apple Silicon M2+)
- Python 3.11
- `requirements.txt` dependencies
- FFmpeg: `brew install ffmpeg`

## Setup
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## Usage

### live_tts_ro_chat2.py

A LLM-based Youtube Lives comments (text) to voice

```sh
python live_tts_ro_chat2.py --video-id <YOUTUBE_VIDEO_ID>
python live_tts_ro_chat2.py --self-test
python live_tts_ro_chat2.py --self-test-all
```
- Options: `--model`, `--speed`, `--workers`, `--fs2-path`, `--log-level`

### live_tts_pool.py

A basic TTS for Youtube Lives comments (text) to voice

```sh
python live_tts_pool.py --video-id <YOUTUBE_VIDEO_ID>
```
- Options: `--rate`, `--workers`

## Notes
- Only tested on macOS M2+.
- Needs `afplay` and `say` (macOS default).
- Log files: `chat.log`, `chat2.log`.
- Temp audio: `wav_queue/`, `wav_tmp/`.
- Emoji translation: `emojis.txt`.

## Troubleshooting
- NLTK error: `python -m nltk.downloader punkt`
- FFmpeg error: `brew install ffmpeg`
- PyTorch/torchaudio: match versions in `requirements.txt`

## License

See LICENSE file.
