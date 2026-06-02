# called — Real-Time Audio Transcription

Continuously captures microphone and/or system audio and generates live transcriptions using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Outputs transcripts in TXT, JSON, and SRT formats.

## Features

- **Dual audio capture** — Microphone and system audio (BlackHole on macOS) simultaneously
- **Silence detection** — Skips silent chunks before processing, saving CPU and disk I/O
- **Adaptive threshold** — Periodic recalibration of the silence threshold based on ambient noise floor
- **Checkpoint recovery** — Resumes transcription from the last checkpoint after a crash
- **Compressed archiving** — Saves audio chunks as Opus/OGG for storage efficiency
- **Real-time metrics** — Latency, queue depth, chunks processed/dropped displayed live
- **Multiple output formats** — TXT, JSON (with metadata), and SRT (subtitles)

## Quick Start

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (for audio archiving)
- On macOS for system audio capture: [BlackHole](https://existential.audio/blackhole/)

### Installation

```bash
git clone https://github.com/your-org/called.git
cd called
pip install -r requirements.txt
cp .env.example .env   # Edit as needed
```

### Usage

```bash
# Auto-detect microphone
python app.py

# List available audio devices
python app.py --list-devices

# Use specific microphone and system audio devices
python app.py --device 2 --system-audio 4

# Force language and model
python app.py --language en --model large-v3
```

### Interactive Controls

| Key | Action |
|-----|--------|
| `r` | Toggle recording (pause/resume) |
| `s` | Show status (metrics, devices, silence stats) |
| `q` | Quit (graceful shutdown with transcript save) |
| `h` | Show help |

## Configuration

All settings are configured via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SILENCE_DETECTION_ENABLED` | `true` | Skip silent audio chunks |
| `SILENCE_THRESHOLD_DB` | `-40.0` | Silence threshold in dB (RMS) |
| `SILENCE_RECALIBRATION_ENABLED` | `true` | Adapt threshold to noise floor |
| `SILENCE_RECALIBRATION_INTERVAL` | `30.0` | Recalibration interval in seconds |
| `SILENCE_RECALIBRATION_MARGIN_DB` | `10.0` | dB margin above noise floor |
| `MIC_DEVICE_INDEX` | auto | Microphone device index |
| `SYSTEM_AUDIO_DEVICE_INDEX` | — | System audio device (BlackHole) |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `CHUNK_DURATION` | `2.0` | Chunk duration in seconds |
| `WHISPER_MODEL_SIZE` | `base` | Model: tiny, base, small, medium, large-v3 |
| `WHISPER_DEVICE` | `auto` | Device: auto, cpu, cuda |
| `WHISPER_LANGUAGE` | — | Force language (e.g., en, es) |
| `CHECKPOINT_RECOVERY_ENABLED` | `true` | Resume from checkpoint after crash |
| `CHECKPOINT_INTERVAL` | `60` | Checkpoint save interval in seconds |
| `OUTPUT_DIR` | `./transcripts` | Where transcripts are saved |
| `SAVE_COMPRESSED_AUDIO` | `true` | Archive audio as compressed Opus/OGG |

## Architecture

```
app.py                    # Application orchestrator
├── audio/
│   ├── capture.py        # PortAudio capture (mic + system audio)
│   ├── silence.py        # Silence detection with adaptive recalibration
│   ├── devices.py        # Device enumeration and selection
│   └── archiver.py       # Opus/OGG compressed audio storage
├── transcription/
│   ├── whisper_service.py  # faster-whisper model wrapper
│   └── processor.py        # Async consumer: transcribes audio chunks
├── storage/
│   ├── checkpoint.py     # Session checkpoint save/load for crash recovery
│   ├── json_writer.py    # JSON transcript output
│   ├── srt_writer.py     # SRT subtitle output
│   └── txt_writer.py     # Plain text output
├── config.py             # Configuration from environment variables
├── utils/logger.py       # Logging setup
└── tests/                # pytest test suite
```

## Outputs

Transcripts are saved to the `OUTPUT_DIR` (default `./transcripts/`) on shutdown:

- **`transcript_YYYYMMDD_HHMMSS.txt`** — Plain text transcript
- **`transcript_YYYYMMDD_HHMMSS.json`** — JSON with timestamps, confidence scores, and metrics
- **`transcript_YYYYMMDD_HHMMSS.srt`** — SubRip subtitle format

Audio archives (if enabled) are saved to `./audio_archive/` as timestamped Opus/OGG files.

## Testing

```bash
pytest tests/ -v
```

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
