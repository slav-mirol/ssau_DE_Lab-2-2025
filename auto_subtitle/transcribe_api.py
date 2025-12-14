# transcribe_api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper Transcriber (SRT Output)")

# --- CORS: разрешаем запросы с локального n8n ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5678",
        "http://127.0.0.1:5678",
        "http://192.168.99.102:5678"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель один раз при старте
print("Loading Whisper model...")
model = WhisperModel("small", device="cpu", compute_type="int8_float32")
print("Model loaded.")


def format_timestamp(seconds: float) -> str:
    """Преобразует секунды в формат HH:MM:SS,mmm для SRT"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1_000
    ms = milliseconds % 1_000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


@app.post("/transcribe_srt")
async def transcribe_srt(file: UploadFile = File(...)):
    # Проверка расширения
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        return JSONResponse({"error": "Only audio files allowed"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, _ = model.transcribe(tmp_path,
                                       language="en",
                                       beam_size=5,
                                       vad_filter=True,
                                       temperature=0.0
                                       )

        def srt_generator():
            for i, segment in enumerate(segments, start=1):
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                text = segment.text.strip().replace('\n', ' ')
                yield f"{i}\n{start} --> {end}\n{text}\n\n".encode("utf-8")

        return StreamingResponse(
            srt_generator(),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{os.path.splitext(file.filename)[0]}.srt"'}
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)