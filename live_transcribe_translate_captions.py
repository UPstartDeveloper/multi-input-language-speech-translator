#!/usr/bin/env python3
"""
nejaf_live_translator.py

Mic → Whisper → Argos Translate → English captions for OBS
Handles silence filtering and supports Arabic/Farsi/Urdu/Dari.
"""

import os
import queue
import threading
import time
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from langdetect import detect, DetectorFactory
import argostranslate.package, argostranslate.translate

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5.0       # seconds per chunk
MODEL_NAME = "medium"       # can be "tiny", "base", "small", "medium"
DEVICE = "cpu"
CAPTIONS_FILE = "captions.txt"
SILENCE_THRESHOLD = 0.005  # adjust if your mic is noisy
FORCE_LANG = None          # e.g., "ar" to force Arabic, or None for auto-detect
# TODO[1]: confirm if Dari is recognized (Afghan language)
# TODO[2]: confirm if urdu script is recognized
# TODO[3]: work on improving quality
SUPPORTED_LANGS = ["ar", "fa", "ur", "ps", "hi", "en"]
# ----------------------------------------

DetectorFactory.seed = 0  # make langdetect deterministic

audio_q = queue.Queue(maxsize=50)
stop_event = threading.Event()

# Initialize Whisper
print(f"[{datetime.now()}] Loading Whisper model '{MODEL_NAME}'...")
model = WhisperModel(MODEL_NAME, device=DEVICE)
print(f"[{datetime.now()}] Whisper ready.")

# Helper: translate text to English
def translate_to_english(text, from_code):
    if not text.strip():
        return ""
    if from_code == "en":
        return text
    try:
        translated = argostranslate.translate.translate(text, from_code, "en")
        return translated
    except Exception as e:
        print(f"[{datetime.now()}] Translation {from_code}->en failed: {e}")
        return text

# Write latest caption to file for OBS
def write_caption(text):
    clean = text.replace("\n", " ").strip()
    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        f.write(clean + "\n")

# Audio callback
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    try:
        audio_q.put_nowait(indata.copy())
    except queue.Full:
        print("Audio buffer full, dropping chunk.")

# Processor thread
def processor_worker():
    buf = []
    accumulated = 0
    target = int(CHUNK_DURATION * SAMPLE_RATE)
    print(f"[{datetime.now()}] Worker started (chunk {CHUNK_DURATION}s).")

    while not stop_event.is_set():
        try:
            frames = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue
        buf.append(frames)
        accumulated += frames.shape[0]

        if accumulated >= target:
            chunk = np.concatenate(buf, axis=0)
            if chunk.ndim > 1:
                chunk = chunk[:, 0]

            # ---- Silence filter ----
            volume = np.abs(chunk).mean()
            if volume < SILENCE_THRESHOLD:
                buf, accumulated = [], 0
                continue

            # Write temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, chunk, SAMPLE_RATE)

            try:
                segments, info = model.transcribe(tmp_path, beam_size=5)
                text = " ".join([s.text.strip() for s in segments if s.text.strip()])
                if not text:
                    buf, accumulated = [], 0
                    os.unlink(tmp_path)
                    continue

                # Language detection
                if FORCE_LANG:
                    lang = FORCE_LANG
                else:
                    try:
                        lang = detect(text)
                        if lang not in SUPPORTED_LANGS:
                            lang = "en"
                    except Exception:
                        lang = "en"

                translated = translate_to_english(text, lang)
                write_caption(translated)
                print(f"[{datetime.now()}] [{lang}] {text} -> {translated}")

            except Exception as e:
                print(f"Error: {e}")
            finally:
                buf, accumulated = [], 0
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    print("Processor stopped.")

def main():
    write_caption("")  # clear file
    worker = threading.Thread(target=processor_worker, daemon=True)
    worker.start()

    try:
        with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, callback=audio_callback):
            print(f"[{datetime.now()}] Listening... Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_event.set()
        worker.join()
        print("Done.")

if __name__ == "__main__":
    main()
