import asyncio
import websockets
from transformers import pipeline
import numpy as np
import io
import torch
import time
import tempfile
import os
import subprocess
import logging
from pydub import AudioSegment
import librosa

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    global pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
    logger.info("Model loaded successfully")

# Initialize the transcription pipeline
load_model()

async def process_audio(websocket, path):
    global pipe
    try:
        while True:
            logger.info("Waiting for audio content...")
            audio_content = await websocket.recv()
            logger.info(f"Received audio content of length: {len(audio_content)} bytes")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
                temp_webm.write(audio_content)
                temp_webm_path = temp_webm.name
            
            # Convert WebM to WAV using pydub
            audio = AudioSegment.from_file(temp_webm_path, format="webm")
            temp_wav_path = temp_webm_path.replace(".webm", ".wav")
            audio.export(temp_wav_path, format="wav")
            
            # Load the WAV file
            audio_array, sampling_rate = librosa.load(temp_wav_path, sr=16000)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Transcribe the audio
            transcription = pipe(audio_array)["text"]

            # Send the transcription result back to the client
            await websocket.send(transcription)
            logger.info(f"Transcription sent to client: {transcription}")

            # Clean up temporary files
            os.remove(temp_webm_path)
            os.remove(temp_wav_path)

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")

async def main():
    server = await websockets.serve(process_audio, "localhost", 8080)
    logger.info("Server started on ws://localhost:8080")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
