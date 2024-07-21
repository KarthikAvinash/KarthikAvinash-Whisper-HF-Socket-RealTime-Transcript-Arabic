import asyncio
import websockets
from transformers import pipeline
import librosa
import numpy as np
import io
import torch
import time

def load_model():
    global pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
    print("Model loaded successfully")

# Initialize the transcription pipeline
load_model()

async def process_audio(websocket, path):
    global pipe
    try:
        print("Waiting for audio content...")
        start_time = time.time()  # Start timing here
        # Receive the audio content
        audio_content = await websocket.recv()
        print("Received audio content")

        # Use io.BytesIO to handle the audio content in memory
        audio_io = io.BytesIO(audio_content)

        # Load the audio data using librosa
        print("Loading audio data...")
        audio_array, sampling_rate = librosa.load(audio_io, sr=16000)
        print(f"Audio data loaded with sampling rate: {sampling_rate}")

        # Ensure the audio is mono
        if len(audio_array.shape) > 1:
            print("Converting audio to mono...")
            audio_array = np.mean(audio_array, axis=1)

        # Transcribe the audio
        print("Transcribing audio...")
        result = pipe(audio_array)
        transcription = result['text']

        # Send the transcription result back to the client
        await websocket.send(transcription)
        print(f"Transcription sent to client: {transcription}")

        end_time = time.time()  # End timing here
        elapsed_time = end_time - start_time
        print(f"Time taken to process and transcribe: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f'Error: {e}')
        print("Reloading model...")
        load_model()  # Reload the model
        await websocket.send('Error occurred while processing the file. Model has been reloaded.')

async def main():
    # Define the server address and port
    address = "localhost"
    port = 8080

    # Start the WebSocket server
    server = await websockets.serve(process_audio, address, port)
    print(f"Server listening on ws://{address}:{port}")

    # Run the server indefinitely
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
