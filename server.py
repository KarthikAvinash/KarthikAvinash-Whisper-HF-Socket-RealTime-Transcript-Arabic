import asyncio
import websockets
import os
import datetime
import librosa
import numpy as np
from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
print("Model loaded successfully")

# Ensure the chunks directory exists
if not os.path.exists('chunks'):
    os.makedirs('chunks')

async def save_audio(websocket, path):
    async for message in websocket:
        print('Received audio data from client.')
        # Save the received audio data
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        storing_file_name = "temp.wav"
        file_path = os.path.join('chunks', storing_file_name)
        
        with open(file_path, 'wb') as f:
            f.write(message)
            print(f'Saved audio file to {file_path}')
        
        audio_array, sampling_rate = librosa.load(file_path, sr=16000)
        # Ensure the audio is mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        print("Processing... Please wait.")
        # Transcribe the audio
        result = pipe(audio_array)

        print(f"Transcription: {result['text']}")

        # Send transcription to the client
        await websocket.send(result['text'])

        # Send signal to send next chunk
        await websocket.send("You can send next chunk")

start_server = websockets.serve(save_audio, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
