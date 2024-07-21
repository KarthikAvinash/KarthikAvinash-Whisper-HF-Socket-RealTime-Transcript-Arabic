
import asyncio
import websockets
from transformers import pipeline
import librosa
import numpy as np
import io

# Initialize the transcription pipeline
pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic")

async def save_audio(websocket, path):
    try:
        # Receive the audio content
        audio_content = await websocket.recv()
        
        # Save the audio content as temp.wav
        with open('temp.wav', 'wb') as f:
            f.write(audio_content)
        
        # Load the audio file
        audio_path = 'temp.wav'
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Ensure the audio is mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Transcribe the audio
        result = pipe(audio_array)
        transcription = result['text']

        # Send the transcription result back to the client
        await websocket.send(transcription)
        print(f"Transcription sent to client: {transcription}")

    except Exception as e:
        print(f'Error: {e}')
        await websocket.send('Error occurred while processing the file.')

async def main():
    # Define the server address and port
    address = "localhost"
    port = 8080

    # Start the WebSocket server
    server = await websockets.serve(save_audio, address, port)

    print(f"Server listening on ws://{address}:{port}")

    # Run the server indefinitely
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

