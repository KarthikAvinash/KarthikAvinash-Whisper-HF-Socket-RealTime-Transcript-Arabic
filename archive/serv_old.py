
import asyncio
import websockets
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np

processor, model = None, None

def load_model():
    global processor, model
    processor = AutoProcessor.from_pretrained("Seyfelislem/whisper-medium-arabic")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("Seyfelislem/whisper-medium-arabic")

def transcribe_audio(audio):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        outputs = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

async def handle_websocket(websocket, path):
    try:
        async for message in websocket:
            print("Received audio data")
            audio_data = np.frombuffer(message, dtype=np.float32)
            print("Audio data shape:", audio_data.shape)
            
            # Ensure the audio data is the correct length (16000 samples per second)
            if len(audio_data) < 16000:
                print("Audio data too short, padding with zeros")
                audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)))
            elif len(audio_data) > 16000:
                print("Audio data too long, truncating")
                audio_data = audio_data[:16000]
            
            transcription = transcribe_audio(audio_data)
            print("Transcription:", transcription)
            await websocket.send(transcription)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"Error in handle_websocket: {e}")

async def main():
    load_model()
    server = await websockets.serve(handle_websocket, "localhost", 65432)
    print(f"Server listening on 127.0.0.1:65432")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())

#_________________________________________________________________________
import asyncio
import websockets
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np

processor, model = None, None

def load_model():
    global processor, model
    processor = AutoProcessor.from_pretrained("Seyfelislem/whisper-medium-arabic")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("Seyfelislem/whisper-medium-arabic")

def transcribe_audio(audio):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        outputs = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

def process_audio_in_batches(audio_data, batch_size=200000):
    transcriptions = []
    for i in range(0, len(audio_data), batch_size):
        print("processing: ",i)
        batch = audio_data[i:i+batch_size]
        if len(batch) < batch_size:
            batch = np.pad(batch, (0, batch_size - len(batch)))
        transcription = transcribe_audio(batch)
        transcriptions.append(transcription)
    return ' '.join(transcriptions)

async def handle_websocket(websocket, path):
    try:
        async for message in websocket:
            print("Received audio data")
            audio_data = np.frombuffer(message, dtype=np.float32)
            print("Audio data shape:", audio_data.shape)
            
            transcription = process_audio_in_batches(audio_data)
            print("Full Transcription:", transcription)
            await websocket.send(transcription)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"Error in handle_websocket: {e}")

async def main():
    load_model()
    server = await websockets.serve(handle_websocket, "localhost", 65432)
    print(f"Server listening on 127.0.0.1:65432")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
