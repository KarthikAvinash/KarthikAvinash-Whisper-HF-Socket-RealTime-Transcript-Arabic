# import asyncio
# import websockets
# import os
# import datetime
# import librosa
# import numpy as np
# from transformers import pipeline
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
# pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
# print("Model loaded successfully")

# # Ensure the chunks directory exists
# if not os.path.exists('chunks'):
#     os.makedirs('chunks')

# async def save_audio(websocket, path):
#     async for message in websocket:
#         print('Received audio data from client.')
#         # Save the received audio data
#         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         storing_file_name = "temp.wav"
#         file_path = os.path.join('chunks', storing_file_name)
        
#         with open(file_path, 'wb') as f:
#             f.write(message)
#             print(f'Saved audio file to {file_path}')
        
#         audio_array, sampling_rate = librosa.load(file_path, sr=16000)
#         # Ensure the audio is mono
#         if len(audio_array.shape) > 1:
#             audio_array = np.mean(audio_array, axis=1)
        
#         print("Processing... Please wait.")
#         # Transcribe the audio
#         result = pipe(audio_array)

#         print(f"Transcription: {result['text']}")

#         # Send transcription to the client
#         await websocket.send(result['text'])

#         # Send signal to send next chunk
#         await websocket.send("You can send next chunk")

# start_server = websockets.serve(save_audio, "localhost", 8765)

# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()


# import asyncio
# import websockets
# import os
# import datetime
# import librosa
# import numpy as np
# from transformers import pipeline
# import torch
# import logging
# import uuid
# import shutil

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize device and ASR pipeline
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logging.info(f"Using device: {device}")
# try:
#     pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
#     logging.info("Model loaded successfully")
# except Exception as e:
#     logging.error("Error loading model: %s", e)
#     raise

# # Ensure the chunks directory exists
# if not os.path.exists('chunks'):
#     os.makedirs('chunks')
#     logging.info("Created 'chunks' directory")

# async def save_audio(websocket, path):
#     try:
#         async for message in websocket:
#             logging.info('Received audio data from client.')
            
#             # Generate a unique filename for each client
#             unique_id = str(uuid.uuid4())
#             storing_file_name = f"temp_{unique_id}.wav"
#             file_path = os.path.join('chunks', storing_file_name)
            
#             with open(file_path, 'wb') as f:
#                 f.write(message)
#                 logging.info(f'Saved audio file to {file_path}')
            
#             # Load and process the audio file
#             audio_array, sampling_rate = librosa.load(file_path, sr=16000)
#             if len(audio_array.shape) > 1:
#                 audio_array = np.mean(audio_array, axis=1)
            
#             logging.info("Processing audio... Please wait.")
#             result = pipe(audio_array)
#             logging.info(f"Transcription: {result['text']}")

#             # Send transcription to the client
#             await websocket.send(result['text'])
#             await websocket.send("You can send next chunk")

#             # Clean up the file after processing
#             os.remove(file_path)
#             logging.info(f"Deleted audio file {file_path} after processing.")
#     except Exception as e:
#         logging.error("Error processing audio: %s", e)
#         await websocket.send("An error occurred during processing. Please try again.")
#         # Attempt to clean up in case of an error
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             logging.info(f"Deleted audio file {file_path} due to an error.")

# async def main():
#     async with websockets.serve(save_audio, "0.0.0.0", 8765):
#         logging.info("WebSocket server started on ws://0.0.0.0:8765")
#         await asyncio.Future()  # Run forever

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except Exception as e:
#         logging.error("Error starting server: %s", e)
#         raise


import asyncio
import websockets
import os
import datetime
import librosa
import numpy as np
from transformers import pipeline
import torch
import logging
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize device and ASR pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
try:
    pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Error loading model: %s", e)
    raise

# Ensure the chunks directory exists
if not os.path.exists('chunks'):
    os.makedirs('chunks')
    logging.info("Created 'chunks' directory")

async def process_audio(websocket, message):
    try:
        logging.info('Received audio data from client.')
        
        # Generate a unique filename for each client
        unique_id = str(uuid.uuid4())
        storing_file_name = f"temp_{unique_id}.wav"
        file_path = os.path.join('chunks', storing_file_name)
        
        with open(file_path, 'wb') as f:
            f.write(message)
            logging.info(f'Saved audio file to {file_path}')
        
        # Load and process the audio file
        audio_array, sampling_rate = librosa.load(file_path, sr=16000)
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        logging.info("Processing audio... Please wait.")
        result = pipe(audio_array)
        logging.info(f"Transcription: {result['text']}")

        # Send transcription to the client
        await websocket.send(result['text'])
        await websocket.send("You can send next chunk")

        # Clean up the file after processing
        os.remove(file_path)
        logging.info(f"Deleted audio file {file_path} after processing.")
    except Exception as e:
        logging.error("Error processing audio: %s", e)
        await websocket.send("An error occurred during processing. Please try again.")
        # Attempt to clean up in case of an error
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted audio file {file_path} due to an error.")

async def save_audio(websocket, path):
    try:
        async for message in websocket:
            # Create a new task for processing each audio message
            asyncio.create_task(process_audio(websocket, message))
    except Exception as e:
        logging.error("Error handling client: %s", e)
        await websocket.send("An error occurred. Please try again.")

async def main():
    async with websockets.serve(save_audio, "localhost", 8765):
        logging.info("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error("Error starting server: %s", e)
        raise
