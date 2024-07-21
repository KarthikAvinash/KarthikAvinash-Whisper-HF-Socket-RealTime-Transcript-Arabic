# import asyncio
# import websockets
# from transformers import pipeline
# import numpy as np
# import io
# import torch
# import time
# import tempfile
# import os
# import subprocess

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# def load_model():
#     global pipe
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#     pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
#     print("Model loaded successfully")

# # Initialize the transcription pipeline
# load_model()

# async def process_audio(websocket, path):
#     global pipe
#     temp_webm_path = None
#     wav_path = None
#     try:
#         print("Waiting for audio content...")
#         start_time = time.time()
#         audio_content = await websocket.recv()
#         print(f"Received audio content of type: {type(audio_content)}")
#         print(f"Audio content length: {len(audio_content)} bytes")

#         # Save the WebM audio to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", mode="wb") as temp_webm:
#             temp_webm.write(audio_content)
#             temp_webm_path = temp_webm.name
#         print(f"Saved WebM audio to: {temp_webm_path}")

#         # Convert WebM to WAV using ffmpeg
#         wav_path = temp_webm_path.replace(".webm", ".wav")
#         ffmpeg_command = f"ffmpeg -i {temp_webm_path} -acodec pcm_s16le -ar 16000 -ac 1 {wav_path}"
#         print(f"Running ffmpeg command: {ffmpeg_command}")
#         subprocess.run(ffmpeg_command, shell=True, check=True)
#         print(f"Converted to WAV: {wav_path}")

#         # Load the WAV file
#         with open(wav_path, 'rb') as wav_file:
#             audio_array = np.frombuffer(wav_file.read(), dtype=np.int16)
#         audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
#         sampling_rate = 16000  # We set this in the ffmpeg command

#         # Transcribe the audio
#         print("Transcribing audio...")
#         result = pipe(audio_array)
#         transcription = result['text']

#         # Send the transcription result back to the client
#         await websocket.send(transcription)
#         print(f"Transcription sent to client: {transcription}")

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"Time taken to process and transcribe: {elapsed_time:.2f} seconds")

#     except Exception as e:
#         print(f'Error: {e}')
#         print("Reloading model...")
#         load_model()
#         await websocket.send('Error occurred while processing the file. Model has been reloaded.')

#     finally:
#         # Clean up temporary files
#         if temp_webm_path and os.path.exists(temp_webm_path):
#             os.unlink(temp_webm_path)
#         if wav_path and os.path.exists(wav_path):
#             os.unlink(wav_path)

# async def main():
#     address = "localhost"
#     port = 8080
#     server = await websockets.serve(process_audio, address, port)
#     print(f"Server listening on ws://{address}:{port}")
#     await asyncio.Future()

# if __name__ == "__main__":
#     asyncio.run(main())

#_______________________________________________________________
# import asyncio
# import websockets
# from transformers import pipeline
# import numpy as np
# import io
# import torch
# import time
# import tempfile
# import os
# import subprocess
# import logging

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def load_model():
#     global pipe
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"Using device: {device}")
#     pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
#     logger.info("Model loaded successfully")

# # Initialize the transcription pipeline
# load_model()

# async def process_audio(websocket, path):
#     global pipe
#     temp_webm_path = None
#     wav_path = None
#     try:
#         while True:
#             logger.info("Waiting for audio content...")
#             start_time = time.time()
#             audio_content = await websocket.recv()
#             logger.info(f"Received audio content of type: {type(audio_content)}")
#             logger.info(f"Audio content length: {len(audio_content)} bytes")

#             # Save the WebM audio to a temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", mode="wb") as temp_webm:
#                 temp_webm.write(audio_content)
#                 temp_webm_path = temp_webm.name
#             logger.info(f"Saved WebM audio to: {temp_webm_path}")

#             # Verify if the WebM file is complete and valid
#             if not os.path.exists(temp_webm_path):
#                 logger.error(f"WebM file not found: {temp_webm_path}")
#                 await websocket.send('Error: WebM file not found.')
#                 continue

#             if os.path.getsize(temp_webm_path) < 1000:
#                 logger.error(f"WebM file too small: {temp_webm_path}")
#                 await websocket.send('Error: WebM file too small.')
#                 continue

#             # Convert WebM to WAV using ffmpeg
#             wav_path = temp_webm_path.replace(".webm", ".wav")
#             ffmpeg_command = f"ffmpeg -loglevel debug -i {temp_webm_path} -acodec pcm_s16le -ar 16000 -ac 1 {wav_path}"
#             logger.info(f"Running ffmpeg command: {ffmpeg_command}")
#             try:
#                 subprocess.run(ffmpeg_command, shell=True, check=True)
#             except subprocess.CalledProcessError as e:
#                 logger.error(f"FFmpeg command failed: {e}")
#                 await websocket.send('Error: Failed to convert audio file.')
#                 continue
#             logger.info(f"Converted to WAV: {wav_path}")

#             # Load the WAV file
#             with open(wav_path, 'rb') as wav_file:
#                 audio_array = np.frombuffer(wav_file.read(), dtype=np.int16)
#             audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
#             sampling_rate = 16000  # We set this in the ffmpeg command

#             # Transcribe the audio
#             logger.info("Transcribing audio...")
#             result = pipe(audio_array)
#             transcription = result['text']

#             # Send the transcription result back to the client
#             await websocket.send(transcription)
#             logger.info(f"Transcription sent to client: {transcription}")

#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             logger.info(f"Time taken to process and transcribe: {elapsed_time:.2f} seconds")

#     except Exception as e:
#         logger.error(f'Error: {e}', exc_info=True)
#         logger.info("Reloading model...")
#         load_model()
#         await websocket.send('Error occurred while processing the file. Model has been reloaded.')

#     finally:
#         # Clean up temporary files
#         if temp_webm_path and os.path.exists(temp_webm_path):
#             os.unlink(temp_webm_path)
#         if wav_path and os.path.exists(wav_path):
#             os.unlink(wav_path)

# async def main():
#     address = "localhost"
#     port = 8080
#     server = await websockets.serve(process_audio, address, port)
#     logger.info(f"Server listening on ws://{address}:{port}")
#     await asyncio.Future()

# if __name__ == "__main__":
#     asyncio.run(main())

#_____________________________________________________________
# import asyncio
# import websockets
# from transformers import pipeline
# import numpy as np
# import torch
# import logging
# import io
# from pydub import AudioSegment

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load model once at startup
# device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe = pipeline("automatic-speech-recognition", model="Seyfelislem/whisper-medium-arabic", device=device)
# logger.info(f"Model loaded successfully on device: {device}")

# class AudioBuffer:
#     def __init__(self, max_size=48000):  # 3 seconds at 16kHz
#         self.buffer = np.array([], dtype=np.float32)
#         self.max_size = max_size

#     def add(self, audio):
#         self.buffer = np.concatenate([self.buffer, audio])
#         if len(self.buffer) > self.max_size:
#             self.buffer = self.buffer[-self.max_size:]

#     def get(self):
#         return self.buffer

# async def process_audio(websocket, path):
#     buffer = AudioBuffer()
#     try:
#         while True:
#             audio_content = await websocket.recv()
#             logger.info(f"Received audio content of length: {len(audio_content)} bytes")

#             # Convert WebM to WAV in memory
#             audio = AudioSegment.from_file(io.BytesIO(audio_content), format="webm")
#             wav_data = io.BytesIO()
#             audio.export(wav_data, format="wav")
#             wav_data.seek(0)

#             # Convert to numpy array
#             audio_array = np.frombuffer(wav_data.read(), dtype=np.int16)
#             audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]

#             # Add to buffer
#             buffer.add(audio_array)

#             # Transcribe if buffer is full enough
#             if len(buffer.get()) >= 16000:  # Process at least 1 second of audio
#                 result = pipe(buffer.get())
#                 transcription = result['text']
#                 await websocket.send(transcription)
#                 logger.info(f"Transcription sent to client: {transcription}")

#     except websockets.exceptions.ConnectionClosed:
#         logger.info("Client disconnected")
#     except Exception as e:
#         logger.error(f'Error: {e}', exc_info=True)
#         await websocket.send('Error occurred while processing the audio.')

# async def main():
#     address = "localhost"
#     port = 8080
#     server = await websockets.serve(process_audio, address, port)
#     logger.info(f"Server listening on ws://{address}:{port}")
#     await asyncio.Future()

# if __name__ == "__main__":
#     asyncio.run(main())

#________________________________________________________________

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
            with open("temp.wav","wb") as f:
                f.write(audio_content)
            audio_path = "temp.wav"
            audio_array , sampling_rate = librosa.load(audio_path, sr=16000)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            # Transcribe the audio
            transcription = pipe(audio_array) ["text"]
            # Send the transcription result back to the client
            await websocket.send(transcription)
            print(f"Transcription sent to client: {transcription}")

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")

async def main():
    server = await websockets.serve(process_audio, "localhost", 8080)
    logger.info("Server started on ws://localhost:8080")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
