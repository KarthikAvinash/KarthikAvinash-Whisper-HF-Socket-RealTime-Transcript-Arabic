import socket
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import time
import sys

def load_model():
    processor = AutoProcessor.from_pretrained("Seyfelislem/whisper-medium-arabic")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("Seyfelislem/whisper-medium-arabic")
    return processor, model

# Server settings
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on

def transcribe_audio(audio, processor, model):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        outputs = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

# def run_server():
#     processor, model = load_model()
    
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         print(f"Server listening on {HOST}:{PORT}")
        
#         while True:
#             try:
#                 conn, addr = s.accept()
#                 with conn:
#                     print(f"Connected by {addr}")
#                     audio_data = b""
#                     while True:
#                         chunk = conn.recv(4096)
#                         if not chunk:
#                             break
#                         audio_data += chunk
                    
#                     # Convert audio data to numpy array
#                     audio_np = np.frombuffer(audio_data, dtype=np.float32)
                    
#                     # Transcribe the audio
#                     transcription = transcribe_audio(audio_np, processor, model)
#                     print("Transcription:", transcription)
                    
#                     # Send the transcription back to the client
#                     conn.sendall(transcription.encode())
#             except Exception as e:
#                 print(f"An error occurred: {e}")
#                 print("Restarting server...")
#                 break


def run_server():
    processor, model = load_model()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            print(f"Connected by {addr}")
            audio_buffer = b""
            
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    audio_buffer += chunk
                    
                    # Process every 2 seconds of audio
                    if len(audio_buffer) >= 16000 * 4 * 2:  # 2 seconds of float32 audio
                        audio_to_process = audio_buffer[:16000*4*2]
                        audio_buffer = audio_buffer[16000*4*2:]
                        
                        audio_np = np.frombuffer(audio_to_process, dtype=np.float32)
                        transcription = transcribe_audio(audio_np, processor, model)
                        print("Transcription:", transcription)
                        conn.sendall(transcription.encode())
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                conn.close()

def main():
    while True:
        try:
            run_server()
        except KeyboardInterrupt:
            print("Server stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"Critical error: {e}")
            print("Attempting to restart server in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()

