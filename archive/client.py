import pyaudio
import numpy as np
import socket
import time

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
INTERVAL_SECONDS = 30

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for 10 seconds...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)

def send_audio_and_get_transcription(audio_data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(audio_data)
        s.shutdown(socket.SHUT_WR)

        print("Waiting for transcription...")
        transcription = s.recv(1024).decode()
        return transcription

def main():
    while True:
        try:
            # Record audio
            audio_data = record_audio()

            # Send audio and get transcription
            transcription = send_audio_and_get_transcription(audio_data)
            print(f"Transcription received: {transcription}")

            # Wait for the remaining time in the 30-second interval
            wait_time = INTERVAL_SECONDS - RECORD_SECONDS
            print(f"Waiting for {wait_time} seconds before next recording...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()