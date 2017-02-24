"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave
from OnlineSTFT import OnlineSTFT
import numpy as np
import struct
import sys
import matplotlib.pyplot as plt
from utilFunctions import wavread

CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "build/output.wav"

def wav_test(ostft, data):
    fs, x = wavread(sys.argv[2])

    for pin in np.arange(0, len(x), CHUNK):
        frame = x[pin : pin + CHUNK]
        proc_frame(ostft, frame)

def mic_test(ostft, data):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        datum = stream.read(CHUNK)
        data = np.append(data, datum)
        frame = unpack_frame(datum)
        proc_frame(ostft, frame)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()

def proc_frame(ostft, frame):
    ostft.proc_frame(frame)
    print ostft.frames.size
    print ostft.spectrogram.size

def unpack_frame(datum):
    return np.array(struct.unpack('%dh' % (len(datum)/2), datum))

if __name__ == "__main__":
    ostft = OnlineSTFT()

    data = np.array([]);

    if sys.argv[1] == 'mic':
        mic_test(ostft, data)
    elif sys.argv[1] == 'wav':
        wav_test(ostft, data)

    if sys.argv[-1] == 'debug':
        plt.pcolormesh(np.transpose(ostft.spectrogram)); plt.show()
