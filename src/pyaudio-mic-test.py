"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave
from OnlineSTFT import OnlineSTFT
import numpy as np
import struct
import sys
import pdb
import matplotlib.pyplot as plt

CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

def wav_test(ostft, data):
    wf = wave.open(sys.argv[1], 'rb')

    datum = wf.readframes(CHUNK)

    while datum != '':
        datum = wf.readframes(CHUNK)
        print "Wav datum lengthn"
        print len(datum)
        proc_frame(ostft, datum)

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
        data = np.append(datum, data)
        proc_frame(ostft, datum)

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

def proc_frame(ostft, datum):
    frame = unpack_frame(datum)
    ostft.proc_frame(frame)
    print ostft.frames.size
    print ostft.spectrogram.size

def unpack_frame(datum):
    return np.array(struct.unpack('%dh' % (len(datum)/2), datum))

ostft = OnlineSTFT()

data = np.array([]);

wav_test(ostft, data)
