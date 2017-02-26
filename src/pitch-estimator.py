"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave
from OnlineSTFT import OnlineSTFT
import numpy as np
import struct
import sys
import matplotlib.pyplot as plt
from utilFunctions import wavread
import socket

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

def websocket_mic_test(ostft, data):
    stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stream.connect(('127.0.0.1', 8124))

    while 1:
        datum = stream.recv(CHUNK)
        if not datum:
            break

        frame = unpack_frame(datum)
        proc_frame(ostft, frame)

    print("* done streaming")

    stream.close()

def proc_frame(ostft, frame):
    ostft.proc_frame(frame)

def unpack_frame(datum):
    return np.array(struct.unpack('%dh' % (len(datum)/2), datum))

def plot_magnitude_spectrogram(ostft):
    # frequency range to plot
    maxplotfreq = 5000.0

    numFrames = int(ostft.magnitudes[:,0].size)
    frmTime = ostft.H*np.arange(numFrames)/float(ostft.fs)
    binFreq = ostft.fs*np.arange(ostft.N*maxplotfreq/ostft.fs)/ostft.N
    plt.pcolormesh(frmTime, binFreq, \
                   np.transpose(ostft.magnitudes[:,:ostft.N*maxplotfreq/ostft.fs+1]))
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.autoscale(tight=True)

def plot_fundamental(ostft):
    # frequency range to plot
    maxplotfreq = 5000.0

    harms = ostft.fundamentals*np.less(ostft.fundamentals,maxplotfreq)
    numFrames = int(ostft.fundamentals.size)
    frmTime = ostft.H*np.arange(numFrames)/float(ostft.fs)
    plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
    plt.xlabel('time(s)')
    plt.ylabel('frequency(Hz)')
    plt.autoscale(tight=True)

if __name__ == "__main__":
    ostft = OnlineSTFT()

    data = np.array([]);

    if sys.argv[1] == 'mic':
        mic_test(ostft, data)
    elif sys.argv[1] == 'wav':
        wav_test(ostft, data)
    elif sys.argv[1] == 'websocket':
        websocket_mic_test(ostft, data)

    if sys.argv[-1] == 'debug':
        plot_magnitude_spectrogram(ostft)
        plot_fundamental(ostft)
        plt.show()
