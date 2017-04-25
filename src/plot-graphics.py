import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy

yin_rec = numpy.loadtxt('build/yin-default-record.txt')
twm_rec = numpy.loadtxt('build/twm-record.txt')

yin_sock = numpy.loadtxt('build/yin-default-daj-ci-boze-dobranoc.txt')
twm_sock = numpy.loadtxt('build/twm-daj-ci-boze-dobranoc.txt')

plt.figure()
ax1 = plt.subplot(111)
ax1.plot(twm_rec[:, 0], twm_rec[:, 1])
ax1.plot(yin_rec[:, 0], yin_rec[:, 1])
ax1.legend(['YIN Default', 'TWM'])
ax1.set_ylabel('Note')
ax1.set_xlabel('Sec')
ax1.yaxis.set_major_formatter(librosa.display.NoteFormatter())
plt.title('WebSocket Microphone Recording')

plt.figure()
ax1 = plt.subplot(111)
ax1.plot(twm_sock[:, 0], twm_sock[:, 1])
ax1.plot(yin_sock[:, 0], yin_sock[:, 1])
ax1.legend(['YIN Default', 'TWM'])
ax1.set_ylabel('Note')
ax1.set_xlabel('Sec')
ax1.yaxis.set_major_formatter(librosa.display.NoteFormatter())
plt.title('WAV file')
plt.show()
