import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    '../tmp/sms-tools/software/models/'))

from scipy.signal import get_window
import numpy as np
import math
import dftModel as DFT
import utilFunctions as UF

class OnlineSTFT:
    N = 4096
    M = 2501
    H = M / 4
    w = get_window('hamming', M)
    # half analysis window size by rounding
    hM1 = int(math.floor((w.size+1)/2))
    # half analysis window size by floor
    hM2 = int(math.floor(w.size/2))
    w = w / sum(w)

    def __init__(self):
        self.frames = np.zeros(self.hM2)
        self.spectrogram = np.array([])
        self.pin = self.hM1

    def proc_frame(self, frame):
        self.frames = np.append(self.frames, frame)
        pend = self.frames.size - self.hM1

        while self.pin<pend:
            # select frame
            x1 = self.frames[self.pin-self.hM1:self.pin+self.hM2]
            # compute dft
            mX, pX = DFT.dftAnal(x1, self.w, self.N)
            if self.pin == self.hM1:
                self.spectrogram = mX
            else:
                self.spectrogram = np.vstack((self.spectrogram, mX))
            self.pin += self.H
