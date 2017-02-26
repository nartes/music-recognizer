import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    '../deps/sms-tools/software/models/'))

from scipy.signal import get_window
import numpy as np
import math
import dftModel as DFT
import utilFunctions as UF

class OnlineSTFT:
    N = 4096
    M = 1801
    H = M / 4
    w = get_window('blackman', M)
    # half analysis window size by rounding
    hM1 = int(math.floor((w.size+1)/2))
    # half analysis window size by floor
    hM2 = int(math.floor(w.size/2))
    w = w / sum(w)
    t = -100
    fs = 44100
    harmDevSlope = 0.01
    minSineDur = .02
    f0et = 20
    minf0 = 100
    maxf0 = 300
    nH = 60

    def __init__(self):
        self.frames = np.zeros(self.hM2)
        self.magnitudes = np.array([])
        self.phases = np.array([])
        self.fundamentals = np.array([])
        self.pin = self.hM1

    def proc_frame(self, frame):
        self.frames = np.append(self.frames, frame)
        pend = self.frames.size - self.hM1
	# initialize f0 track
	f0t = 0
	# initialize f0 stable
	f0stable = 0

        while self.pin<pend:
            # select frame
            x1 = self.frames[self.pin-self.hM1:self.pin+self.hM2]
            # compute dft
            mX, pX = DFT.dftAnal(x1, self.w, self.N)
            if self.pin == self.hM1:
                self.magnitudes = mX
                self.phases = pX
            else:
                self.magnitudes = np.vstack((self.magnitudes, mX))
                self.phases = np.vstack((self.phases, mX))

            # detect peak locations
            ploc = UF.peakDetection(mX, self.t)
            # refine peak values
            iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
            # convert locations to Hz
            ipfreq = self.fs * iploc/self.N
            # find f0
            f0t = UF.f0Twm(ipfreq, ipmag, self.f0et, \
                           self.minf0, self.maxf0, f0stable)
            if ((f0stable==0)&(f0t>0)) \
                            or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
                    # consider a stable f0 if it is close to the previous one
                    f0stable = f0t
            else:
                    f0stable = 0

            self.fundamentals = np.append(self.fundamentals, f0t)
            print f0t

            self.pin += self.H
