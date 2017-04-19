import librosa
import librosa.display
import numpy
import matplotlib.pyplot as plt

def nmf(V, r = 2, tol = 1e-2,
        max_iter = 100, attempts = 5):
    """

       Non-Negative Matrix Factorization algorithm.
       Implemented from the paper by Lee and Simon.

    """
    n, m = V.shape

    PERGRADIENT = max_iter
    count = 0
    MAX_ITER = attempts * PERGRADIENT

    bestW = numpy.random.uniform(size = (n, r))
    bestH = numpy.random.uniform(size = (r, m))
    bestd = numpy.inf

    while True:
        W = numpy.random.uniform(size = (n, r))
        H = numpy.random.uniform(size = (r, m))

        def updateH(V, H, W):
            return H * (W.T.dot(V)) / (W.T.dot(W).dot(H))

        def updateW(V, H, W):
            return W * (V.dot(H.T)) / (W.dot(H).dot(H.T))

        def D(A, B):
            return numpy.sum(A * numpy.log(A / B) - A + B)

        def N(A, B):
            return ((A - B) ** 2).sum()

        prev_diff = None
        cur_diff = None

        while True:
            tH = updateH(V, H, W)
            H[~numpy.isnan(tH)] = tH[~numpy.isnan(tH)]
            tW = updateW(V, H, W)
            W[~numpy.isnan(tW)] = tW[~numpy.isnan(tW)]

            count += 1

            prev_diff = cur_diff
            cur_diff = N(V, W.dot(H))
            print(cur_diff)
            if prev_diff != None and abs(cur_diff - prev_diff) < tol or\
               count % PERGRADIENT == 0:
                if count % PERGRADIENT != 0:
                    count = (count // PERGRADIENT + 1) * PERGRADIENT
                break

        d = N(V, W.dot(H))

        if d < bestd:
            bestH = H.copy()
            bestW = W.copy()
            bestd = d

        if d < 0.1 or count == MAX_ITER:
            break;

    return (bestW, bestH, bestd, count)

def realaudio_nmf_post(fname = 'tmp/organ.wav', M = 2048,
                       N = 4096, MFH = 6000, TR = 5.5, R = 4,
                       fs = 44100, tol = 1e-2):
    # TODO(nartes): parse wave header with an analogous function
    # to octave's audioinfo to obtain Frequency of Sampling

    FS = fs
    HS = int(numpy.floor(M * 0.25))
    MFS = int(numpy.floor(MFH / FS * N))

    TL = 0.0;
    SL = int(numpy.floor(TL * FS))
    SR = int(numpy.floor(TR * FS))

    x = librosa.load(fname, fs, mono = True)[SL : SR][0]
    y = librosa.stft(x, n_fft = N // 2, hop_length = HS, win_length = M)
    L = y.shape[1]

    sy = y[0 : MFS, :]

    W, H = nmf(numpy.abs(sy), R, tol)[:2]

    plot_grayscale_spectrogram(sy, HS, FS)
    plot_h_matrix(L, M, HS, FS, TL, R, H)
    plot_w_matrix(FS, N, MFS, R, W)

    plt.show()

def plot_grayscale_spectrogram(sy, HS, FS):
    librosa.display.specshow(sy, y_axis = 'log', x_axis = 'time',
                             hop_length = HS, sr = FS)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')

def plot_h_matrix(L, M, HS, FS, TL, R, H):
    tsec = (numpy.arange(0, L) * HS + M / 2) / FS + TL
    plt.figure()
    for k in numpy.arange(1, R + 1):
        plt.subplot(R, 1, k)
        plt.plot(tsec, H[k - 1, :])
        plt.xlabel('sec')
        plt.ylabel('amplitude')

def plot_w_matrix(FS, N, MFS, R, W):
    fhz = numpy.arange(1, MFS + 1) / N * FS

    plt.figure()
    for k in numpy.arange(1, R + 1):
        plt.subplot(R, 1, k)
        plt.plot(fhz, W[:, k - 1].T)
        plt.xlabel('Hz')
        plt.ylabel('amplitude')