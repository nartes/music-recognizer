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

def test3(fname = 'tmp/organ.wav', sr = 44100):
    y = librosa.load(fname, sr, mono = True)[0]
    C = librosa.cqt(y, sr = sr)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref = numpy.max),
                             sr = sr, x_axis = 'time', y_axis = 'cqt_note')
    plt.colorbar(format = '%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()

    plt.show()

def test4(fname = 'tmp/organ.wav', sr = 44100, d_sr = 16000,\
          R = 11, tol = 1e-2,\
          max_iter = 100, attempts = 5,\
          hop_length = 512, bins_per_octave=12, n_bins = 84):
    C = wav_to_cqt(fname, sr, downsample_sr = d_sr,\
                   n_bins = n_bins, bins_per_octave = bins_per_octave)[1]

    W, H = nmf(numpy.abs(C), R, tol, max_iter, attempts)[:2]

    L = C.shape[1]
    HS = hop_length
    MFS = C.shape[0]
    FS = sr
    TL = 0.0
    M = 2048
    N = 2048

    librosa.display.specshow(librosa.amplitude_to_db(C, ref = numpy.max),
                             sr = sr, x_axis = 'time', y_axis = 'cqt_note',\
                             bins_per_octave = bins_per_octave,
                             hop_length = hop_length)
    plt.colorbar(format = '%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()

    plot_h_matrix(L, M, HS, FS, TL, R, H)
    plot_w_matrix(FS, N, MFS, R, W)

    plt.show()

def wav_to_cqt(fname = 'tmp/organ.wav', original_sr = 44100,\
        downsample_sr = 16000,\
        n_bins = 84,\
        bins_per_octave = 12,\
        fmin = librosa.note_to_hz('C1')):
    y = librosa.load(fname, original_sr, mono = True)[0]
    y_ds = librosa.resample(y, original_sr, downsample_sr)
    C = librosa.cqt(y_ds, sr = downsample_sr, n_bins = n_bins,\
                    bins_per_octave = bins_per_octave)

    return y, C, y_ds

def normalize_cqt(C):
    C = C - numpy.mean(C)
    C = C / numpy.sqrt(numpy.var(C))
    return C

def maps_notes_to_y_seq(notes):
    MINN = librosa.note_to_midi('C1')
    MINT = 0.0
    MAXT = 10.0
    N_BINS = 252
    FPS = 16000 / 512
    y_seq = numpy.zeros((88, int(numpy.ceil(MAXT * FPS))), dtype = numpy.float)

    for k in numpy.arange(notes.shape[0]):
        cur_note = numpy.int(numpy.floor(numpy.float(notes[k][2]) - MINN))

        if cur_note < 0:
            continue

        start_frame = numpy.int(numpy.floor(\
                FPS * numpy.float(notes[k][0])\
                ))
        end_frame = numpy.int(numpy.floor(\
                FPS * numpy.float(notes[k][1])\
                ))
        for t in numpy.arange(start_frame, end_frame + 1):
            y_seq[cur_note][t] = 1.0;

    return y_seq
