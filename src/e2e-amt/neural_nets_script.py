import keras
import keras.models
import keras.layers
import keras.optimizers
import keras.utils
import numpy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import nmf_tools
import glob
import sys
import io

def gen_seq(NOISE_AMPL = 0.5 * 1e-2,\
            ORDER = 2, AHEAD = 1, N = 4,
            SL = 1000, L = 10, f_name = 'sin'):
    S = numpy.__dict__[f_name](1.0 * numpy.arange(SL) / N * 2 * numpy.pi) +\
        numpy.random.rand(SL) * NOISE_AMPL
    K = ORDER + AHEAD
    T = numpy.random.randint(0, SL - K, L)
    XY = S[(T + numpy.reshape(numpy.arange(K), (K, 1))).T]
    X = XY[:, 0 : ORDER]
    Y = XY[:, ORDER : K]

    return X, Y

def sin_model():
    model = keras.models.Sequential()
    for k in numpy.arange(1):
        DW = 20
        if k > 1:
            model.add(keras.layers.Dense(DW, kernel_initializer = 'uniform'))
        else:
            model.add(keras.layers.Dense(DW, input_dim = 2,\
                      kernel_initializer = "uniform"))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(3))
    model.add(keras.layers.Activation('softmax'))

    model.compile(loss = 'mse',\
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])

    model.summary()

    return model

def sin_model2():
    model = keras.models.Sequential()
    for k in numpy.arange(1):
        DW = 512
        if k > 1:
            model.add(keras.layers.Dense(DW, kernel_initializer = 'uniform'))
        else:
            model.add(keras.layers.Dense(DW, input_dim = 2,\
                      kernel_initializer = "uniform"))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation('softmax'))
    model.add(keras.layers.Dense(1))

    model.compile(loss = 'mae',\
                  optimizer = 'adadelta',\
                  metrics = ['mse', 'mae'])

    model.summary()

    return model

def test1():
    nb_classes = 3
    batch_size = 64
    nb_epoch = 1
    X_train, Y_train = gen_seq(L = 60000, NOISE_AMPL = 5e-2)
    X_test, Y_test = gen_seq(L = 100)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = keras.utils.np_utils.to_categorical(\
            numpy.array(numpy.round(Y_train), dtype = numpy.int), nb_classes)
    Y_test = keras.utils.np_utils.to_categorical(\
            numpy.array(numpy.round(Y_test), dtype = numpy.int), nb_classes)

    model = sin_model()
    model.fit(X_train, Y_train,\
              epochs = nb_epoch,\
              batch_size = batch_size,\
              verbose = 1, validation_data = (X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose = 0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return model

def test2(batch_size = 128, nb_epoch = 3, N = 10):
    X_train, Y_train = gen_seq(\
            N = N, L = 60000, NOISE_AMPL = 5e-4, f_name = 'cos')
    X_test, Y_test = gen_seq(N = N, L = 1000)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    #Y_train = keras.utils.np_utils.to_categorical(\
    #        numpy.array(numpy.round(Y_train), dtype = numpy.int), nb_classes)
    #Y_test = keras.utils.np_utils.to_categorical(\
    #        numpy.array(numpy.round(Y_test), dtype = numpy.int), nb_classes)

    model = sin_model2()
    model.fit(X_train, Y_train,\
              epochs = nb_epoch,\
              batch_size = batch_size,\
              verbose = 1, validation_data = (X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose = 0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print_test_samples(N = N, model = model)

    return model

def print_test_samples(N, model):
    for f_name in ['sin', 'cos']:
        a = gen_seq(N = N, L = 10, NOISE_AMPL = 0,
                    f_name = f_name)
        b = model.predict(numpy.array(a[0]));
        print(f_name,\
              '\n\n=====\n\n',\
              'a[1]\n\n',
              a[1],\
              '\n\n***\n\n',\
              'b\n\n',
              b,\
              '\n\n***\n\n',\
              'a[1] - b\n\n',
              a[1] - b)

def dumb_amt_model():
    model = keras.models.Sequential()

    model.add(keras.layers.SimpleRNN(
            batch_input_shape = (None, 1, 252),
            units = 32,
            activation = 'relu'))
    model.add(keras.layers.Dense(512, activation = 'relu'))
    model.add(keras.layers.Dense(88, activation = 'softmax'))
    model.add(keras.layers.Reshape(target_shape = (88, 1)))
    model.add(keras.layers.SimpleRNN(88, activation = 'relu'))

    model.compile(loss = 'mae',\
                  optimizer = 'adadelta',\
                  metrics = ['mse', 'mae'])

    return model

def dnn_amt_model(loss = 'binary_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy']):
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(
            batch_input_shape = (None, 1, 252),
            units = 64,
            kernel_initializer = 'uniform',
            activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(512, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(88, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(88, activation = 'sigmoid'))
    model.add(keras.layers.Reshape(target_shape = (88,)))
    #model.add(keras.layers.Dropout(0.3))

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model

def parse_notes(txt_fname,
                num_frames,
                n_bins = 252,
                bins_per_octave = 36):
    notes = numpy.loadtxt(fname = txt_fname, skiprows = 1, ndmin = 2)

    y_seq = nmf_tools.maps_notes_to_y_seq(notes)

    Y_test = y_seq[:, :num_frames].T

    return Y_test, y_seq

def process_C(wav_fname,
              n_bins = 252,
              bins_per_octave = 36):
    C = nmf_tools.wav_to_cqt(fname = wav_fname,
                             n_bins = n_bins,
                             bins_per_octave = bins_per_octave)[1]
    C = numpy.abs(C)
    C = nmf_tools.normalize_cqt(C)

    X = C.T.reshape(-1, 1, C.shape[0])

    return X

def set_cached_data(txt_fname, wav_fname, X, Y_test, ignore = False):
    cache_file = txt_fname[:-4] + '.meta.npz'

    if len(glob.glob(cache_file)) >= 1 and not ignore:
        return False

    try:
        numpy.savez(cache_file, X = X, Y_test = Y_test)
        return True
    except:
        return False


def get_cached_data(txt_fname, wav_fname):
    cache_file = txt_fname[:-4] + '.meta.npz'

    try:
        metadata = numpy.load(cache_file)
        X = metadata['X']
        Y_test = metadata['Y_test']
        return X, Y_test
    except:
        return None, None

def generate_training_data(n_bins = 252,
                  bins_per_octave = 36,
                  hop_length = 512,
                  sr = 16000,
                  o_sr = 44100,
                  fmin = librosa.note_to_hz('C1'),
                  max_files = 10,
                  batch_size = 1,
                  glob_param = 'ISOL/NO',
                  ignore = False):
    txts = glob.glob(\
            'tmp/MAPS-dataset/*/' + glob_param + '/*.txt')
    wavs = glob.glob(\
            'tmp/MAPS-dataset/*/' + glob_param + '/*.wav')

    if max_files == -1 or max_files is None:
        max_files = len(txts)

    X = None
    Y_test = None

    for txt_fname in txts:
        wav_fname = txt_fname[:-4] + '.wav'

        print(f'Left {max_files}')
        print(f'File: {txt_fname} started.')

        cur_X = None
        cur_Y_test = None

        if not ignore:
            cur_X, cur_Y_test = get_cached_data(txt_fname, wav_fname)

        if cur_X is None or cur_Y_test is None:
            cur_X = process_C(wav_fname = wav_fname,
                          n_bins = n_bins,
                          bins_per_octave = bins_per_octave)
            cur_Y_test, y_seq = parse_notes(\
                    txt_fname = txt_fname,
                    n_bins = n_bins,
                    bins_per_octave = bins_per_octave,
                    num_frames = cur_X.shape[0])

            set_cached_data(txt_fname = txt_fname,
                            wav_fname = wav_fname,
                            X = cur_X,
                            Y_test = cur_Y_test,
                            ignore = ignore)
        else:
            print('Found a cached file.')

        if X is None:
            X = cur_X
        else:
            X = numpy.vstack((X, cur_X))

        if Y_test is None:
            Y_test = cur_Y_test
        else:
            Y_test = numpy.vstack((Y_test, cur_Y_test))

        print(f'File: {txt_fname} ended.')

        max_files -= 1
        if max_files == 0:
            break

    return X, Y_test

def dumb_amt_test(model = dumb_amt_model(),
                  n_bins = 252,
                  bins_per_octave = 36,
                  hop_length = 512,
                  sr = 16000,
                  o_sr = 44100,
                  fmin = librosa.note_to_hz('C1'),
                  plot_each = False,
                  max_files = 10,
                  batch_size = 1,
                  glob_param = 'ISOL/NO'):

    X, Y_test = generate_training_data(n_bins = n_bins,
            bins_per_octave = bins_per_octave,
            hop_length = hop_length,
            sr = sr,
            o_sr = o_sr,
            fmin = fmin,
            max_files = max_files,
            batch_size = batch_size,
            glob_param = glob_param)

    model.fit(X, Y_test, epochs = 1, batch_size = 1, verbose = 1,
              validation_data = (X, Y_test))

    return model

def plot_processing(p_y_seq,
        y_seq,
        sr = 16000,
        bins_per_octave = 36,
        hop_length = 512):
    plt.figure()
    plot_y_seq(y_seq, sr = sr, bins_per_octave = bins_per_octave,
               hop_length = hop_length,
               title = 'Ground truth midi transcription')


    plt.figure()
    plot_y_seq(p_y_seq, sr = sr, bins_per_octave = bins_per_octave,
               hop_length = hop_length,
               title = 'Predicted midi transcription')

    plt.show()

def plot_y_seq(y_seq, sr, bins_per_octave, hop_length, title):
    librosa.display.specshow(librosa.amplitude_to_db(y_seq, ref = numpy.max),
                             sr = sr, x_axis = 'time', y_axis = 'cqt_note',\
                             bins_per_octave = bins_per_octave,
                             hop_length = hop_length)
    plt.title(title)
    plt.tight_layout()

