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
import multiprocessing
import os
import h5py

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

def dumb_dnn_amt_model(loss = 'binary_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy']):
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(
            batch_input_shape = (None, 1, 252),
            units = 2048,
            kernel_initializer = 'uniform',
            activation = 'relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(88, activation = 'relu'))
    model.add(keras.layers.Reshape(target_shape = (88,)))

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model

def amt_frame_metrics(y_true, y_pred):
    TP = keras.backend.sum(
            keras.backend.abs(keras.backend.minimum(y_true, y_pred)),
            axis = None)
    TP_FN = keras.backend.sum(keras.backend.abs(y_true), axis = None)
    TP_FP = keras.backend.sum(keras.backend.abs(y_pred), axis = None)

    R = keras.backend.sum(TP / (TP_FN + 1e-6))

    P = keras.backend.sum(TP / (TP_FP + 1e-6))

    F = keras.backend.clip(2 * P * R / (P + R + 1e-6), 0.0, 1.0)

    return {'P': P, 'R': R, 'F': F}

def numpy_amt_frame_metrics(y_true, y_pred):
    TP = numpy.sum(
            numpy.abs(numpy.minimum(y_true, y_pred)),
            axis = None)
    TP_FN = numpy.sum(numpy.abs(y_true), axis = None)
    TP_FP = numpy.sum(numpy.abs(y_pred), axis = None)

    R = numpy.sum(TP / (TP_FN + 1e-6))

    P = numpy.sum(TP / (TP_FP + 1e-6))

    F = numpy.clip(2 * P * R / (P + R + 1e-6), 0.0, 1.0)

    return {'P': P, 'R': R, 'F': F}

def frame_loss_f_measure(y_true, y_pred):
    return 1.0 - amt_frame_metrics(y_true, y_pred)['F']

def dnn_amt_model(loss = frame_loss_f_measure,
                  optimizer = 'adadelta',
                  metrics = ['accuracy'],
                  batch_size = 30):
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(
            input_shape = (1, 252),
            batch_size = batch_size,
            units = 1024,
            kernel_initializer = 'uniform',
            activation = 'relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(
            units = 88,
            kernel_initializer = 'uniform',
            stateful = True,
            return_sequences = True,
            activation = 'relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(88, activation = 'relu'))
    model.add(keras.layers.Reshape(target_shape = (88,)))

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    return model

def parse_notes(txt_fname,
                num_frames,
                n_bins = 252,
                decay = False,
                bins_per_octave = 36):
    notes = numpy.loadtxt(fname = txt_fname, skiprows = 1, ndmin = 2)

    y_seq = nmf_tools.maps_notes_to_y_seq(notes,
                                          decay = decay,
                                          nframes = num_frames)
    Y_test = y_seq.T

    return Y_test, y_seq

def process_C(wav_fname,
              n_bins = 252,
              bins_per_octave = 36):
    C = nmf_tools.wav_to_cqt(fname = wav_fname,
                             n_bins = n_bins,
                             bins_per_octave = bins_per_octave)[1]
    C = numpy.abs(C)
    C = nmf_tools.normalize_cqt(C)

    C += numpy.min(C)

    X = C.T.reshape(-1, 1, C.shape[0])

    return X

def set_cached_data(txt_fname = None,
                    wav_fname = None,
                    X = None,
                    Y_test = None,
                    ignore = False,
                    cache_file = None):
    if cache_file is None:
        cache_file = txt_fname[:-4] + '.meta.npz'

    if len(glob.glob(cache_file)) >= 1 and not ignore:
        return False

    try:
        numpy.savez(cache_file, X = X, Y_test = Y_test)
        return True
    except:
        return False

def get_cached_data(txt_fname = None, wav_fname = None, cache_file = None):
    if cache_file is None:
        cache_file = txt_fname[:-4] + '.meta.npz'

    try:
        metadata = numpy.load(cache_file)
        X = metadata['X']
        Y_test = metadata['Y_test']
        return X, Y_test
    except:
        return None, None

def get_globs(glob_param):
    txts = glob.glob(\
            'tmp/MAPS-dataset/*/' + glob_param + '/*.txt', recursive = True)
    wavs = glob.glob(\
            'tmp/MAPS-dataset/*/' + glob_param + '/*.wav', recursive = True)

    return txts, wavs

def proc_one_file(params):
    params, txt_fname, wav_fname, cur = params

    wav_fname = txt_fname[:-4] + '.wav'
    print(f'File: #{cur} {txt_fname} started.')

    cur_X = None
    cur_Y_test = None

    if not params['ignore']:
        ignore_x, ignore_y = (True, True)
        cur_X, cur_Y_test = get_cached_data(txt_fname, wav_fname)

    if cur_X is None or cur_Y_test is None or\
       params['ignore_x'] or params['ignore_y']:
        if params['ignore_x']:
            print('Regenerate X')
            cur_X = process_C(wav_fname = wav_fname,
                          n_bins = params['n_bins'],
                          bins_per_octave = params['bins_per_octave'])
        if params['ignore_y']:
            print('Regenerate Y_test')
            cur_Y_test, y_seq = parse_notes(\
                    txt_fname = txt_fname,
                    n_bins = params['n_bins'],
                    decay = params['decay'],
                    bins_per_octave = params['bins_per_octave'],
                    num_frames = cur_X.shape[0])

        print('New cache saved')
        set_cached_data(txt_fname = txt_fname,
                        wav_fname = wav_fname,
                        X = cur_X,
                        Y_test = cur_Y_test,
                        ignore = params['ignore'])
    else:
        print('Found a cached file.')

    print(f'File: #{cur} {txt_fname} ended.')

    return cur_X, cur_Y_test

def generate_training_data(n_bins = 252,
                  y_n_bins = 88,
                  y_bins_per_octave = 12,
                  bins_per_octave = 36,
                  hop_length = 512,
                  sr = 16000,
                  o_sr = 44100,
                  fmin = librosa.note_to_hz('C1'),
                  txts = None,
                  wavs = None,
                  max_files = 10,
                  batch_size = 30,
                  glob_param = 'ISOL/NO',
                  ignore = False,
                  ignore_x = False,
                  ignore_y = False,
                  decay = False,
                  no_result = False):

    if wavs is None or txts is None:
        txts, wavs = get_globs(glob_param)

    if max_files == -1 or max_files is None:
        max_files = len(txts)
    else:
        max_files = min(max_files, len(txts))

    print(f'Left {max_files}')

    X = None
    Y_test = None

    result = None

    TASK = list(zip(txts[:max_files],
                    wavs[:max_files],
                    list(numpy.arange(max_files))))
    context = {
        'n_bins' : locals()['n_bins'],
        'bins_per_octave' : locals()['bins_per_octave'],
        'hop_length' : locals()['hop_length'],
        'sr' : locals()['sr'],
        'o_sr' : locals()['o_sr'],
        'fmin' : locals()['fmin'],
        'txts' : locals()['txts'],
        'wavs' : locals()['wavs'],
        'max_files' : locals()['max_files'],
        'batch_size' : locals()['batch_size'],
        'glob_param' : locals()['glob_param'],
        'ignore' : locals()['ignore'],
        'ignore_x' : locals()['ignore_x'],
        'ignore_y' : locals()['ignore_y'],
        'decay' : locals()['decay'],
        'no_result' : locals()['no_result'],
    }
    TASK = list(map(lambda o: (context, *o), TASK))

    print('Test RUN started.')
    proc_one_file(TASK[0])
    print('Test RUN ended.')

    with multiprocessing.Pool(os.cpu_count() + 1) as p:
        result = p.map(proc_one_file, TASK)

    xs = numpy.vstack(list(map(lambda e: e[0].shape, result)))
    ys = numpy.vstack(list(map(lambda e: e[1].shape, result)))

    L = numpy.max(xs[ : , 0])
    N = xs.shape[0]

    R = N % batch_size

    TX = numpy.zeros((batch_size, L, 1, n_bins))
    TY = numpy.zeros((batch_size, L, y_n_bins))

    X = ()
    Y_test = ()

    offset = 0

    for k in numpy.arange(N):
        if k % batch_size == 0 and k > 0:
            offset -= batch_size

        TX[k + offset, : result[k][0].shape[0] , : , : ] =\
            result[k][0]
        TY[k + offset, : result[k][1].shape[0] , : ] =\
            result[k][1]

        if k == N - 1 or k == batch_size - 1:
            X = (*X, TX.T.reshape(-1, 1, n_bins))
            Y_test = (*Y_test, TY.T.reshape(-1, y_n_bins))

    del result, TX, TY

    return X, Y_test

def dumb_amt_test(model = dnn_amt_model(),
                  n_bins = 252,
                  bins_per_octave = 36,
                  hop_length = 512,
                  sr = 16000,
                  o_sr = 44100,
                  fmin = librosa.note_to_hz('C1'),
                  plot_each = False,
                  epochs = 1,
                  max_files = 10,
                  val_max_files = 3,
                  batch_size = 30,
                  glob_param = 'ISOL/NO',
                  ignore = False,
                  ignore_x = False,
                  ignore_y = False,
                  decay = False,
                  shuffle = False,
                  training_data = None,
                  validate_visual_predict = False,
                  validation_split = 0.0):

    X = None
    Y_test = None

    if training_data is None:
        X, Y_test = generate_training_data(n_bins = n_bins,
                bins_per_octave = bins_per_octave,
                hop_length = hop_length,
                sr = sr,
                o_sr = o_sr,
                fmin = fmin,
                decay = decay,
                max_files = max_files,
                batch_size = batch_size,
                glob_param = glob_param,
                ignore = ignore,
                ignore_x = ignore_x,
                ignore_y = ignore_y)
    else:
        X, Y_test = training_data

    indexes = numpy.arange(len(X))
    if shuffle:
        numpy.random.shuffle(indexes)

    for p in indexes:
        model.fit(X[p], Y_test[p],
                  epochs = epochs,
                  batch_size = batch_size,
                  verbose = 1)

    if validate_visual_predict:
        predict_wav(model = model)

    return model

def predict_wav(wav_fname = None,
                model = None,
                txt_fname = None,
                batch_size = 30,
                x_n_bins = 252,
                x_bins_per_octave = 36,
                y_n_bins = 88,
                y_bins_per_octave = 12,
                decay = False):

    if wav_fname is None and txt_fname is None:
        wav_fname = 'res/organ.wav'
        txt_fname = 'res/organ.txt'

    val_X = process_C(wav_fname = wav_fname)

    y_seq = None

    if txt_fname is not None:
        val_Y_test, y_seq = parse_notes(\
                txt_fname = txt_fname,
                n_bins = x_n_bins,
                bins_per_octave = x_bins_per_octave,
                decay = decay,
                num_frames = val_X.shape[0])

    L = val_X.shape[0] // batch_size * batch_size

    val_X = val_X[ : L, : , : ]
    val_Y_test = val_Y_test[ : L, : ]

    pred_Y = model.predict(val_X,
                           batch_size = batch_size,
                           verbose = 1)

    F = numpy_amt_frame_metrics(val_Y_test, pred_Y)['F']
    print(f'Frame based F-measure { F } \n')

    plot_processing(pred_Y.T,
                    y_seq,
                    bins_per_octave = y_bins_per_octave)

    return val_X, pred_Y, y_seq

def plot_processing(p_y_seq,
        y_seq = None,
        sr = 16000,
        bins_per_octave = 36,
        hop_length = 512):
    if y_seq is not None:
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
    librosa.display.specshow(y_seq,
                             sr = sr, x_axis = 'time', y_axis = 'cqt_note',\
                             bins_per_octave = bins_per_octave,
                             hop_length = hop_length)
    plt.title(title)
    plt.tight_layout()

def preprocess_dataset(glob_param = 'ISOL/NO',
                       max_files = None,
                       ignore_each = False,
                       ignore_each_x = False,
                       ignore_each_y = False,
                       ignore_all = False,
                       wavs = None,
                       txts = None,
                       decay = False,
                       no_result = False,
                       cache_file = 'build/ISOL_NO_ALL.meta.npz'):
    X, Y_test = (None, None)

    if not ignore_all:
        X, Y_test = get_cached_data(cache_file = cache_file)

    if X is None or Y_test is None:
        X, Y_test = generate_training_data(glob_param = glob_param,
                                           max_files = max_files,
                                           ignore = ignore_each,
                                           ignore_x = ignore_each_x,
                                           ignore_y = ignore_each_y,
                                           decay = decay,
                                           txts = txts,
                                           wavs = wavs,
                                           no_result = no_result)
        if not no_result:
            set_cached_data(X = X,
                            Y_test = Y_test,
                            cache_file = cache_file,
                            ignore = ignore_all)

    return X, Y_test

def convert_into_h5py():
    try:
        X1, Y_test1 = preprocess_dataset(\
                glob_param = 'RAND/**',
                cache_file = 'build/RAND_ALL.meta.npz')
        X2, Y_test2 = preprocess_dataset(\
                glob_param = 'ISOL/NO',
                cache_file = 'build/ISOL_NO_ALL.meta.npz')
        X3, Y_test3 = preprocess_dataset(\
                glob_param = 'ISOL/TR1',
                cache_file = 'build/ISOL_TR1_ALL.meta.npz')
        X4, Y_test4 = preprocess_dataset(\
                glob_param = 'ISOL/TR2',
                cache_file = 'build/ISOL_TR2_ALL.meta.npz')

        f = h5py.File('build/maps-storage.hdf5', 'a')

        f['rand_all/X'] = X1
        f['rand_all/Y_test'] = Y_test1
        f['isol_no_all/X'] = X2
        f['isol_no_all/Y_test'] = Y_test2
        f['isol_tr1_all/Y_test'] = Y_test3
        f['isol_tr1_all/X'] = X3
        f['isol_tr2_all/X'] = X4
        f['isol_tr2_all/Y_test'] = Y_test4
    finally:
        del X1, Y_test1, X2, Y_test2, X3, Y_test3, X4, Y_test4

    return f

def load_whole_data():
    f = h5py.File('build/maps-storage.hdf5', 'r')

    X, Y_test = (None, None)

    try:
        xs = list(map(lambda x: f[x]['X'].shape, f.keys()))
        ys = list(map(lambda x: f[x]['Y_test'].shape, f.keys()))

        X = numpy.zeros((numpy.sum(list(map(lambda x: x[0], xs))), 1, 252))
        list(f.values())[0]['X'].read_direct(\
                X, dest_sel = numpy.s_[:xs[0][0], :, :])
        list(f.values())[1]['X'].read_direct(\
                X, dest_sel = numpy.s_[xs[0][0]:xs[0][0] + xs[1][0], :, :])
        list(f.values())[2]['X'].read_direct(\
                X, dest_sel = numpy.s_[xs[0][0] + xs[1][0] :\
                                       xs[0][0] + xs[1][0] + xs[2][0], :, :])
        list(f.values())[3]['X'].read_direct(\
                X, dest_sel = numpy.s_[xs[0][0] + xs[1][0] + xs[2][0] : , :, :])

        Y_test = numpy.zeros((numpy.sum(list(map(lambda x: x[0], ys))), 88))
        list(f.values())[0]['Y_test'].read_direct(\
                Y_test, dest_sel = numpy.s_[:ys[0][0], :])
        list(f.values())[1]['Y_test'].read_direct(\
                Y_test, dest_sel = numpy.s_[ys[0][0] : ys[0][0] + ys[1][0], :])
        list(f.values())[2]['Y_test'].read_direct(\
                Y_test, dest_sel = numpy.s_[ys[0][0] + ys[1][0] :\
                                            ys[0][0] + ys[1][0] + ys[2][0], :])
        list(f.values())[3]['Y_test'].read_direct(\
                Y_test, dest_sel = numpy.s_[ys[0][0] + ys[1][0] + ys[2][0] : , :])
    except:
        del X, Y_test

    return X, Y_test
