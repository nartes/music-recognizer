import keras
import keras.models
import keras.layers
import keras.optimizers
import keras.utils
import numpy

def gen_sin_train_and_test(NOISE_AMPL = 0.5 * 1e-2,\
                           ORDER = 2, AHEAD = 1, N = 4,
                           SL = 1000, L = 10):
    S = numpy.sin(1.0 * numpy.arange(SL) / N * 2 * numpy.pi) +\
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
        DW = 32
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

    model.compile(loss = 'mse',\
                  optimizer = 'rmsprop',\
                  metrics = ['mse'])

    model.summary()

    return model

def test1():
    nb_classes = 3
    batch_size = 64
    nb_epoch = 1
    X_train, Y_train = gen_sin_train_and_test(L = 60000, NOISE_AMPL = 5e-2)
    X_test, Y_test = gen_sin_train_and_test(L = 100)
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

def test2():
    batch_size = 64
    nb_epoch = 1
    X_train, Y_train = gen_sin_train_and_test(L = 60000, NOISE_AMPL = 5e-2)
    X_test, Y_test = gen_sin_train_and_test(L = 100)
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

    return model
