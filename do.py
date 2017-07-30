import glob
import shutil
import bcolz
import numpy as np
import multiprocessing
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import PIL

def _makepng(f):
    r = svg2rlg(f)
    ff = os.path.join('png', os.path.basename(f.replace('.svg', '.png')))
    print('writing {}->{}'.format(f, ff))
    renderPM.drawToFile(r, ff)

def makepng():
    files = glob.glob('svgs/*.svg')
    if not os.path.exists('png'):
        os.makedirs('png')
    pool = multiprocessing.Pool()
    res = list()
    for i, f in enumerate(files):
        res.append(pool.apply_async(_makepng, (f,)))
    pool.close()
    pool.join()
    for x in res:
        x.get()

_default_size = 32

def _makearray(f, newsize=_default_size):
    print(f)
    im = PIL.Image.open(f)
    im = im.resize((newsize, newsize))
    return np.fromstring(im.tobytes(), dtype=np.uint8).reshape((_default_size, _default_size))

def make_np_data(newsize=_default_size):
    """ save to array_{newsize}.bcolz """
    filename = 'array_{}.bcolz'.format(newsize)
    if os.path.exists(filename):
        print('{} exists'.format(filename))
        shutil.rmtree(filename)
    files = glob.glob('png/*.png')
    pool = multiprocessing.Pool()
    res = list()
    for i, f in enumerate(files):
        res.append(pool.apply_async(_makearray, (f,), dict(newsize=newsize)))
    pool.close()
    pool.join()
    m = len(files)
    n = newsize
    a = np.empty((m, n, n), dtype=np.uint8)
    a[:] = 0
    for i, x in enumerate(res):
        a[i,:,:] = x.get()
    print('saving {}'.format(filename))
    return bcolz.carray(a, rootdir=filename, )

def load():
    filename = 'array_{}.bcolz'.format(_default_size)
    a = bcolz.carray(rootdir=filename, mode='r')
    print('{} shape {}'.format(filename, a.shape))
    return a

def _make_str(d):
    # prob should just hash
    g = lambda x: ':'.join(map(str, x))
    f = lambda x: '|'.join(map(g, x))
    return 'input_dim={}&encoder={}&decoder={}'.format(d['input_dim'], f(d['encoder']), f(d['decoder']))

def ae():
    data = load()
    data = data[:]
    from keras.layers import Input, Dense
    from keras.models import Model
    input_dim = data.shape[1] * data.shape[2]
    encoding_dim = 32  # 32 floats -> compression of factor 1024 / 32 = 32, assuming the input is input_dim floats
    input_img = Input(shape=(input_dim,))

    params = dict(
            input_dim=input_dim,
            encoder=[
                (128, 'relu'),
                (32, 'relu')
                ],
            decoder=[
                (64, 'relu'),
                (input_dim, 'sigmoid')
                ]
            )

    encoded = input_img
    for s, t in params['encoder']:
        encoded = Dense(s, activation=t)(encoded)
    # https://stackoverflow.com/questions/44472693/how-to-decode-encoded-data-from-deep-autoencoder-in-keras-unclarity-in-tutorial

    decoded = encoded
    for s, t in params['decoder']:
        decoded = Dense(s, activation=t)(decoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    deco = encoded_input
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    # print(len(params['decoder']))
    # for i in range(-len(params['decoder']), 0):
    #     print('here', i)
    #     deco = autoencoder.layers[-i](deco)
    # create the decoder model
    decoder = Model(encoded_input, deco)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    import numpy as np
    import sklearn.model_selection as ms
    x_train, x_test = ms.train_test_split(data)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    return locals()

def doplot(l):
    globals().update(l)
    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    aa = np.random.permutation(range(x_test.shape[0]))[:n]
    for i in range(n):
        ii = aa[i]
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(_default_size, _default_size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(_default_size, _default_size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


