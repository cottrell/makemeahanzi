import glob
import tempfile
import shutil
import bcolz
import numpy as np
import multiprocessing
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import PIL
from keras.layers import Input, Dense
from keras.models import Model, load_model

_script_dir = os.path.dirname(os.path.realpath(__file__))

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

class ModelWriter():
    # make a class to be careful to not lose long running files, ugh. could just not delete
    def __init__(self, filename):
        self.cleanup_on_success = True
        self.filename = filename
    def exists(self):
        return os.path.exists(self.filename)
    def load(self):
        return load_model(self.filename)
    def save(self, model):
        temp = tempfile.mktemp(prefix=self.filename + '.tmp.')
        try:
            if os.path.exists(self.filename):
                shutil.move(self.filename, temp)
            print('writing {}'.format(self.filename))
            model.save(self.filename)
        except Exception as e:
            print('caught exception, moving file {} back to {}'.format(temp, self.filename))
            shutil.move(temp, self.filename)
            raise e
        if self.cleanup_on_success:
            os.remove(temp)

def ae():
    data = load()
    data = data[:]
    input_dim = data.shape[1] * data.shape[2]

    params = dict(
            input_dim=input_dim,
            encoder=[
                (512, 'relu'),
                (256, 'relu'),
                (128, 'relu'),
                (64, 'relu'),
                (32, 'relu'),
                ],
            decoder=[
                (32, 'relu'),
                (64, 'relu'),
                (128, 'relu'),
                (256, 'relu'),
                (512, 'relu'),
                (input_dim, 'sigmoid')
                ]
            )
    encoding_dim = params['encoder'][-1][0] # 32 floats -> compression of factor 1024 / 32 = 32, assuming the input is input_dim floats

    mpath = os.path.join(_script_dir, 'models')
    if not os.path.exists(mpath):
        os.makedirs(mpath)
    fbase = os.path.join(mpath, _make_str(params))

    input_img = Input(shape=(input_dim,))
    encoded = input_img
    modelwriter = ModelWriter("{}_{}.h5".format(fbase, 'autoencoder'))
    if modelwriter.exists():
        autoencoder = modelwriter.load()
    else:
        print('fresh autoencoder, will save to {}'.format(modelwriter.filename))
        for s, t in params['encoder']:
            encoded = Dense(s, activation=t)(encoded)
        decoded = encoded
        for s, t in params['decoder']:
            decoded = Dense(s, activation=t)(decoded)

        autoencoder = Model(input_img, decoded)
        modelwriter.save(autoencoder)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # dunno, can we do this here and recompile as necessary?

    # https://stackoverflow.com/questions/44472693/how-to-decode-encoded-data-from-deep-autoencoder-in-keras-unclarity-in-tutorial
    # just pull the layers from the autoencoder so it works in both the load preload case
    enco = input_img
    for i in range(len(params['encoder'])):
        enco = autoencoder.layers[i+1](enco)
    encoder = Model(input_img, enco)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    deco = encoded_input
    for i in range(-len(params['decoder']), 0):
        deco = autoencoder.layers[i](deco)
    decoder = Model(encoded_input, deco)

    import numpy as np
    import sklearn.model_selection as ms
    x_train, x_test = ms.train_test_split(data)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    autoencoder.fit(x_train, x_train, epochs=1000, batch_size=256 * 4, shuffle=True, validation_data=(x_test, x_test))
    modelwriter.save(autoencoder)

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


