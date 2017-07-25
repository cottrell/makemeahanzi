import glob
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

def _makearray(f, newsize=32):
    print(f)
    im = PIL.Image.open(f)
    im = im.resize((newsize, newsize))
    return np.fromstring(im.tobytes(), dtype=np.uint8)

def make_np_data(newsize=32):
    files = glob.glob('png/*.png')
    pool = multiprocessing.Pool()
    res = list()
    for i, f in enumerate(files):
        res.append(pool.apply_async(_makearray, (f,), dict(newsize=newsize)))
    pool.close()
    pool.join()
    m = len(files)
    n = newsize ** 2
    a = np.empty((m, n), dtype=np.uint8)
    for x in res:
        a[i,:] = x.get()
    filename = 'array_{}.bcolz'.format(newsize)
    print('saving {}'.format(filename))
    bcolz.carray(a, rootdir=filename)
    return a
