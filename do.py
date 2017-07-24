import glob
import multiprocessing
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
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
