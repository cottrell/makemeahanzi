import glob
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
def makepng():
    files = glob.glob('svgs/*.svg')
    if not os.path.exists('png'):
        os.makedirs('png')
    for i, f in enumerate(files):
        r = svg2rlg(f)
        ff = os.path.join('png', os.path.basename(f.replace('.svg', '.png')))
        print(i, f, ff)
        renderPM.drawToFile(r, ff)
