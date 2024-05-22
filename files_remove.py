import os
import glob

folder = '/home/gmhelm/repo/gaussian-mesh-splatting/data/person_1/train/*'

files = glob.glob(folder)
for f in files:
    number = f[-8:-7]
    if "0" not in number:
        os.remove(f)