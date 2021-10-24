import os
import glob
import ffmpeg


for file in glob.glob("data/SUBESCO/*.wav"):
    basename = os.path.basename(file)
    #install ffmpeg package on local machine
    os.system(f"ffmpeg -i data/SUBESCO/{basename} -ac 1 -ar 16000 data/GUBAS/{basename}")