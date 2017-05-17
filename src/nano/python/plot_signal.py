'''
Generate raw signal plot for each FAST5 file in a directory. Can also run on a
single file. Running on a directory uses multithreading, number of processors
hardcoded with NUM_THREADS.

USAGE
python plot_signal.py [FAST5 FILE]|[DIRECTORY]
'''

import h5py, sys, os, glob, re
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def make_figure(fast5_file):
    # parse out read and channel/pore number
    path_match = re.match(r'.+_ch(\d+)_read(\d+)_strand\d?\.fast5', fast5_file)
    (channel, read) = (path_match.group(1), path_match.group(2))

    hdf = h5py.File(fast5_file,'r')

    raw_signal_path = '/Raw/Reads/Read_'+read+'/Signal'
    if raw_signal_path in hdf:
        raw_signal = hdf[raw_signal_path]
    else:
        print("Couldn't find raw signal")

    read_id = hdf['/Analyses/EventDetection_000/Reads/Read_'+read].attrs['read_id']
    read_id = read_id.decode('UTF-8')

    # save plot
    plt.plot(raw_signal)
    plt.savefig(str(read_id)+'.raw.png')
    plt.close()

    return None

if __name__ == '__main__':

    path = sys.argv[1]

    # for parallel processing of directories
    NUM_THREADS = 8

    if os.path.isdir(path):
        fast5_files = glob.glob(path+'/*.fast5')
        pool = Pool(processes=NUM_THREADS)
        pool.map(make_figure, fast5_files)
    else:
        make_figure(path)
