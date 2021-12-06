import pickle
import gzip
import numpy as np
import tensorflow as tf
fname = '../Replays/Freeway/5/replay_logs/$store$_terminal_ckpt.39.gz'
with tf.io.gfile.GFile(fname, 'rb') as f:
    with gzip.GzipFile(fileobj=f) as infile:
        x = np.load(infile)

print(len(x))