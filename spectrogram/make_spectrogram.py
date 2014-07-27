""" Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""


import os
import sunau
import glob
import pylab
import matplotlib
import matplotlib.pyplot as plt

def graph_spectrogram(au_file,name):
    sound_info, frame_rate = get_wav_info(au_file)
    fig=plt.figure(num=None, figsize=(19, 12))
    plt.subplot(111)
    plt.title('spectrogram of %r' % au_file)
    spec = plt.specgram(sound_info, Fs=frame_rate)
    fig.savefig('Spectrograms/spectrogram-%s.png' % os.path.splitext(name)[0],format='png')
    plt.close(fig)

def get_wav_info(au_file):
    au = sunau.open(au_file, 'r')
    frames = au.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = au.getframerate()
    au.close()
    return sound_info, frame_rate

def explore(path):
    sounds = glob.glob(path + '/*.au')
    for filename in os.listdir(path):
        if sounds.count(path + '/' + filename) != 0:
            graph_spectrogram(path + '/' + filename, filename)
        else:
            explore(path + '/' + filename)
        

if __name__ == '__main__':
    explore('../gtzan_genre')
    
