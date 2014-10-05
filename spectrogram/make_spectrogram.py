""" Spectrogram image for a given sunau audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""


import os
import glob
import pylab
import matplotlib
import matplotlib.pyplot as plt
from scikits.audiolab import auread,wavread
import numpy as np

n = 0

def graph_spectrogram(audio_file,name):    #function to plot spectrogram figure
    #signal, fs, enc = auread(audio_file) #call function to determine data needed for spectrogram of sound file
    signal, fs, enc = wavread(audio_file) #call function to determine data needed for spectrogram of sound file
    if len(signal[0]) == 2:
        signal = np.array(signal)
        signal = np.mean(signal,axis=1)
        signal = signal.tolist()
    fig=plt.figure(num=None, frameon=False) #create figure to build spectrogram on
    #fig.set_size_inches(0.5,0.2)
    fig.set_size_inches(0.28,0.28)
    ax = plt.Axes(fig, [0., 0., 1.18,1.1])
    ax.set_axis_off()   #hide axis notation
    fig.add_axes(ax)    #add axis size to figure
    #NFFT = int(frame_rate*0.020)  # 20ms window
    #noverlap = int(frame_rate*0.010)   #10ms overlap
    NFFT = 32
    noverlap = 10
    spec = plt.specgram(signal, NFFT = NFFT,Fs=fs,noverlap=noverlap)  #plot spectrogram with audio data
    fig.savefig('./ISMIR_genre/ismirg_3sec_28x28/{0}.png'.format(os.path.splitext(name)[0]),format='png')  #save spectrogram
    plt.close(fig)  #close spectrogram figure

def explore(path):
    global n
    #sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    sounds = glob.glob(path + '/*.wav')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, graph it's spectrogram
            graph_spectrogram(path + '/' + filename, filename)
            n+=1
            print n
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename)

if __name__ == '__main__':
    #explore('./test')
    explore('../audio_snippets/ISMIR_genre_training_3sec')

