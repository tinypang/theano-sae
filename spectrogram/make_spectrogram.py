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
from scikits.audiolab import auread

num_frames={}

def graph_spectrogram(au_file,name):    #function to plot spectrogram figure
    signal, fs, enc = auread(au_file) #call function to determine data needed for spectrogram of sound file
    fig=plt.figure(num=None, frameon=False) #create figure to build spectrogram on
    fig.set_size_inches(0.5,0.2)
    '''if nframes == 661504:   #set axis size dependant on number of frames in audio to avoid whitespace in final spectrogram
        ax = plt.Axes(fig, [0., 0., 1.,1.09])
    else:
        ax = plt.Axes(fig, [0., 0., 1.18, 1.09])
    '''
    ax = plt.Axes(fig, [0., 0., 1.18,1.1])
    ax.set_axis_off()   #hide axis notation
    fig.add_axes(ax)    #add axis size to figure
    #NFFT = int(frame_rate*0.020)  # 20ms window
    #noverlap = int(frame_rate*0.010)   #10ms overlap
    NFFT = 32
    noverlap = 10
    spec = plt.specgram(signal, NFFT = NFFT,Fs=fs,noverlap=noverlap)  #plot spectrogram with audio data
    fig.savefig('./3sec_50x20/{0}-raw-spectrogram.png'.format(os.path.splitext(name)[0]),format='png')  #save spectrogram
    print name
    plt.close(fig)  #close spectrogram figure

def explore(path):
    sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, graph it's spectrogram
            graph_spectrogram(path + '/' + filename, filename)
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename)

if __name__ == '__main__':
    #explore('./test')
    explore('../audio_snippets/GTZAN_3sec')

