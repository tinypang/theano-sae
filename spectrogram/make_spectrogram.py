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

num_frames={}

def graph_spectrogram(au_file,name):    #function to plot spectrogram figure
    sound_info, frame_rate, nframes = get_wav_info(au_file) #call function to determine data needed for spectrogram of sound file
    fig=plt.figure(num=None, frameon=False) #create figure to build spectrogram on
    fig.set_size_inches(19,12)
    #plt.subplot(111)
    #plt.title('spectrogram of %r' % au_file)
    if nframes == 661504:   #set axis size dependant on number of frames in audio to avoid whitespace in final spectrogram
        ax = plt.Axes(fig, [0., 0., 1.,1.09])
    else:
        ax = plt.Axes(fig, [0., 0., 1.18, 1.09])
    ax.set_axis_off()   #hide axis notation
    fig.add_axes(ax)    #add axis size to figure
    NFFT = int(frame_rate*0.020)  # 20ms window
    noverlap = int(frame_rate*0.010)   #10ms overlap
    spec = plt.specgram(sound_info, NFFT = NFFT,Fs=frame_rate,noverlap=noverlap)  #plot spectrogram with audio data
    fig.savefig('./3sec_spectrograms/{0}-raw-spectrogram-{1}.png'.format(os.path.splitext(name)[0],nframes),format='png')  #save spectrogram
    print name
    plt.close(fig)  #close spectrogram figure

def get_wav_info(au_file):  #function to determine audio data of audio files
    au = sunau.open(au_file, 'r')   #open file with python's au module
    frames = au.readframes(-1)  #get data from last frame of audio file
    sound_info = pylab.fromstring(frames, 'Int16')  #convert frame data to usable data type
    frame_rate = au.getframerate()  #get framerate of audio file
    nframes = au.getnframes()   #get number of frames in audio file
    au.close()  #close audio file
    return sound_info, frame_rate, nframes  #return audio data, frame rate and number of frames

def explore(path):
    sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, graph it's spectrogram
            '''
            au_file = path + '/' + filename
            au = sunau.open(au_file, 'r')
            frames = au.getnframes()
            if frames in num_frames:
                num_frames[frames] = num_frames[frames]+1
            else:
                num_frames[frames] = 1 
            '''
            graph_spectrogram(path + '/' + filename, filename)
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename)
'''
    def main():
    explore('../gtzan_genre')
    keylist = num_frames.keys()
    keylist.sort()
    for i in keylist:
        print i, num_frames[i]
    return
'''

if __name__ == '__main__':
    #explore('./test')
    explore('../audio_snippets/GTZAN_3sec')

