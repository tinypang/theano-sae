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

def graph_spectrogram(au_file,name):
    sound_info, frame_rate, nframes = get_wav_info(au_file)
    fig=plt.figure(num=None, frameon=False)
    fig.set_size_inches(19,12)
    #plt.subplot(111)
    #plt.title('spectrogram of %r' % au_file)
    if nframes == 661504:
        ax = plt.Axes(fig, [0., 0., 1.,1.09])
    else:
        ax = plt.Axes(fig, [0., 0., 1.18, 1.09])
    ax.set_axis_off()
    fig.add_axes(ax)
    spec = plt.specgram(sound_info, Fs=frame_rate)
    print name, nframes
    print spec
    #ig.savefig('Spectrograms/{0}-raw-spectrogram-{1}.png'.format(os.path.splitext(name)[0],nframes),format='png')
    plt.close(fig)

def get_wav_info(au_file):
    au = sunau.open(au_file, 'r')
    frames = au.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = au.getframerate()
    nframes = au.getnframes()
    au.close()
    return sound_info, frame_rate, nframes

def explore(path):
    sounds = glob.glob(path + '/*.au')
    for filename in os.listdir(path):
        if sounds.count(path + '/' + filename) != 0:
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
        else:
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
    #main()
    explore('../pca/test')
    #explore('../gtzan_genre')

