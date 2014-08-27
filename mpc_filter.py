import sunau
import numpy as np
import glob
import os
from features import sigproc
import features

def explore(path):
    sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, graph it's spectrogram
            mpc(path + '/' + filename, filename)
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename)

def get_audio_info(filepath):
    wf = sunau.open(filepath, 'r')
    f = wf.readframes(-1)
    if wf.getsampwidth() == 1:
        data = np.fromstring(f, dtype=np.int8)
    elif wf.getsampwidth() == 2:
        data = np.fromstring(f, dtype=np.int16)
    elif wf.getsampwidth() == 4:
        data = np.fromstring(f, dtype=np.int32)
    wdata = np.array(data,dtype='int32')
    return wdata,wf.getframerate()
'''
def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)    
    :param winstep: the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)    
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97. 
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """          
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate)
    pspec = sigproc.powspec(frames,nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    
    fb = get_filterbanks(nfilt,nfft,samplerate)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    return feat,energy

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft/2+1])
    for j in xrange(0,nfilt):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank 
'''
def mpc_filter(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22):
    feat,energy = features.fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
    feat = np.log(feat)
    return feat

def mpc(path,filename):
    signal, rate = (get_audio_info(path))
    mpcfeatures = mpc_filter(signal=signal,samplerate=rate,numcep=592,nfilt=1184)
    print mpcfeatures
    
if __name__ == '__main__':
    explore('./test')
    




