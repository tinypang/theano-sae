import matplotlib.pyplot as plt
import numpy as np
import sunau
import sys
import matplotlib as mpl
import wave
from scipy.io.wavfile import read
from scikits.audiolab import auread

#mpl.rcParams['agg.path.chunksize']= 20000

au = sunau.open('blues.00002.au','r')

ausignal = au.readframes(-1)
ausignal = np.fromstring(ausignal, 'Int16')

scausignal, scaufs, enc = auread('./blues.00002.au')

aufs = au.getframerate()
auTime=np.linspace(0, len(ausignal)/aufs, num=len(ausignal))

scauTime = np.linspace(0, len(scausignal)/scaufs, num=len(scausignal))

plt.subplot(3,1,1)
plt.plot(auTime,ausignal)
plt.xlabel('AU file with SUNAU module')

plt.subplot(3,1,2)
plt.plot(scauTime,scausignal)
plt.xlabel('AU file with SCIKITS AU module')

plt.subplot(3,1,3)
Pxx, freqs, bins, im = plt.specgram(scausignal, NFFT=32, Fs=scaufs, noverlap=10)
plt.xlabel('AU Spectrogram with SCIKITS AU module')

plt.show()
