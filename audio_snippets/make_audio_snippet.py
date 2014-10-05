import sunau
import random
import glob
import os
import math
import wave
from subprocess import check_call
from tempfile import mktemp

def get_genre(path):
    genre = ''
    for i in range(0,4):
        head, tail = os.path.split(path)
        path = head
        genre = tail
    return genre

def snippet(filepath,filename,n):

    wname = mktemp('.wav')
    check_call(['avconv', '-i', filepath,wname])
    print count
    full = wave.open(wname,'r')   #open full length file
    genre = get_genre(filepath)

    #full = sunau.open(filepath,'r')   #open full length file
    fr = full.getframerate()    #get frame rate of original file
    nsecframes = n*fr   #get number of frames needed for n sec sample
    nframes = full.getnframes()
    #nsecframes = int(math.floor(nframes/10))
    params = [full.getnchannels(),full.getsampwidth(),fr,nsecframes,full.getcomptype(),full.getcompname()]
    i = 0
    for j in range(0,nframes,nsecframes):   #segment full clip into length/n clips
        if nframes - j < nsecframes:
            full.close() #close original file
            return
        else:
            sample = sunau.open('./autest/{0}-{1}-{2}.au'.format(genre,filename[0:-4],i), 'w') #open sample wau file to write to
            sample.setparams(params) #set audio paramters for sample
            full.setpos(j)  #set audio reader start point to new start point
            sample.writeframes(full.readframes(nsecframes)) #write n sec of frames from original file to sample file
            sample.close()  #close sample file
            i += 1
    os.unlink(wname)

count = 0

def explore(path):
    global count
    sounds = glob.glob(path + '/*.mp3')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, cut a random 10sec snippet
            print filename
            snippet(path + '/' + filename, filename,3)
            count+=1
            print count
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename)


if __name__ == '__main__':
    explore('../ISMIR_genre/Training/pop')
