import sunau
import random
import glob
import os
import math

def snippet(filepath,filename,n):
    full = sunau.open(filepath,'r')   #open full length file
    fr = full.getframerate()    #get frame rate of original file
    nsecframes = n*fr   #get number of frames needed for n sec sample
    nframes = full.getnframes()
    nsecframes = int(math.floor(nframes/10))
    params = [full.getnchannels(),full.getsampwidth(),fr,nsecframes,full.getcomptype(),full.getcompname()]
    i = 0
    for j in range(0,nframes,nsecframes):   #segment full clip into length/n clips
        if nframes - j < nsecframes:
            full.close() #close original file
            return
        else:
            sample = sunau.open('./GTZAN_3sec/{0}-{1}.au'.format(filename[0:-3],i), 'w') #open sample au file to write to
            sample.setparams(params) #set audio paramters for sample
            full.setpos(j)  #set audio reader start point to new start point
            sample.writeframes(full.readframes(nsecframes)) #write n sec of frames from original file to sample file
            sample.close()  #close sample file
            i += 1

def explore(path):
    sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, cut a random 10sec snippet
            print filename
            snippet(path + '/' + filename, filename,3)
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename)


if __name__ == '__main__':
    explore('../GTZAN_genre')
