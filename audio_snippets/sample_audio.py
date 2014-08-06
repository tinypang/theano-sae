import sunau
import random
import glob
import os

def snippet(filepath,filename,n):
    full = sunau.open(filepath,'r')   #open full length file
    sample = sunau.open('10secSnippets/sample-{0}'.format(filename), 'w') #open sample au file to write to
    fr = full.getframerate()    #get frame rate of original file
    nsecframes = n*fr   #get number of frames needed for n sec sample
    sample.setparams([full.getnchannels(),full.getsampwidth(),fr,nsecframes,full.getcomptype(),full.getcompname()]) #set audio paramters for sample
    start = random.randrange(0,full.getnframes()-nsecframes,1)  #define random start point such that n sec of sample audio can be found from start point
    full.setpos(start)  #set audio reader start point to new start point
    sample.writeframes(full.readframes(nsecframes)) #write n sec of frames from original file to sample file
    full.close()    #close original file
    sample.close()  #close sample file

def explore(path):
    sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, cut a random 10sec snippet
            snippet(path + '/' + filename, filename,10)
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename)
    



if __name__ == '__main__':
    explore('../gtzan_genre')
