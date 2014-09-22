from PIL import Image
import os

standard_size = (57,36)

def preprocess_img(filename,name):
    print name
    img = Image.open(filename)
    img = img.convert('L')
    img = img.resize(standard_size)
    img.save('./gs_3sec_spectrograms_33rd/{0}-prep'.format(name[0:-11]),'png')
 
def explore(path):
    for filename in os.listdir(path):
        preprocess_img(path + '/' +filename, filename)

if __name__ == '__main__':
    explore('./3sec_spectrograms')
