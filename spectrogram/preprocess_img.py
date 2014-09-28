from PIL import Image
import os

standard_size = (28,28)

def preprocess_img(filename,name):
    print name
    img = Image.open(filename)
    img = img.convert('L')
    #img = img.resize(standard_size)
    img.save('./3sec_50x20_gs/{0}-prep'.format(name[0:-16]),'png')
 
def explore(path):
    n = 0
    for filename in os.listdir(path):
        preprocess_img(path + '/' +filename, filename)
        n +=1
        print n

if __name__ == '__main__':
    explore('./3sec_50x20')
