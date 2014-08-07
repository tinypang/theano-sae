import Image
import os

standard_size = (380,240)

def preprocess_img(filename,name):
    img = Image.open(filename)
    img = img.convert('L')
    img = img.resize(standard_size)
    img.save('./preprocessed/{0}-prep'.format(name), 'png')
    
def explore(path):
    for filename in os.listdir(path):
        preprocess_img(path + '/' +filename, filename)

if __name__ == '__main__':
    explore('./SampleSpectrograms')
