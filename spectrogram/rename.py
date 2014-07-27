'''this python script renames all files of the form abc.001.au to abc001.au
'''
import os
import fnmatch
import re

def rename(path):
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, '*.au'):
            newname = re.sub('(?P<genre>^[a-z]*)\.','\g<genre>', filename,1)
            os.rename(path + '/' +filename,path + '/' + str(newname))
        else:
            rename(path + '/' + filename)


if __name__ == '__main__':
    rename('../gtzan_genre')
