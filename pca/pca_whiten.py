from scipy import misc
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca(path):
    for filename in os.listdir(path):
        imgfile = path + '/' + filename
        img = misc.imread(imgfile)
        img_mean = np.mean(img, axis=0)
        norm_img = img - img_mean
        pca = PCA(n_components='mle',whiten=True)
        norm_img_arr = norm_img.shape[0:1]
        img_pca = pca.fit(norm_img_arr)
        #img_pca, pca_shape = pca.transform(norm_img)
        plt.show(img_pca)
       











if __name__ == '__main__':
    pca('./test')
    #pca('../spectrogram/Spectrograms')
