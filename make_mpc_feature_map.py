from import_dataset import import_dataset
from utils import tile_raster_images
import numpy
import PIL.Image


data, labels = import_dataset('./audio_snippets/ismir_test','mpc','ismir',nceps=33,minmax=True)
tile = data.shape[0]
img = data.shape[1]
if img == 594:
    dim = (18,33)
elif img == 784:
    dim = (28,28)
else:
    h = 20
    w = img/h
    dim = (h,w)
for i in data[0]:
    print i
image = PIL.Image.fromarray(tile_raster_images(X=data,img_shape=dim, tile_shape=(int(tile/10),10),tile_spacing=(1,1)))
image.save('./feature_maps/ismirg_3sec_detail_mpc/raw/raw_feature_map.png')


