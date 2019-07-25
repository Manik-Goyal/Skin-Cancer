import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
preprocessing_name = "dermatologic"
image_preprocessing_fn = preprocessing_factory.get_preprocessing( preprocessing_name, is_training=10)

eval_image_size = 299

def preprocess(img) :
    return image_preprocessing_fn(img, eval_image_size, eval_image_size,
            bbox=None,
            fast_mode=True,
            area_range=(0.05, 1.0),
            add_rotations=True,
            normalize_per_image=0)
            
img=cv2.imread("melanoma.jpg")
print(img[122,122])
img=img/255.0
print(img[122,122])
img = tf.cast(img, tf.float32)
#print(img)
for r in range(20,25):
    img_aug=preprocess(img)
    file_name='data_augment/'+str(r)+'.jpg'
    print(img_aug)
    img_new=(tf.Session().run(img_aug) )
    
    print(img_new[122,122])
    img_new=img_new*255.0
    print(img_new[122,122])
    #img_new+=1.0
    #img_new/=2.0
    #img = Image.fromarray(img_new, "RGB")
    #print img_new[122,122]
    cv2.imwrite(file_name,img_new)
    #img.save(file_name)
