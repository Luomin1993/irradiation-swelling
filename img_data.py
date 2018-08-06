#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest');


def make_data_for_one_class(img,class_name,dir):
    """
    make many images data by one image from one class;

    ------------------
    @param img        :               the input image;
    @param class_name : the class for the input image;
    @dir              :             the dir for saved;
    """
    # this is a Numpy array with shape (3, 32, 32);
    x = img_to_array(img);
    # this is a Numpy array with shape (1, 3, 32, 32);
    x = x.reshape((1,) + x.shape);
    # the .flow() command below generates batches of randomly transformed images;
    # and saves the results to the `preview/` directory;
    i = 0
    for batch in datagen.flow(x,batch_size=1,save_to_dir=dir,save_prefix=class_name,save_format='jpg'):
        i += 1;
        if i > 10:
            # stop;
            break;

def processing_image_data():
    """
    to do processing_image_data;
    -----------------
    """    
    for head in ['A','B','C','D']:
        for num in range(1,6):
            img_name = './images/' + head + '_' + str(num) + '.jpg';
            img      = load_img(img_name);
            img      = img.resize((32,32));
            make_data_for_one_class(img,head,'./img_data');

def make_Xy():
    """
    to trans the images in the img_data to Xy;
    """
    img_names = os.listdir('./img_data');
    X = [];
    y = [];
    for img_name in img_names:




if __name__ == '__main__':
    #processing_image_data();
    make_Xy();