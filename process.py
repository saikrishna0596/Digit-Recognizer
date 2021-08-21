import pandas as pd
import numpy as np
import os
from helpers import *



class Image_process():

    def __init__(self):
        self.config = yaml_to_dict('config.yml')
        self.raw_train_data = load_flat_file(self.config['TRAIN_DATA'])
        self.raw_test_data = load_flat_file(self.config['TEST_DATA'])
        self.cv_perc = self.config['CV_PERCENT']/100
        self.y_var = self.config['Y_VAR']
        self.image_shape = self.config['IMAGE_SHAPE']

        self.raw_train_data['ImageID'] = np.arange(1,len(self.raw_train_data)+1)
        self.raw_train_data.set_index('ImageID', inplace=True)

        self.raw_test_data['ImageID'] = np.arange(1,len(self.raw_test_data)+1)
        self.raw_test_data.set_index('ImageID', inplace=True)


    def format_inputs(self, X, y):
        images = []
        for index,row in X.iterrows():
            images.append(row.values.reshape(self.image_shape+[1]))
        labels = y.values.reshape(len(y),1)
        return np.array(images), labels


    def create_model_inputs(self):
        X_train, X_cv, y_train, y_cv = cv_train_split(self.raw_train_data, self.cv_perc, self.y_var)
        train_images, train_labels = self.format_inputs(X_train, y_train)
        cv_images, cv_labels = self.format_inputs(X_cv, y_cv)
        return train_images, train_labels, cv_images, cv_labels


    


