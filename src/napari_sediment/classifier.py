from torchvision import models
from torch import nn
from collections import OrderedDict
import torch
import numpy as np
import skimage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class Classifier():
    """Class to segment image using pixel classification based on VGG16 features.
    
    Parameters
    ----------
    model: torch.nn.Sequential
        model to use for classification
    data: np.ndarray
        image to segment with shape 3 x n_rows x n_cols
    scalings: list of int
        list of scalings to use for multiscale features
    n_filters: int
        number of filters in the first layer of the model
    
    """

    def __init__(self, data=None, annotations=None, scalings=[1,2,4], n_filters=64):

        self.model = None
        self.data = data
        self.annotations = annotations
        self.scalings = scalings
        self.n_filters = n_filters
        self.all_scales = None
        self.all_pixels = None

    def load_model(self, model_type='vgg16'):

        if model_type == 'vgg16':
            vgg16 = models.vgg16(pretrained=True)

            self.model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3,64,3,1,1))]))

            pretrained_dict = vgg16.state_dict()
            reduced_dict = self.model.state_dict()
            reduced_dict['conv1.weight'] = pretrained_dict['features.0.weight']
            reduced_dict['conv1.bias'] = pretrained_dict['features.0.bias']

        elif model_type == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            self.model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3,64,7,2,3, bias=False))]))

            pretrained_dict = resnet.state_dict()
            reduced_dict = self.model.state_dict()
            reduced_dict['conv1.weight'] = pretrained_dict['conv1.weight']

        self.model.load_state_dict(reduced_dict)

    def compute_multiscale_features(self):

        all_scales=[]
        for s in self.scalings:
            im_tot = self.data[:, ::s,::s].astype(np.float32)
            im_torch = torch.from_numpy(im_tot[np.newaxis, ::])
            out = self.model.forward(im_torch)
            out_np = out.detach().numpy()[0]
            #if s > 1:
            out_np = np.stack(
                [skimage.transform.resize(
                    x, (self.data.shape[1], self.data.shape[2]), preserve_range=True)
                    for x in out_np
                ],
                axis=0)
            all_scales.append(out_np)

        self.all_scales = np.concatenate(all_scales, axis=0)
        self.all_pixels = pd.DataFrame(
            np.reshape(
                self.all_scales,
                (len(self.scalings)*self.n_filters, self.data.shape[1]*self.data.shape[2])
            ).T)

    def extract_annotated_features(self):

        # replicate annotations across n_features
        full_annotation = self.annotations > 0

        all_values = self.all_scales[:, full_annotation]

        features = pd.DataFrame(all_values.T)
        target_im = self.annotations[self.annotations > 0]
        targets = pd.Series(target_im)

        self.features = features
        self.targets = targets

    def train_model(self):
        
        # train model
        #split train/test
        X, X_test, y, y_test = train_test_split(self.features, self.targets, 
                                            test_size = 0.2, 
                                            random_state = 42)
        
        #train a random forest classififer
        self.random_forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.random_forest.fit(X, y)

    def predict(self):
            
        # predict
        predictions = self.random_forest.predict(self.all_pixels)
        predicted_image = np.reshape(predictions, [self.data.shape[1], self.data.shape[2]])

        return predicted_image

            