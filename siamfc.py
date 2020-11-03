"""
Created on Mon Aug  5 10:26:39 2019

@author: Md. Maklachur Rahman
"""

from __future__ import absolute_import, division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from got10k.trackers import Tracker


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)

class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()

        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))      
        
        self.feature2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        
        self.feature3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        
        self.feature4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        
        self.feature5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca_fc1 = nn.Conv2d(512,  128, 1)
        self.ca_relu1 = nn.ReLU(inplace=True)
        self.ca_fc2 = nn.Conv2d(128, 512, 1)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()
        
        self._initialize_weights()

    def template(self, z):
        z = self.feature1(z)
        z = self.feature2(z)
        z = self.feature3(z)
        z = self.feature4(z)
        layer = self.feature5(z)
   
        layer_saved = layer
        # Channel attention module
        module_input = layer
        avg = self.avg_pool(layer)
        mx = self.max_pool(layer)
        avg = self.ca_fc1(avg)
        mx = self.ca_fc1(mx)
        avg = self.ca_relu1(avg)
        mx = self.ca_relu1(mx)
        avg = self.ca_fc2(avg)
        mx = self.ca_fc2(mx)
        layer = avg + mx
        layer = self.sigmoid_channel(layer)
        
        # Spatial attention module
        layer = module_input * layer
        module_input = layer 
        avg = torch.mean(layer, 1, True)
        mx, _ = torch.max(layer, 1, True)
        layer = torch.cat((avg, mx), 1)
        layer = self.conv_after_concat(layer)
        layer = self.sigmoid_spatial(layer)
        layer = module_input * layer
        
        layer = torch.add(layer_saved, layer)
        return layer
    
    def Search(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.feature5(x)
        return x

    
    def forward(self, z, x):
        z = self.template(z)
        x = self.Search(x)

        # define fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # scale adjustment for the response maps
        out = 0.001 * out + 0.0

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__(
            name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)

        # setup GPU device if available or not
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # model setup
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # optimizer setup 
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # lr scheduler setup
        self.lr_scheduler = ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_decay)

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 135,
            'instance_sz': 263,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.27,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
			
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def init(self, image, box):
        image = np.asarray(image)
         
        self.frame = 2 #for adding frame number on image

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # target and search image sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # target image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)
 
        # target features
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()         
            self.kernel = self.net.template(exemplar_image)

    def update(self, image):
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()          
            
            instances = self.net.Search(instance_images)
            responses = F.conv2d(instances, self.kernel) * 0.001
        responses = responses.squeeze(1).cpu().numpy()
        #print('\nInitial_responses_shape2 =',responses.shape)

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        #print('\nFinal_responses_shape2 =',responses.shape)
        mx_val = np.amax(responses, axis=(1, 2))
        #print('\nFinal_responses_mx_val =',mx_val)  

        # peak scale
        #scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        scale_id = np.argmax(mx_val)
        #print('\nresponse scale_id =',scale_id)

        # calculate the peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # target center locating 
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # target size updating
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            self.net.train()
        else:
            self.net.eval()

        z = batch[0].to(self.device)
        x = batch[1].to(self.device)

        with torch.set_grad_enabled(backward):
            responses = self.net(z, x)
            labels, weights = self._create_labels(responses.size())

            loss = F.binary_cross_entropy_with_logits(
                responses, labels, weight=weights, reduction='mean')

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if update_lr:
                    self.lr_scheduler.step()
            pytorch_trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            print('trainable_params= ', pytorch_trainable_params)
        return loss.item()

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # apply padding when it necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # cropping the image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights

