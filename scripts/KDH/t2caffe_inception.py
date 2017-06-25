import caffe
import sys
import array
import numpy as np
import os
import lutorpy as lua
import m2list
import modules2net
require('nn')
require('cudnn')
require('cutorch')

net = caffe.Net('models/InceptionV3/VOC0712/SSD_300x300/train.prototxt', 'models/InceptionV3/VOC0712/SSD_300x300/InceptionV3_VOC0712_SSD_300x300_iter_80000.caffemodel', caffe.TRAIN)
model = torch.load('/home/kangdongh/mymodels/inceptionV3/inceptionv3.net')
modules = model.listModules(model)

bnmodules = []
convmodules = []

for i in xrange(0,1000):
    if modules[i] == None:
        break
    tp = torch.type(modules[i])
    if tp == 'cudnn.SpatialConvolution' or tp == 'nn.SpatialDilatedConvolution':
        convmodules.append(modules[i])

cl, bl, sl = m2list.InceptionV3Body()
modules2net.modules2net(net, cl, [], convmodules, [])

