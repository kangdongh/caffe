import caffe
import sys
import array
import numpy as np
import os
import lutorpy as lua
import m2list

def modules2params(net, convparams, scaleparams, convmodules, bnmodules): 
    for i in xrange(0,len(convparams)):
        param = net.params[convparams[i]]
        weight = convmodules[i].weight.asNumpyArray()
        print param[0].data.shape
        print weight.shape
        param[0].data.flat = weight.flat
        if len(param) == 2:
            bias = convmodules[i].bias.asNumpyArray()
            param[1].data.flat = bias.flat

    for i in xrange(0, len(scaleparams)):
        scparam = net.params[scaleparams[i]]
        weight = bnmodules[i].weight.asNumpyArray()
        scparam[0].data.flat = weight.flat
        bias = bnmodules[i].bias.asNumpyArray()
        scparam[1].data.flat = bias.flat
        """
        bnparam = net.params[bnparams[i]]
        mean = bnmodules[i].running_mean.asNumpyArray()
        bnparam[0].data.flat = mean.flat
        var = bnmodules[i].running_var.asNumpyArray()
        bnparam[1].data.flat = var.flat
        """
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
modules2params(net, cl, [], convmodules, [])


