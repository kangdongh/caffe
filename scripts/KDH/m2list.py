import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


def ConvBNLayer(conv_list, bn_list, scale_list, out_layer, use_bn, use_scale=True,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias', **bn_param):
  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  conv_list.append(conv_name)
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    bn_list.append(bn_name)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      scale_list.append(sb_name)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      scale_list.append(bias_name)

def ResBody(conv_list, bn_list, scale_list, block_name, use_branch1):
  # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

  conv_prefix = 'res{}_'.format(block_name)
  conv_postfix = ''
  bn_prefix = 'bn{}_'.format(block_name)
  bn_postfix = ''
  scale_prefix = 'scale{}_'.format(block_name)
  scale_postfix = ''
  use_scale = True

  branch_name = 'branch2a'
  ConvBNLayer(conv_list, bn_list, scale_list, branch_name, use_bn=True, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2b'
  ConvBNLayer(conv_list, bn_list, scale_list, branch_name, use_bn=True, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  out_name = '{}{}'.format(conv_prefix, branch_name)

  branch_name = 'branch2c'
  ConvBNLayer(conv_list, bn_list, scale_list, branch_name, use_bn=True, use_scale=use_scale,
      conv_prefix=conv_prefix, conv_postfix=conv_postfix,
      bn_prefix=bn_prefix, bn_postfix=bn_postfix,
      scale_prefix=scale_prefix, scale_postfix=scale_postfix)
  branch2 = '{}{}'.format(conv_prefix, branch_name)

  if use_branch1:
    branch_name = 'branch1'
    ConvBNLayer(conv_list, bn_list, scale_list, branch_name, use_bn=True, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch1 = '{}{}'.format(conv_prefix, branch_name)

def ResNet50Body():
    conv_list = []
    bn_list = []
    scale_list = []
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(conv_list, bn_list, scale_list, 'conv1', use_bn=True,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    ResBody(conv_list, bn_list, scale_list, '2a', use_branch1=True)
    ResBody(conv_list, bn_list, scale_list, '2b', use_branch1=False)
    ResBody(conv_list, bn_list, scale_list, '2c', use_branch1=False)

    ResBody(conv_list, bn_list, scale_list, '3a', use_branch1=True)

    for i in xrange(1, 4):
      block_name = '3b{}'.format(i)
      ResBody(conv_list, bn_list, scale_list, block_name, use_branch1=False)

    ResBody(conv_list, bn_list, scale_list, '4a', use_branch1=True)
    for i in xrange(1, 6):
      block_name = '4b{}'.format(i)
      ResBody(conv_list, bn_list, scale_list, block_name, use_branch1=False)

    ResBody(conv_list, bn_list, scale_list, '5a', use_branch1=True)
    ResBody(conv_list, bn_list, scale_list, '5b', use_branch1=False)
    ResBody(conv_list, bn_list, scale_list, '5c', use_branch1=False)


    return conv_list, bn_list, scale_list


def InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, layer_params, **bn_param):
  use_scale = False
  for param in layer_params:
    tower_layer = '{}/{}'.format(tower_name, param['name'])
    del param['name']
    if 'pool' in tower_layer:
        "hi"
        #do nothing
    else:
      param.update(bn_param)
      ConvBNLayer(conv_list, bn_list, scale_list, tower_layer, use_bn=True,
          use_scale=use_scale, **param)
    from_layer = tower_layer


def InceptionV3Body(**bn_param):
  conv_list = []
  bn_list = []
  scale_list = []
  # scale is fixed to 1, thus we ignore it.
  use_scale = False

  out_layer = 'conv'
  ConvBNLayer(conv_list, bn_list, scale_list, out_layer, use_bn=True, 
      num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_1'
  ConvBNLayer(conv_list, bn_list, scale_list, out_layer, use_bn=True, 
      num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_2'
  ConvBNLayer(conv_list, bn_list, scale_list, out_layer, use_bn=True, 
      num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool'
  from_layer = out_layer

  out_layer = 'conv_3'
  ConvBNLayer(conv_list, bn_list, scale_list, out_layer, use_bn=True, 
      num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'conv_4'
  ConvBNLayer(conv_list, bn_list, scale_list, out_layer, use_bn=True, 
      num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale,
      **bn_param)
  from_layer = out_layer

  out_layer = 'pool_1'
  from_layer = out_layer

  # inceptions with 1x1, 3x3, 5x5 convolutions
  for inception_id in xrange(0, 3):
    if inception_id == 0:
      out_layer = 'mixed'
      tower_2_conv_num_output = 32
    else:
      out_layer = 'mixed_{}'.format(inception_id)
      tower_2_conv_num_output = 64
    tower_name = '{}'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    
    tower_name = '{}/tower'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ], **bn_param)
    
    tower_name = '{}/tower_1'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    
    tower_name = '{}/tower_2'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    
    out_layer = '{}/join'.format(out_layer)
    from_layer = out_layer

  # inceptions with 1x1, 3x3(in sequence) convolutions
  out_layer = 'mixed_3'
  tower_name = '{}'.format(out_layer)
  InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
      dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  
  tower_name = '{}/tower'.format(out_layer)
  InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
      dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
      dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  
  tower_name = '{}'.format(out_layer)
  InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  
  out_layer = '{}/join'.format(out_layer)
  from_layer = out_layer

  # inceptions with 1x1, 7x1, 1x7 convolutions
  for inception_id in xrange(4, 8):
    if inception_id == 4:
      num_output = 128
    elif inception_id == 5 or inception_id == 6:
      num_output = 160
    elif inception_id == 7:
      num_output = 192
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    
    tower_name = '{}/tower'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ], **bn_param)
    
    tower_name = '{}/tower_1'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ], **bn_param)
    
    tower_name = '{}/tower_2'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    
    out_layer = '{}/join'.format(out_layer)
    from_layer = out_layer

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  out_layer = 'mixed_8'
  towers = []
  tower_name = '{}/tower'.format(out_layer)
  InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  
  tower_name = '{}/tower_1'.format(out_layer)
  InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
      dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
      dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  
  tower_name = '{}'.format(out_layer)
  InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ], **bn_param)
  
  out_layer = '{}/join'.format(out_layer)
  from_layer = out_layer

  for inception_id in xrange(9, 11):
    num_output = 384
    num_output2 = 448
    if inception_id == 9:
      pool = P.Pooling.AVE
    else:
      pool = P.Pooling.MAX
    out_layer = 'mixed_{}'.format(inception_id)
    tower_name = '{}'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    

    tower_name = '{}/tower'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    subtower_name = '{}/mixed'.format(tower_name)
    InceptionTower(conv_list, bn_list, scale_list, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    
    InceptionTower(conv_list, bn_list, scale_list, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    
    tower_name = '{}/tower_1'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ], **bn_param)
    subtower_name = '{}/mixed'.format(tower_name)
    InceptionTower(conv_list, bn_list, scale_list, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ], **bn_param)
    
    InceptionTower(conv_list, bn_list, scale_list, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ], **bn_param)
    

    tower_name = '{}/tower_2'.format(out_layer)
    InceptionTower(conv_list, bn_list, scale_list, from_layer, tower_name, [
        dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ], **bn_param)
    
    out_layer = '{}/join'.format(out_layer)
    from_layer = out_layer
    return conv_list, bn_list, scale_list

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(conv_list, bn_list, scale_list, inter_name, use_bn=use_batchnorm,  lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(conv_list, bn_list, scale_list, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(conv_list, bn_list, scale_list, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(conv_list, bn_list, scale_list, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers
