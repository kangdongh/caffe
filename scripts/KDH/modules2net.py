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
