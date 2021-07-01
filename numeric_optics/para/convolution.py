from numeric_optics.initialize import glorot_uniform
from numeric_optics.para import Para, ParaInit, to_para, to_para_init
from numeric_optics.lens import convolution

def multicorrelate(shape):
    return ParaInit(lambda: glorot_uniform(shape), Para(convolution.multicorrelate))

def correlate_2d(kernel_shape, input_channels, output_channels):
    shape = (output_channels,) + kernel_shape + (input_channels,)
    return ParaInit(lambda: glorot_uniform(shape), Para(convolution.multicorrelate))

def max_pool_2d(kh, kw):
    return to_para_init(convolution.max_pool_2d(kh, kw))

flatten = to_para_init(convolution.flatten)
