from numeric_optics.initialize import glorot_uniform
from numeric_optics.para import Para, to_para
from numeric_optics.lens import convolution

def multicorrelate(shape):
    return Para(lambda: glorot_uniform(shape), convolution.multicorrelate)

def correlate_2d(kernel_shape, input_channels, output_channels):
    shape = (output_channels,) + kernel_shape + (input_channels,)
    return Para(lambda: glorot_uniform(shape), convolution.multicorrelate)

def max_pool_2d(kh, kw):
    return to_para(convolution.max_pool_2d(kh, kw))

flatten = to_para(convolution.flatten)
