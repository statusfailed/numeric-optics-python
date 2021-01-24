from numeric_optics.initialize import glorot_uniform
from numeric_optics.para import Para, to_para
from numeric_optics.lens import convolution

def multicorrelate(shape):
    return Para(lambda: glorot_uniform(shape), convolution.multicorrelate)

def max_pool_3d(kh, kw):
    return to_para(convolution.max_pool_3d(kh, kw))

flatten = to_para(convolution.flatten)
