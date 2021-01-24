import scipy.signal as signal
import scipy.ndimage as ndimage
import numpy as np

from numeric_optics.lens import Lens

################################################################
# Convolution
################################################################

# Convolve a single 2D kernel (k) over a 2D image (a)
# args: (k, a), where k is the filter kernel and a is an image
def convolve2d_fwd(args):
    k, a = args
    return signal.correlate2d(a, k, mode='valid')

# Reverse derivative of 2D convolution.
def convolve2d_rev(args):
    (k, a), dy = args
    dk = signal.correlate2d(a, dy, mode='valid')
    da = np.flip(signal.correlate2d(np.flip(k), dy, boundary='fill', fillvalue=0))
    return (dk, da)

convolve2d = Lens(convolve2d_fwd, convolve2d_rev)

################################################################
# Flattening, Pooling
################################################################

# Flatten an array
# NOTE: this is a "type-level" operation: nothing about the array data is
# changed, so we should have the property that
# >>> x == flatten.rev(x, flatten.fwd(x))
def unflatten(args):
    x, dy = args
    return np.reshape(dy, x.shape)

# The "flatten" lens simply changes the type of the input
flatten = Lens(np.ndarray.flatten, unflatten)

# Max pooling layer with kernel width, height of (kh, kw)
def max_pool_2d_onechannel(kh, kw):
    def max_pool_fwd(m):
        # note: assumes 2D array
        h, w = m.shape
        mh = h // kh
        mw = w // kw
        
        return m[:mh * kh, :mw * kw].reshape(mh, kh, mw, kw).max(axis=(1,3))

    def max_pool_rev(args):
        m, dm = args
        n = max_pool_fwd(m)
        # Expand the size of the output vector so each item is repeated (kh x kw) times.
        n_rep  = n.repeat(kh, axis=0).repeat(kw, axis=1)
        dm_rep = dm.repeat(kh, axis=0).repeat(kw, axis=1)
        result = (m == n_rep) * dm_rep
        check = result.shape == m.shape
        return result

    return Lens(max_pool_fwd, max_pool_rev)

################################################################
# Convolutional layers with channels
# TODO: vectorise the functions in this section
################################################################

# Correlate a 3D kernel volume over a 3D image with the same depth.
# Inputs:
#   k:  (Kw, Kh,  c)
#   a:  ( w,  h,  c)
# Outputs:
#   y:  (w - Kw + 1,  h - Kh + 1)
def correlate_volume_fwd(k, a):
    kw, hw, kc = k.shape
    w,  h,  c  = a.shape
    if c != kc:
        raise ValueError("image channels {} != {} kernel channels".format(c, kc))

    rs = []
    for i in range(0, c):
        ker = k[:, :, i]
        img = a[:, :, i]
        rs.append(convolve2d_fwd((ker, img)))
    return np.sum(rs, axis=0)

# Reverse derivative of 3D correlation
# Inputs:
#   k:  (Kw, Kh,  c)
#   a:  ( w,  h,  c)
#   dy: (w - Kw + 1, h - Kh + 1)
# Outputs:
#   dks: (Kw, Kh, c)
#   das: ( w,  h, c)
def correlate_volume_rev(k, a, dy):
    kw, hw, kc = k.shape
    w,  h,  c  = a.shape
    if c != kc:
        raise ValueError("image channels {} != {} kernel channels".format(c, kc))

    dks = []
    das = []
    for i in range(0, c):
        ker = k[:, :, i]
        img = a[:, :, i]
        dk, da = convolve2d_rev(((ker, img), dy))
        dks.append(dk)
        das.append(da)

    # Move the channel axis to rightmost position
    dks = np.moveaxis(np.array(dks), 0, -1)
    das = np.moveaxis(np.array(das), 0, -1)
    return dks, das

# Correlate multiple 3D volumes over a 3D input image
# Inputs:
#   ks: (NK, Kw, Kh,  c)
#   a:      ( w,  h,  c)
# Outputs:
#   ys: (w - Kw + 1, h - Kh + 1, NK)
def multicorrelate_volume_fwd(ks, a):
    nk, _, _, _ = ks.shape
    acc = []
    # Correlate each filter with the volume "a"
    for i in range(0, nk):
        k = ks[i]
        acc.append(correlate_volume_fwd(k, a))
    return np.moveaxis(np.array(acc), 0, -1)

# Reverse derivative of multicorrelate_volume_fwd
# Inputs:
#   ks: (NK, Kw, Kh,  c)
#   a:      ( w,  h,  c)
#   ys: (w - Kw + 1, h - Kh + 1, NK)
# Outputs:
#   dks: (NK, Kw, Kh, c)
#   da:     ( w,  h,  c)
def multicorrelate_volume_rev(ks, a, dys):
    nk, kh, kw, kc = ks.shape
    w, h, c = a.shape
    yw, yh, yc = dys.shape
    if kc != c:
        raise ValueError("Kernel channels != image channels")
    if nk != yc:
        raise ValueError("Number of kernels != number of output channels")

    dkss = []
    da = np.zeros(a.shape)
    for i in range(0, nk):
        k  = ks[i]
        dy = dys[:, :, i]
        dks, das = correlate_volume_rev(k, a, dy)
        da += das
        dkss.append(dks)
    return np.array(dkss), da

# TODO: Vectorise fully. Currently uses two nested for loops.
multicorrelate = Lens(
    lambda ksa: multicorrelate_volume_fwd(ksa[0], ksa[1]), 
    lambda kay: multicorrelate_volume_rev(kay[0][0], kay[0][1], kay[1]))


# Max Pooling for images with channels
# TODO: vectorise this
def max_pool_2d(kh, kw):
    lens = max_pool_2d_onechannel(kh, kw)

    def max_pool_2d_fwd(m):
        # note: assumes 2D array
        h, w, c = m.shape

        acc = np.zeros((h // kh, w // kw, c))
        for i in range(0, c):
            acc[:, :, i] += lens.fwd(m[:, :, i])

        return acc

    def max_pool_2d_rev(args):
        x, dy = args
        h, w, c = x.shape

        dx = np.zeros(x.shape)
        for i in range(0, c):
            dx[:, :, i] = lens.rev((x[:, :, i], dy[:, :, i]))

        return dx

    return Lens(max_pool_2d_fwd, max_pool_2d_rev)
