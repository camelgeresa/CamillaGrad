import numpy as np
from ..tensor import Tensor

def Im2Col(input_image, kernel_h, kernel_w, stride=1, pad=0):
    '''
    input_image: Tensor of shape [BS x C x H x W]
    '''
    N,C,H,W = input_image.shape

    assert (H + 2*pad - kernel_h) % stride == 0
    assert (W + 2*pad - kernel_w) % stride == 0

    out_h = (H + 2*pad - kernel_h)//stride + 1
    out_w = (H + 2*pad - kernel_w)//stride + 1

    # pad should be a tuple specifying how much you want to pad height and width
    padded_data = input_image.pad(pad)

    col = Tensor(np.zeros((N,C, kernel_h, kernel_w, out_h, out_w)))

    for y in range(kernel_h):
        y_max = y + stride*out_h # y_max is the max idx the kernel sees. 
        # Why? because stride is the amount we move the slide the kernel for each convolution & out_h is the num of convs we perform along y.

        for x in range(kernel_w):
            x_max = x + stride*out_w

            # We extract the num of entries each part of the kernel touches.
            # E.g a 2x2 kernel -> what entries of the image does k_1,1 see?
            # Each extraction will be of shape [N x C x out_h x out_w] because out_h x out_W is the num of convs performed.
            col[:,:,y,x] = padded_data[:,:,y:y_max:stride, x:x_max:stride]

    return col.transpose((0,4,5,1,2,3)).reshape((N*out_h*out_w, -1))  # convs x patches
    # We want [N*out_h*out_w, num entries ] -> the output images for each image in batch x filter size * C (num of entries)


def Col2Im(col, img_shape, kernel_h, kernel_w, stride = 1, pad = 0):

    N, C, H, W = img_shape

    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    col= col.reshape((N,out_h, out_w, C, kernel_h, kernel_w)).transpose((0,3,4,5,1,2))

    image = Tensor(np.zeros((N,C,H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))) # ???
    # we want to get the padded matrix, but why + stride?

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            # multiple img entries can be touched by a kernel (e.g when stride = 1)
            # y:y_max:stride are the indices (of the img) seen by the kernel idx (y,x) as it slides along.
            image[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return image[:, :, pad:H + pad, pad:W + pad]

    