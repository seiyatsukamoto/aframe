from typing import Optional, Union

import torch
from jaxtyping import Float, Int64
from torch import Tensor
from torch.nn.functional import unfold

from ml4gw.types import TimeSeries1d, TimeSeries1to3d, TimeSeries2d, TimeSeries3d

BatchTimeSeriesTensor = Union[Float[Tensor, "batch time"], TimeSeries3d]

from ml4gw.utils.slicing import slice_kernels

def sample_kernels_MM(
    X: TimeSeries1to3d,
    kernel_size: int,
    signal_time: int,
    min_kernel_size: int,
    offset: int,
    filter_size: int,
    N: Optional[int] = None,
    coincident: bool = True,
) -> BatchTimeSeriesTensor:
    #This is from ml4gw.utils.slicing
    #Here we dont assume center to be signal time
    #Instead we assume that rightmost kernel is valid
    if X.shape[-1] < kernel_size:
        raise ValueError(
            "Can't sample kernels of size {} from tensor with shape {}".format(
                kernel_size, X.shape
            )
        )
    elif X.ndim > 3:
        raise ValueError(
            f"Can't sample kernels from tensor with {X.ndim} dimensions"
        )
    elif X.ndim < 3 and N is None:
        raise ValueError(
            "Must specify number of kernels N if X has fewer than 3 dimensions"
        )
    elif X.ndim == 3 and N is not None and N != len(X):
        raise ValueError(
            (
                "Can't sample {} kernels from 3D tensor with "
                "batch dimension {}"
            ).format(N, len(X))
        )
    
    min_val = signal_time - kernel_size + offset + filter_size//2
    max_val = signal_time - kernel_size + min_kernel_size - offset + filter_size//2

    if max_val > X.shape[-1]:
        raise ValueError(
            (
                "max_val too big {} > X.shape[-1] {}"
            ).format(max_val, X.shape[-1])
        )
    
    if min_val < 0:
        raise ValueError(
            (
                "min_val negative {}"
            ).format(min_val)
        )

    if X.ndim == 3 or coincident:
        # sampling coincidentally, so just need a single
        # index for each element in the output batch
        N = N or len(X)
        shape = (N,)
    else:
        # otherwise, each channel in each batch sample
        # will require its own sampling index
        shape = (N, len(X))

    idx = torch.randint(min_val, max_val, size=shape).to(X.device)
    return slice_kernels(X, idx, kernel_size)