import torch as to
from typing import Iterable
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

class BinLookupWithGrad(to.autograd.Function):
    @staticmethod
    def forward(input, edges, content, dspline):
        bin_idx = to.searchsorted(edges, input)
        bin_idx[bin_idx != 0] -= 1
        return content[bin_idx]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        input, edges, content, dspline = inputs
        ctx.dspline = dspline
        ctx.save_for_backward(input)

    @staticmethod
    # because we are saving dspline in ctx
    # see https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
    @to.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return to.tensor(np.vectorize(ctx.dspline)(input)) * grad_output, None, None, None


def midpoints(x):
    return 0.5 * (x[1:] + x[:-1])


class Binning:
    def __init__(self, edges: Iterable[float], content: Iterable[float]):
        self._edges = to.tensor(edges)
        self._content = to.tensor(content)
        self._spline = CubicSpline(
            midpoints(self._edges), self._content, bc_type="clamped"
        )
        self._dspline = self._spline.derivative(1)

    def __call__(self, input):
        return BinLookupWithGrad.apply(input, self._edges, self._content, self._dspline)


if __name__ == "__main__":
    edges = np.linspace(-10., 10., 10)
    b = Binning(edges=edges, content=edges[:-1]**2)
    x = to.linspace(-10., 10., 100, requires_grad=True)
    y = b(x)
    y.backward(x)
    print(x.grad)

    grads = x.grad
    x = x.detach().numpy()
    y = y.detach().numpy()
    plt.plot(x, y, label="bin values")
    plt.plot(x, b._spline(x), label="spline")
    plt.legend()

    plt.figure()
    plt.plot(x, grads, label="spline derivative")
    plt.legend()
    plt.show()
