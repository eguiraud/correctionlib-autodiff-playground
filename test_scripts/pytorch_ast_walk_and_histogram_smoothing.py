import torch as to
from typing import Iterable
from scipy.interpolate import CubicSpline

# An AST that expresses `x*x + binning(y)*y`
ast = {
    "+": [
        {"*": [{"in": "x"}, {"in": "x"}]},
        {
            "*": [
                {
                    "binning": {
                        "edges": [0.0, 1.0, 3.0],
                        "content": [3.0, 4.0],
                        "in": "y",
                    }
                },
                {"in": "y"},
            ]
        },
    ]
}


class BinLookupWithGrad(to.autograd.Function):
    @staticmethod
    def forward(ctx, input, edges, content, dspline):
        ctx.dspline = dspline
        ctx.save_for_backward(input)
        bin_idx = to.searchsorted(edges, input)
        if bin_idx != 0:
            bin_idx -= 1
        return content[bin_idx]

    @staticmethod
    # once_differentiable because we are saving dspline in ctx:
    # see https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
    @to.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return to.tensor(ctx.dspline(input.item())) * grad_output, None, None, None


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


def compute(ast: dict, inputs: dict) -> to.Tensor:
    """Walk the AST, apply the corresponding operations to the inputs"""
    key, value = next(iter(ast.items()))
    if key == "+":
        return compute(value[0], inputs) + compute(value[1], inputs)
    elif key == "*":
        return compute(value[0], inputs) * compute(value[1], inputs)
    elif key == "binning":
        b = Binning(value["edges"], value["content"])
        in_var = inputs[value["in"]]
        return b(in_var)
    elif key == "in":
        return inputs[value]


# Pytorch input tensors
inputs = {
    "x": to.tensor([1.0], requires_grad=True),
    "y": to.tensor([2.0], requires_grad=True),
}

# Apply the operation to Pytorch tensors:
# Pytorch dynamically records what operations are performed on them
# during the forward pass, so it can do reverse mode autodiff
out = compute(ast, inputs)

# Run backprop/reverse mode autodiff
out.backward()

print(f'{inputs["x"].grad=}, {inputs["y"].grad=}')
