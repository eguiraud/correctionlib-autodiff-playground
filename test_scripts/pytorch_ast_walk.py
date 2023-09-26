import torch as to

# Our starting point: an AST that expresses `x*x + y`
# (in reality the AST is built by parsing a math expression coming from an input file)
ast = {"+": [{"*": [{"in": "x"}, {"in": "x"}]}, {"in": "y"}]}


def compute(ast: dict, inputs: dict) -> to.Tensor:
    """Walk the AST, apply the corresponding operations to the inputs"""
    key, value = next(iter(ast.items()))
    if key == "+":
        return compute(value[0], inputs) + compute(value[1], inputs)
    elif key == "*":
        return compute(value[0], inputs) * compute(value[1], inputs)
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
