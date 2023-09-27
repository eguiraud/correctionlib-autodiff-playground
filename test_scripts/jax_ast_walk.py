from functools import partial
import jax.numpy as jnp
import jax

ast = {"+": [{"*": [{"in": "x"}, {"in": "x"}]}, {"in": "y"}]}

def compute(ast: dict, inputs: dict) -> jax.Array:
    """Walk the AST, apply the corresponding operations to the inputs"""
    key, value = next(iter(ast.items()))
    if key == "+":
        return compute(value[0], inputs) + compute(value[1], inputs)
    elif key == "*":
        return compute(value[0], inputs) * compute(value[1], inputs)
    elif key == "in":
        return inputs[value]

inputs = {
    "x": jnp.float32(1.0),
    "y": jnp.float32(2.0),
}

evaluator = jax.jit(jax.value_and_grad(partial(compute, ast)))
print(evaluator(inputs))
