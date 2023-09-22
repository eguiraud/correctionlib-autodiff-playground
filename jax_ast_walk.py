import jax.numpy as jnp
from jax import grad

# Our starting point: an AST that expresses `x*x + y`
# (in reality the AST is built by parsing a math expression coming from an input file)
ast = {"+": [{"*": [{"in": "x"}, {"in": "x"}]}, {"in": "y"}]}


def synthesize(ast: dict) -> str:
    """Walk the AST and build code for the corresponding Python function"""
    used_inputs = set()

    def write_fn_body(ast):
        key, value = next(iter(ast.items()))
        if key == "+":
            return f"{write_fn_body(value[0])} + {write_fn_body(value[1])}"
        elif key == "*":
            return f"{write_fn_body(value[0])} * {write_fn_body(value[1])}"
        elif key == "in":
            used_inputs.add(value)
            return value

    fn_body = write_fn_body(ast)

    # we need to establish a convention for the ordering of the inputs.
    # alphabetical seems reasonable.
    used_inputs = sorted(used_inputs)
    fn_header = f"lambda {','.join([var for var in used_inputs])}: "

    return eval(fn_header + fn_body), used_inputs


f, used_inputs = synthesize(ast)
df = grad(f, argnums=[0, 1])

inputs = {
    "x": jnp.array(1.0),
    "y": jnp.array(2.0),
}

print(df(*(inputs[k] for k in used_inputs)))
