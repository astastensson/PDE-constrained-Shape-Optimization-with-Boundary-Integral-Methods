# Example density to use as boundary condition
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt


def rad(a, b, angle):
    return 1 + a * jnp.cos(b*angle)    

def r(a, b, t):
    rt = rad(a, b, t)
    return rt * jnp.cos(t), rt * jnp.sin(t)

def Gr(a, b, x, y):
    rxy = jnp.sqrt(x**2 + y**2)
    t = jnp.arctan2(y, x)
    return jnp.exp(-(rxy - rad(a, b, t))**2 / 0.1)



def main():
    a = 1; b = 5
    def G(x, y):
        return Gr(a, b, x, y)
    x = jnp.linspace(-1.5, 1.5, 100)
    y = jnp.linspace(-1.5, 1.5, 100)
    X, Y = jnp.meshgrid(x, y)
    t = jnp.linspace(0, 2*jnp.pi, 100)
    x, y = vmap(r, (None, None, 0))(a, b, t)
    plt.plot(x, y)
    Gxy = vmap(vmap(G, (0,0)), (0, 0))(X, Y)
    plt.pcolormesh(X, Y, Gxy)
    plt.show()

if __name__== "__main__":
    main()