# Solver for Laplaces equation with Dirichlet boundary condition using boundary integral methods

from jax import lax
from jax import numpy as jnp
from jax import vmap, jacrev, jacfwd

cond = lax.cond

#Useful operations in two dimensional space
def vector_subtraction_2D(x, y):
    return (x[0]-y[0], x[1]-y[1])

def determinant_2D(x, y):
    return x[0]*y[1]-x[1]*y[0]

def norm_squared(x):
    return x[0]**2 + x[1]**2

def integrate(f, sigma, tdisk, w):
    return jnp.sum(vmap(f)(tdisk)*w*sigma)


#Integral kernels
def kern_lim(r, t):
    dr = jacrev(r)(t)
    ddr = jacfwd(jacrev(r))(t)
    return determinant_2D(ddr, dr)/norm_squared(dr)/4/jnp.pi

def kern(r, s, t):
    diff = vector_subtraction_2D(r(s), r(t))
    drt = jacrev(r)(t)
    return determinant_2D(diff, drt)/norm_squared(diff)/2/jnp.pi

def kernel(r, s, t, tol=1e-13):
    s_lim_t = abs(s-t)< tol
    limit = lambda t: kern_lim(r, t)
    no_limit = lambda t: kern(r, s, t)
    return cond(s_lim_t, limit, no_limit, t)

def kernelOfx(x, r, t):
    diff = vector_subtraction_2D(x, r(t))
    drt = jacrev(r)(t)
    return determinant_2D(diff, drt)/norm_squared(diff)/2/jnp.pi

def evalFunc(x,y,r,sigma,tdisk,w):
    integrand = lambda t: kernelOfx((x,y),r,t)
    return integrate(integrand, sigma, tdisk, w)


#Solver
def solveInsideDirichlet(r, f, t, w):
    #Solves Laplaces equation with Dirichlet condition in two dimensions
    #r: boundary of domain as a parameterization in t
    #f: boundary condition u = f on boundary
    #t: quadrature points
    #w: quadrature weights
    #Returns: function u: R2 --> R
    N = len(t)
    sMatrix = jnp.transpose(jnp.tile(t, (N, 1)))
    tMatrix = jnp.tile(t, (N,1))
    wMatrix = (jnp.tile(w, (N,1)))
    
    def bndry_ker(s,t,w):
        return kernel(r,s,t)*w    
    
    B = vmap(bndry_ker, (0,0,0))(sMatrix.flatten(),tMatrix.flatten(), wMatrix.flatten()).reshape(N,N)
    I = jnp.identity(N)
    F = vmap(f)(t)
    

    sigma = jnp.linalg.solve(jnp.add(-1/2*I, B), F)

    def u(x,y):
        return evalFunc(x, y, r, sigma, t, w)
    
    return u

