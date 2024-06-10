# Geometries parameterized
from jax import numpy as jnp
from jax.lax import cond
from jax import vmap, jacrev
from insideDirichlet import *
## Useful geometries and approximations related to geometries

def star(a, b):
    #Returns curve as function of t
    #in shape of a star if t = [0, 2*pi]
    #a: curvature
    #b: nr of arms
    z = lambda t: ((1 + a*jnp.cos(b*t))*jnp.exp(1j*t))
    return lambda t: (jnp.real(z(t)), jnp.imag(z(t)))

def rectangle(a,b):
    #Returns curve as function of t
    #in shape of a rectangle if t = [0, 2*pi]
    #a: width, b: height
    def rect(t):
        s1 = lambda t: (t*2/jnp.pi*a, 0.)
        s2 = lambda t: (a, (t-jnp.pi/2)*2/jnp.pi*b)
        s3 = lambda t: ((3*jnp.pi/2-t)*a*2/jnp.pi, b)
        s4 = lambda t: (0., (2*jnp.pi-t)*2*b/jnp.pi)
        def firstHalf(t):
            cond2 = t <= jnp.pi/2
            return cond(cond2, s1, s2, t)
        def secondHalf(t):
            cond3 = t <= 3*jnp.pi/2
            return cond(cond3, s3, s4, t)
        cond1 = t <= jnp.pi
        return cond(cond1, firstHalf, secondHalf, t)
    return rect


def ellipse(a, b):
    #Returns curve as function of t
    #in shape of a ellipse if t = [0, 2*pi]
    #a and b ellipse axes
    return lambda t: (a*jnp.cos(t), b*jnp.sin(t))

def putTogether(r1, r2, t0):
    #Returns curve as function of t
    #combines two parameterizations r1, r2 into one
    #t0 the point where r1 stops and r2 begins
    def r(t):
        tcond = t<=t0
        return cond(tcond, r1, r2, t)
    return r

def Fourier(coffs):
    #Returns curve as function of t
    #Symmetric truncated Fourier series with coefficients coffs    
    N = len(coffs)
    K = (N-1)//2
    ks = jnp.linspace(-K, K, 2*K+1)
    kompl = lambda t: jnp.sum(coffs*jnp.exp(1j*ks*t))   
    return lambda t: (jnp.real(kompl(t)), jnp.imag(kompl(t)))


def circumference(r, t, w):
    #Approximation of circumference of the shape defined with r as boundary
    #t, w quadrature points and weights
    dr = lambda t: jnp.sqrt(norm_squared(jacrev(r)(t)))
    return integrate(dr, 1, t, w)

def area(r, t, w):
    #Approximation of area of the shape defined with r as boundary
    #t, w quadrature points and weights
    dr = lambda t: jacrev(r)(t)
    det = lambda t: determinant_2D(r(t), dr(t))
    return 1/2*integrate(det,1, t, w)

def curvature(r, t):
    #Approcimation of curvature of curve r in point t
    dr = jacrev(r)(t)
    ddr = jacfwd(jacrev(r))(t)
    return determinant_2D(dr, ddr)/(norm_squared(dr)*jnp.sqrt(norm_squared(dr)))

def normal(r, t):
    #Returns coordinates and direction of normalvector to r(t) in point t
    rx,ry = vmap(r)(t)
    drx, dry = vmap(jacrev(r))(t)
    return (rx,ry,dry,-drx)

def middleP(r, t, w):
    #Approximation of the middle point of the shape defined with r as boundary
    #t, w quadrature points and weights
    nrm = integrate(lambda t: jnp.sqrt(norm_squared(jacrev(r)(t))),1, t, w)  
    integ1 = lambda t: r(t)[0]*jnp.sqrt(norm_squared(jacrev(r)(t)))/nrm
    integ2 = lambda t: r(t)[1]*jnp.sqrt(norm_squared(jacrev(r)(t)))/nrm      
    return (integrate(integ1, 1, t, w), integrate(integ2, 1, t, w))
