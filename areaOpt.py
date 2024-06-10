#Functions to optimize area of ellipse under constraint: constant circumference, constant middle point
from geometries import *
from quadrature import *
from insideDirichlet import *
from plotfuncs import *
from jax import numpy as jnp
from jax import random, grad, vmap
import matplotlib.pyplot as plt

def loss(a, b, t, w):
    return area(FourierEllipse(a, b), t, w)

def FourierEllipse(a, b):
    return Fourier(jnp.array([a, 0., b]))

def initiateEllipse(n, seed = 0):
    #Pseudorandom ellipse
    #n: nr of quadrature points in each Gauss-Legendre interval
    #returns: ellipse axis parameters a0, b0
    #quadrature points and weights t, w
    #initial interval
    k = random.key(seed=seed)
    [a, b] = random.uniform(k, shape=(2,), minval=1/8, maxval=6.)
    #print(a, b)
    a0 = 1/2*(a-b); b0 = 1/2*(a+b)
    r0 = FourierEllipse(a0, b0)
    seg = GaussSegments(splitIntervals(findCorners(r0), 1), n)
    t, w = seg.getSegments()
    circ0 = circumference(r0, t, w)
    a0 = a0/circ0*2*jnp.pi; b0 = b0/circ0*2*jnp.pi
    return a0, b0, t, w, seg.intervals


def gradientDesc(a, b, t, w, n, gam=0.1):
    #Performs gradient descent for maximize area of ellipse
    #a, b: start axes of ellipse
    # t, w: start quadrature points and weghts
    # n: number of points to use in each Gauss-Legendre segment
    # gam: step length for gradient descent
    #Returns:
    #iterations performed
    #parameters (ellipse axis a, b) obtained in each iteration
    #gradients dL/da, dL/db obtained in each iteration
    #differences in loss between each iteration
    #areas obtained in each iteration
    #quadrature points and weights for each iteration
    params = [(a, b)]
    areas = [area(FourierEllipse(a, b), t, w)]
    ts = [t]
    ws = [w]
    diff = 100
    diffs = [abs(areas[0] - jnp.pi)]
    grads = []
    gamma = gam
    iterations = [0]
    while diff > 1e-3:
        iterations.append(iterations[-1] + 1)
        aGrad, bGrad = grad(loss, (0, 1))(a, b, t, w)
        grads.append((aGrad, bGrad))
        a += gamma*aGrad; b += gamma*bGrad
        r0 = FourierEllipse(a, b)
        seg = GaussSegments(splitIntervals(findCorners(r0), 1), n)
        t, w = seg.getSegments()
        ts.append(t)
        ws.append(w)
        circ = circumference(r0, t, w)
        a = a/circ*jnp.pi*2; b = b/circ*jnp.pi*2
        params.append((a, b))
        areas.append(area(FourierEllipse(a, b), t, w))
        diff = abs(areas[-1] - areas[-2])
        diffs.append(diff)

    aGrad, bGrad = grad(loss, (0, 1))(a, b, t, w)
    grads.append((aGrad, bGrad))

    return iterations, params, grads, diffs, areas, ts, ws 

def plotConv(iterations, params):
    #Plots the shapes obtained in gradient descent
    #Iterations: list of completed iterations
    #params: list of parameters for ellipse
    #iterations and params must be same length
    fig, axes = plt.subplots()
    t = jnp.linspace(0, 2*jnp.pi, 100)
    for i, (a, b) in zip(iterations, params):
        ex, ey = vmap(FourierEllipse(a, b))(t)
        plt.plot(ex, ey, label=f'Iteration {i}')

    plt.title('Maximize area of ellipse with constant circumference')
    plt.legend()
    plt.axis('equal')
    plt.show()

def plotErrorAndShape(iterations, params, areas, trueLoss):
    #Plots the shapes obtained in gradient descent and the error between area and true loss in each iteration
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    t = jnp.linspace(0, jnp.pi*2, 50)
    logerrors = jnp.log(trueLoss-jnp.array(areas))
    plt.plot(iterations,logerrors, color='#648BC0')
    plt.scatter(iterations, logerrors, color='#648BC0')
    for i, it in enumerate(iterations):
        (rx, ry) = vmap(FourierEllipse(*params[it]))(t)
        ax_inset = inset_at_point(ax, (it, logerrors[it]), 0.2, 0.2)
        ax_inset.plot(rx, ry, color="#fc7307")
        ax_inset.set_xlim(-1.75, 1.75)
        ax_inset.set_ylim(-1.75, 1.75)
        plot_background_highlight(ax_inset)
        ax_inset.axis("off")

    plt.yticks([(i) for i in [0,-3,-6,-9,-12]], [f"$10^{{{i}}}$" for i in [0,-3,-6,-9,-12]])
    plt.xticks(iterations)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Convergence")



def plotGrads(iterations, params, grads):
    #Plots the shapes obtained in gradient descent, with the gradient visualized as vector field
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9,3))
    plt.suptitle(f'Maximize area of ellipse with constant circumference, with gradient')
    its = [iterations[0], iterations[int(jnp.round(len(iterations)/2))], iterations[-1]]
    its2 = [0, 1, 2]
    #for i, ax in zip(its, axs.ravel()):
    for i, j in zip(its, its2):
        r0 = FourierEllipse(*params[i])
        t = jnp.linspace(0, 2*jnp.pi, 40)
        rx,ry = vmap(r0)(t)
        drx, dry = vmap(jacrev(r0))(t)
        da, db = grads[i-2]
        drx, dry = vmap(FourierEllipse(da, db))(t)
        axs[j].set_title(f'Iteration {i}')
        axs[j].plot(rx, ry, label=f'Curve, iteration {iterations[-1]}')
        axs[j].quiver(rx, ry, drx, dry, label=f'Gradient, iteration {iterations[-1]}')
        axs[j].set_xlim((-1.75, 1.75))
        axs[j].set_ylim((-1.75, 1.75))
    
def plotPoints(iterations, params, ts):  
    #Plots shapes obtained in gradient descent and discretization points used
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9,3))
    plt.suptitle(f'Maximize area of ellipse with constant circumference, with discrete points')
    its = [iterations[0], iterations[int(jnp.round(len(iterations)/2))], iterations[-1]]
    its2 = [0, 1, 2]
    #for i, ax in zip(its, axs.ravel()):
    for i, j in zip(its, its2):
        r0 = FourierEllipse(*params[i])
        diskt = ts[i]
        t = jnp.linspace(0, 2*jnp.pi, 40)
        rx,ry = vmap(r0)(t)
        diskx, disky = vmap(r0)(diskt)
        axs[j].set_title(f'Iteration {i}')
        axs[j].plot(rx, ry, label=f'Curve, iteration {iterations[-1]}')
        axs[j].scatter(diskx, disky,label=f'Discrete points, iteration {iterations[-1]}')
        axs[j].set_xlim((-1.75, 1.75))
        axs[j].set_ylim((-1.75, 1.75))  
