from numpy import polynomial, quantile
from jax import numpy as jnp
leggauss = polynomial.legendre.leggauss
from geometries import *
from jax import vmap

# Generating quadrature weights and points for integrating over [0, 2*pi]

def leggaussInterval(a,b, n):
    #Returns n-point Gauss-Legendre quadrature weights and points over the interval [a, b] 
    points, weights = leggauss(n)
    t = (b-a)/2*points + (a+b)/2
    w = (b-a)/2*weights
    return t,w

def gaussPanels(intervals, n):
    #Returns concatenated list of n-point Gauss-Legendre quadrature weights and points over all intervals in intervals
    tlist = []
    wlist = []
    for a,b in zip(intervals[:-1], intervals[1:]):
        t,w = leggaussInterval(a,b,n)
        tlist.append(t)
        wlist.append(w)
    return jnp.concatenate(tlist), jnp.concatenate(wlist)

class GaussSegments:
    #Class to handle Gauss-Legendre quadrature
    def __init__(self, intervals, n):
        #Returns new GaussSegment object
        #intervals: intervals over which to create n-point Gauss-Legendre quadrature
        self.t, self.w = gaussPanels(intervals, n)
        self.bounds = (intervals[0], intervals[-1])
        self.intervals = intervals
        self.n = n

    def refineSegment(self, t):
        #Method to split segment which contains t into two
        # t needs to be a point which splits two segments
        if t==self.bounds[0]:
            t = self.bounds[1]
        idx = self.intervals.index(t)
        lower = self.intervals[idx-1]            
        self.intervals.append(t - (t-lower)/2)

        if t==self.bounds[1]:
            upper = self.intervals[1] + t
            self.intervals.append((upper-t)/2)
        else:
            upper = self.intervals[idx+1]
            self.intervals.append((upper-t)/2 + t)        
        
        self.intervals.sort()
        self.t, self.w = gaussPanels(self.intervals, self.n)

    def refineSegments(self, ts):
        #Method to split the segments which contains t in ts into two
        for t in ts:
            self.refineSegment(t)

    def reduceSegment(self, t):
        #Method to reduce the number of segments.
        # t needs to be a point which splits two segments
        self.intervals.remove(t)
        self.t, self.w = gaussPanels(self.intervals, self.n)

    def reduceSegments(self, ts):
        #Method to reduce the number of segments
        #all t in ts needs to be points which splits two segments
        for t in ts:
            self.reduceSegment(t)

    def getSegments(self):
        #Returns quadrature weights w and points t
        return self.t, self.w
    
def trapPanels(intervals, n):
    t = jnp.linspace(intervals[0], intervals[1], n)
    tlist = []
    wlist = []
    for a,b in zip(intervals[:-1], intervals[1:]):
        t = jnp.linspace(a, b-(b-a)/n, n)
        w = jnp.ones(len(t))*(b-(b-a)/n-a)/n
        tlist.append(t)
        wlist.append(w)
    return jnp.concatenate(tlist), jnp.concatenate(wlist)
    

class TrapSegments:
    #Same as GaussSegments but with trapezoidal rule quadrature
    def __init__(self, intervals, n):
        self.t, self.w = trapPanels(intervals, n)
        self.bounds = (intervals[0], intervals[-1])
        self.intervals = intervals
        self.n = n

    def refineSegment(self, t):
        if t==self.bounds[0]:
            t = self.bounds[1]
        idx = self.intervals.index(t)
        lower = self.intervals[idx-1]            
        self.intervals.append(t - (t-lower)/2)

        if t==self.bounds[1]:
            upper = self.intervals[1] + t
            self.intervals.append((upper-t)/2)
        else:
            upper = self.intervals[idx+1]
            self.intervals.append((upper-t)/2 + t)        
        
        self.intervals.sort()
        self.t, self.w = trapPanels(self.intervals, self.n)

    def refineSegments(self, ts):
        for t in ts:
            self.refineSegment(t)

    def reduceSegment(self, t):
        self.intervals.remove(t)
        self.t, self.w = trapPanels(self.intervals, self.n)

    def reduceSegments(self, ts):
        for t in ts:
            self.reduceSegment(t)

    def getSegments(self):
        return self.t, self.w

def findCorners(r):
    #Function that given a curve r defined on [0, 2*pi] finds points t where curvature is high (in the top 20%)
    #To find more points increase nrOfTestPoints or percentage
    nrOfTestPoints = 50
    percentage = 0.2 # 0.2 = 20%
    ttest = jnp.linspace(0, jnp.pi*2, nrOfTestPoints)
    curvatures = vmap(curvature, in_axes=(None, 0))(r, ttest)
    corners = list(ttest[curvatures > jnp.quantile(curvatures, 1-percentage)])
    corners.insert(len(corners), jnp.pi*2)
    corners.insert(0,0)
    return corners

def splitIntervals(corners, n):
    #Function that splits list of points between 0 and 2*pi into intervals
    #corners: list of points to split
    #n: number of intervals to split each interval into
    #Example: if corners = [0, pi, 2*pi], n=2 then we return [0, pi/2, pi, 3*pi/2, 2*pi]
    tlist = []
    for a,b in zip(corners[:-1], corners[1:]):
        t = [a + (b-a)/n*i for i in range(n)]
        tlist.append(t)    
    ts = jnp.concatenate(jnp.array(tlist))
    return jnp.append(ts, jnp.array([jnp.pi*2]))  

def getTandW(r, n, m):
    #Function that finds Gauss-Legendre quadrature weights w and points t with breakpoints where curvature is high
    #r: curve to integrate over (defined between 0, 2*pi) 
    #n: number of interval between each breakpoint
    #m: number of quadrature points in each interval
    seg = GaussSegments(splitIntervals(findCorners(r), n),m)
    t, w  = seg.getSegments()
    return t, w