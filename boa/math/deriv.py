import numpy as np
def deriv(y, x=None):
    """
    ==================================================================
    NOTE!!!!!! The Numdifftools from PyPI may do this a lot faster and more 
    accurately!!!!
    ===================================================================

    Perform numerical differentiation using 3-point, Lagrangian interpolation.
    Seems to be a lack of good derivative function in Python.
    Translated from IDL deriv.pro.
    
    Three-point Lagrangian interpolation polynomial for [x0,x1,x2], [y0, y1, y2]
    f = (x-x1)(x-x2)/(x01*x02)*y0 +
       (x-x0)(x-x2)/(x10*x12)*y1 +
       (x-x0)(x-x1)/(x02*x12)*y2
    Where: x01 = x0 - x1, x02 = x0 - x2, x12 = x1-x2, etc.

    df/dx = y' = y0*(2x-x1-x2)/(x01*x02) +
              y1*(2x-x0-x2)/(x10*x12) +
              y2*(2x-x0-x1)/(x02*x12)

    Evaluate at central point x=x1 (flip sign of last term so we can use x01):
    y' = y0*x12/(x01*x02) + y1*(1/x12 - 1/x01) - y2*x01/(x02*x12)

    At x=x0 (for the first point):
    y' = y0*(x01+x02)/(x01*x02) - y1*x02/(x01*x12) + y2*x01/(x02*x12)

    At x=x2 (for the last point):
    y' = -y0*x12/(x01*x02) + y1*x02/(x01*x12) - y2*(x02+x12)/(x02*x12)
       
    Parameters
    ----------
    y: list
       Variable to be differentiated.
    x: list (optional)
       Variable to differentiated with respect to. If omitted, unit spacing
       for y (i.e. x(i) = i) is assumed.

    Output
    ------
    deriv: list
        derivative of y with respect to x
    """
    n = len(y)
    if n < 3:
        raise ValueError('y must have at least 3 elements!')
    if(x is not None):
        if len(x) != len(y):
            raise ValueError('vectors x and y must have the same size!')
        x1 = np.asarray(x, dtype=np.float64)
        x0 = np.roll(x1, 1)
        x2 = np.roll(x1, -1)
        x12 = x1 - x2  # x1 - x2
        x01 = x0 - x1  # x0 - x1
        x02 = x0 - x2 # x0 - x2
        
        # middle points
        d = np.roll(y, 1) * (x12 / (x01*x02)) + \
            y * (1./x12 - 1./x01) - \
            np.roll(y, -1) * (x01/ (x02*x12))
        # first and last points
        d[0] = y[0] * (x01[1]+x02[1])/(x01[1]*x02[1]) -\
            y[1] * x02[1]/(x01[1]*x12[1]) + \
            y[2] * x01[1]/(x02[1]*x12[1])
        d[-1] = -y[-3]*x12[-2]/(x01[-2]*x02[-2]) + \
            y[-2]* x02[-2]/(x01[-2]*x12[-2]) -\
            y[-1]* (x02[-2]+x12[-2]) / (x02[-2]*x12[-2])
    else:
          #Equally spaced point case
        d = (np.roll(y,-1) - np.roll(y,1))/2.
        d[0] = (-3.0*y[0] + 4.0*y[1] - y[2])/2.
        d[-1] = (3.*y[-1] - 4.*y[-2] + y[-3])/2.
    return d

def pderiv2D(field, xld, dim = 0):
    """
    Generalises the numerical differentiation function deriv to 
    calculate the partial derivative of a 2-D array representing 
    a two-variable function with respect to x or y given by dim
    
    Parameters
    ----------

    field: array_like
        An MxN array containing the values of the function f(x, y) 
        to be differentiated at given (x, y) coordinate points.

    xld: array_like
        Variable to be differentiated with respect to. If dim = 0, then
        len(xld) = M, else len(xld) = N

    dim: int (optional)
        determines which variable to be differentiated wtih respect to.
        dim = 0 (x)
        dim = 1 (y) 
        If omitted assume diffentiation by x
    Output
    ------
    dfield: array_like
        MxN array containing the values of the derivative of f

    Example
    -------
    f = [[1, 2, 3][4, 5, 6]] (3 x 2)
    x = [0.5, 0.1, 1.5]
    y = [0.5, 0.1]
    df/dx = pderiv(f, x, dim = 0) 
    df/dy = pderiv(f, y, dim = 1)
    """
    n_x, n_y = field.shape
    dfield = np.zeros_like(field)
    if (dim not in [0, 1]): 
        raise ValueError("2-D function, enter dim = 0 (df/dx) or dim = 1 (df/dy)")
    if (dim == 0):
        # check if len(x) equals M
        if len(xld) != n_x : 
            raise ValueError("x-direction lengths do not match")
        for j in range(n_y):
            dfield[:, j] = deriv(field[:,j], np.array(xld))
    if (dim == 1):
        if len(xld) != n_y:
            raise ValueError('y-direction lengths do not match')
        for i in range(n_x):
            dfield[i,:] = deriv(field[i,:], np.array(xld))
    return dfield

def pderiv3D(infield, xld, dim = 0):
    """
    The same as pderiv2D above, except for 3-dimensions
    TODO:: NEED TO Generalise to n-D
    Sufficient for meteorological purposes
    """
    n_x, n_y, n_z = infield.shape
    outfield = np.zeros_like(infield)
  
    if (dim == 0):
        if len(xld) != n_x:
            raise ValueError('x-lengths do not match')
        for j in range(n_y):
            for k in range(n_z):
                outfield[:, j, k] = deriv(infield[:,j,k], xld)
    elif (dim == 1):
        if len(xld) != n_y:
            raise ValueError('y-lengths do not match')
        for i in range(n_x):
            for k in range(n_z):
                outfield[i,:,k] = deriv(infield[i,:,k], xld)
    elif (dim == 2):
        if len(xld) != n_z:
            raise ValueError('z-lengths do not match')
        for i in range(n_x):
            for j in range(n_y):
                outfield[i,j,:] = deriv(infield[i,j,:], xld)
    else:
        raise ValueError('3-Dimensional array, dim = 0, 1, or 2')
    return outfield
    





         