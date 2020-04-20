### mathematical functions
### 2018.5.7 Mon.


### modules
import numpy as np


### functions
def gaussian1D(x, A, mx, sigx):
    # Generate normalized 1D Gaussian

    # x: x value (coordinate)
    # A: Amplitude. Not a peak value, but the integrated value.
    # mx: mean value
    # sigx: standard deviation
    coeff = A/np.sqrt(2.0*np.pi*sigx*sigx)
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    gauss=coeff*expx
    return(gauss)

def gaussian2D(x, y, A, mx, my, sigx, sigy, pa=0):
    # Generate normalized 2D Gaussian

    # x: x value (coordinate)
    # y: y value
    # A: Amplitude. Not a peak value, but the integrated value.
    # mx, my: mean values
    # sigx, sigy: standard deviations
    # pa: position angle [deg]. Counterclockwise is positive.
    x, y = rotate2d(x,y,pa)


    coeff = A/(2.0*np.pi*sigx*sigy)
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    gauss=coeff*expx*expy
    return(gauss)


# 2D rotation
def rotate2d_matrx(array2d, angle, deg=True, coords=False):
    '''
    Rotate Cartesian coordinates.
    Right hand direction will be positive.

    array2d: input array
    angle: rotational angle [deg or radian]
    axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
    deg (bool): If True, angle will be treated as in degree. If False, as in radian.
    '''

    # degree --> radian
    if deg:
    	angle = np.radians(angle)
    else:
        pass

    if coords:
        angle = -angle
    else:
        pass

    cos = np.cos(angle)
    sin = np.sin(angle)

    Rrot  = np.array([[cos,-sin],
    				[sin,cos]])

    xyrot = np.dot(Rrot,array2d)
    return xyrot


# 2D rotation
def rotate2d(x, y, angle, deg=True, coords=False):
    '''
    Rotate Cartesian coordinates.
    Right hand direction will be positive.

    array2d: input array
    angle: rotational angle [deg or radian]
    axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
    deg (bool): If True, angle will be treated as in degree. If False, as in radian.
    '''

    # degree --> radian
    if deg:
    	angle = np.radians(angle)
    else:
        pass

    if coords:
        angle = -angle
    else:
        pass

    cos = np.cos(angle)
    sin = np.sin(angle)

    xrot = x*cos - y*sin
    yrot = x*sin + y*cos

    return xrot, yrot


# 3D rotation
def rotate3d(array3d, angle, axis=0, deg=True, coords=False):
    '''
    Rotate Cartesian coordinate or (1,3) array.
    Right hand direction will be positive.

    array3d: input array
    angle: rotational angle [deg or radian]
    axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
    deg (bool): If True, angle will be treated as in degree. If False, as in radian.
    coords (bool): If True, rotate positions in a coordinate. If False, rotate a coordinate to another. Default, False.
    '''

    if deg:
        angle = np.radians(angle)
    else:
    	pass

    if coords:
        angle = -angle
    else:
        pass

    cos = np.cos(angle)
    sin = np.sin(angle)

    Rx = np.array([[1.,0.,0.],
                  [0.,cos,-sin],
                  [0.,sin,cos]])

    Ry = np.array([[cos,0.,sin],
                  [0.,1.,0.],
                  [-sin,0.,cos]])

    Rz = np.array([[cos,-sin,0.],
                  [sin,cos,0.],
                  [0.,0.,1.]])

    if axis == 0:
        # rotate around x axis
        newarray = np.dot(Rx,array3d)
    elif axis == 1:
        # rotate around y axis
        newarray = np.dot(Ry,array3d)
    elif axis == 2:
        # rotate around z axis
        newarray = np.dot(Rz,array3d)
    else:
        print ('ERROR\trotate3d: axis value is not suitable.\
            Choose value from (0,1,2). (0,1,2) mean rotatinal axes, (x,y,z) respectively.')

    return newarray


# calculate error propagation of dividing
def err_frac(denm,numr,sigd,sign):
    # denm: denominator (bunbo)
    # numr: numerator (bunshi)
    # sigd: error of denm
    # sign: error of numr
    err = np.sqrt((sign/denm)*(sign/denm) + (numr*sigd/(denm*denm))*(numr*sigd/(denm*denm)))
    return err


if __name__ == '__main__':
	test2d = np.array([1.,0.])
	ans2d  = rotate2d(1.,0., 45., deg=True, coords=True)
	print (ans2d)