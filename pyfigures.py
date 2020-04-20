### modules
import numpy as np
import sys
import subprocess
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import axes3d
from astropy.io import fits
import matplotlib.patches as patches
import mpl_toolkits.axes_grid1
from mpl_toolkits.axes_grid1 import ImageGrid
#from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
#from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
from astropy import units as u
from astropy.coordinates import SkyCoord
import copy
from scipy.interpolate import griddata
#plt.style.use('seaborn-dark')
#plt.style.use('ggplot')
#plt.style.use('seaborn-deep')
#plt.style.use('default')


### setting for figures
#mpl.use('Agg')
plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
#plt.rcParams['xtick.major.width'] = 1.0 # x ticks width
#plt.rcParams['ytick.major.width'] = 1.0 # y ticks width
plt.rcParams['font.size'] = 14           # fontsize
#plt.rcParams['axes.linewidth'] = 1.0    # edge linewidth



### parameters
formatlist = np.array(['eps','pdf','png','jpeg'])
clight     = 2.99792458e10 # light speed [cm s^-1]



### functions
def change_aspect_ratio(ax, ratio):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    aspect = np.abs(aspect)
    aspect = float(aspect)
    ax.set_aspect(aspect)




def fits_deprojection(self, data=None, header=None, noreg=False, inmode='fits'):
    '''
    Deproject coordinates.

    self: fits file
    '''
    ### reading fits files
    if inmode == 'fits':
        data, header = fits.getdata(self,header=True)
    elif inmode == 'data':
        if data is None:
            print ("inmode ='data' is selected. data must be provided.")
            return
        elif header is None:
            print ("inmode ='data' is selected. header must be provided.")
            return
    else:
        print ("inmode is incorrect. Must be choosen from 'fits' or 'data'.")



    # number of axis
    naxis    = header['NAXIS']
    if naxis < 2:
        print ('ERROR\tfits_deprojection: NAXIS of fits is < 2 although It must be > 2.')
        return

    naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
    label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
    refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
    refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
    if 'CDELT1' in header:
        del_i    = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree

    # beam size (degree)
    if 'BMAJ' in header:
        bmaj     = header['BMAJ'] # degree
        bmin     = header['BMIN'] # degree
        bpa      = header['BPA']  # degree
        if 'BUNIT' in header:
            unit = header['BUNIT']
        else:
            unit = 'deg'
    else:
        plot_beam = False
        bmaj, bmin, bpa = [0,0,0]

    # rest frequency (Hz)
    if 'RESTFRQ' in header:
        restfreq = header['RESTFRQ']
    elif 'RESTFREQ' in header:
        restfreq = header['RESTFREQ']
    elif 'FREQ' in header:
        restfreq = header['FREQ']

    if 'LONPOLE' in header:
        phi_p = header['LONPOLE']
    else:
        phi_p = 180.


    # coordinates
    # read projection type
    try:
        projection = label_i[0].replace('RA---','')
    except:
        print ('Cannot read information about projection from fits file.')
        print ('Set projection SIN for radio interferometric data.')
        projection = 'SIN'

    # rotation of pixel coordinates
    if 'PC1_1' in header:
        pc1_1 = header['PC1_1']
        pc1_2 = header['PC1_2']
        pc2_1 = header['PC2_1']
        pc2_2 = header['PC2_2']
        pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
        pc_ij = pc_ij*np.array([del_i[0],del_i[1]])
    elif 'CD1_1' in header:
        if 'CD1_2' in header:
            pc1_1 = header['CD1_1']
            pc1_2 = header['CD1_2']
            pc2_1 = header['CD2_1']
            pc2_2 = header['CD2_2']
            pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
        else:
            pc1_1 = header['CD1_1']
            pc1_2 = 0.0
            pc2_1 = 0.0
            pc2_2 = header['CD2_2']
            pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
    else:
        print ('CAUTION\tfits_deprojection: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
        pc_ij = np.array([[1.,0.],[0.,1.]])
        pc_ij = pc_ij*np.array([del_i[0],del_i[1]])


    # x & y axes
    xaxis, yaxis = np.array([np.dot(pc_ij, (i+1 - refpix_i[0:2]))\
     for i in range(np.max(naxis_i[0:2]))]).T # +1 in i+1 comes from 0 start index in python
    xaxis = xaxis[:naxis_i[0]]
    yaxis = yaxis[:naxis_i[1]]

    # intermidiate coordinates before the projection correction
    xx, yy = np.meshgrid(xaxis, yaxis)
    #print xx[0,0],xx[-1,-1],yy[0,0],yy[-1,-1]



    # 2. (x,y) --> (phi, theta): native coordinates
    # correct projection effect, and then put into polar coordinates
    # For detail, look into Mark R. Calabretta and Eric W. Greisen (A&A, 2002)
    if projection == 'SIN':
        #print 'projection: SIN'
        phi   = np.arctan2(xx,-yy)*180./np.pi
        theta = np.arccos(np.sqrt(xx*xx + yy*yy)*np.pi/180.)*180./np.pi
        #print phi
        #print theta

        # values for converstion from (phi, theta) to (ra, dec)
        alpha_0 = refval_i[0] # degree
        delta_0 = refval_i[1]
        alpha_p = alpha_0
        delta_p = delta_0
        the_0   = 90.
        phi_0   = 0.
        reg     = False
    elif projection == 'SFL':
        # (ra, dec) of reference position is (0,0) in (phi, theta) and (x,y)
        # (0,0) is on a equatorial line, and (0, 90) is the pole in a native spherical coordinate
        #print 'projection: SFL'
        cos   = np.cos(np.radians(yy))
        phi   = xx/cos # deg
        theta = yy     # deg

        # values for converstion from (phi, theta) to (ra, dec)
        alpha_0 = refval_i[0]
        delta_0 = refval_i[1]
        the_0   = 0.
        phi_0   = 0.
        alpha_p = None
        delta_p = None
        reg     = True
    elif projection == 'GLS':
        print ('WARNING\tfits_deprojection: The projection GFL is treated as a projection SFL.')
        cos   = np.cos(np.radians(yy))
        phi   = xx/cos # deg
        theta = yy     # deg

        # values for converstion from (phi, theta) to (ra, dec)
        alpha_0 = refval_i[0]
        delta_0 = refval_i[1]
        the_0   = 0.
        phi_0   = 0.
        alpha_p = None
        delta_p = None
        reg     = True
    elif projection == 'TAN':
        #print 'projection: TAN'
        phi   = np.arctan2(xx,-yy)*180./np.pi
        theta = np.arctan2(180.,np.sqrt(xx*xx + yy*yy)*np.pi)*180./np.pi

        # values for converstion from (phi, theta) to (ra, dec)
        alpha_0 = refval_i[0]
        delta_0 = refval_i[1]
        the_0   = 90.
        phi_0   = 0.
        alpha_p = alpha_0
        delta_p = delta_0
        reg     = False
    else:
        print ('ERROR\tfits_deprojection: Input value of projection is wrong. Can be only SIN or SFL now.')
        pass


    # 3. (phi, theta) --> (ra, dec) (sky plane)
    # Again, for detail, look into Mark R. Calabretta and Eric W. Greisen (A&A, 2002)
    #print 'Now only relative coordinate is available.'
    # (alpha_p, delta_p): cerestial coordinate of the native coordinate pole
    # In SFL projection, reference point is not polar point

    # parameters
    sin_th0  = np.sin(np.radians(the_0))
    cos_th0  = np.cos(np.radians(the_0))
    sin_del0 = np.sin(np.radians(delta_0))
    cos_del0 = np.cos(np.radians(delta_0))

    # delta_p
    if delta_p is not None:
        pass
    else:
        argy    = sin_th0
        argx    = cos_th0*np.cos(np.radians(phi_p-phi_0))
        arg     = np.arctan2(argy,argx)
        #print arg

        cos_inv  = np.arccos(sin_del0/(np.sqrt(1. - cos_th0*cos_th0*np.sin(np.radians(phi_p - phi_0))*np.sin(np.radians(phi_p - phi_0)))))

        delta_p = (arg + cos_inv)*180./np.pi

        if (-90. > delta_p) or (delta_p > 90.):
            delta_p = (arg - cos_inv)*180./np.pi

        #if (-90. > delta_p) or (delta_p > 90.):
            #print ('delta_p: ', delta_p)
            #print ('No valid delta_p. Use value in LATPOLE.')
            #delta_p = header['LATPOLE']

    sin_delp = np.sin(np.radians(delta_p))
    cos_delp = np.cos(np.radians(delta_p))

    # alpha_p
    if alpha_p:
        #print 'pass'
        pass
    else:
        sin_alpha_p = np.sin(np.radians(phi_p - phi_0))*cos_th0/cos_del0
        cos_alpha_p = sin_th0 - sin_delp*sin_del0/(cos_delp*cos_del0)
        #print sin_alpha_p, cos_alpha_p
        #print np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
        alpha_p = alpha_0 - np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
        #print alpha_p


    # ra, dec
    sin_th = np.sin(np.radians(theta))
    cos_th = np.cos(np.radians(theta))

    in_sin = sin_th*sin_delp + cos_th*cos_delp*np.cos(np.radians(phi-phi_p))
    delta  = np.arcsin(in_sin)*180./np.pi

    argy  = -cos_th*np.sin(np.radians(phi-phi_p))
    argx  = sin_th*cos_delp - cos_th*sin_delp*np.cos(np.radians(phi-phi_p))
    alpha = alpha_p + np.arctan2(argy,argx)*180./np.pi
    #print alpha

    alpha[np.where(alpha < 0.)]   = alpha[np.where(alpha < 0.)] + 360.
    alpha[np.where(alpha > 360.)] = alpha[np.where(alpha > 360.)] - 360.

    # modify length
    alpha = alpha_0 + (alpha - alpha_0)*np.cos(np.radians(delta))
    # cosine term is to rescale alpha to delta at delta
    # exact alpha can be derived without cosine term

    # check
    # both must be the same
    #print sin_th[0,0]
    #print (np.sin(np.radians(delta[0,0]))*sin_delp + np.cos(np.radians(delta[0,0]))*cos_delp*np.cos(np.radians(alpha[0,0]-alpha_p)))
    # both must be the same
    #print cos_th[0,0]*np.sin(np.radians(phi[0,0]-phi_p))
    #print -np.cos(np.radians(delta[0,0]))*np.sin(np.radians(alpha[0,0]-alpha_p))
    # both must be the same
    #print cos_th[0,0]*np.cos(np.radians(phi[0,0]-phi_p))
    #print np.sin(np.radians(delta[0,0]))*cos_delp - np.cos(np.radians(delta[0,0]))*sin_delp*np.cos(np.radians(alpha[0,0]-alpha_p))



    # new x & y axis (ra, dec in degree)
    xx     = alpha
    yy     = delta
    #del_x  = xx[0,0] - xx[1,1]
    #del_y  = yy[0,0] - yy[1,0]
    #xmin   = xx[0,0] - 0.5*del_x
    #xmax   = xx[-1,-1] + 0.5*del_x
    #ymin   = yy[0,0] - 0.5*del_y
    #ymax   = yy[-1,-1] + 0.5*del_y
    #extent = (xmin, xmax, ymin, ymax)
    #print extent
    #print xx[0,0]-alpha_0, xx[-1,-1]-alpha_0,yy[0,0]-delta_0,yy[-1,-1]-delta_0

    # check
    #print xx[0,0],yy[0,0]
    #print xx[0,-1],yy[0,-1]
    #print xx[-1,0],yy[-1,0]
    #print xx[-1,-1],yy[-1,-1]
    #print del_i[0], del_x, xx[0:naxis_i[1]-1,0:naxis_i[1]-1] - xx[1:naxis_i[1],1:naxis_i[1]]
    #print del_i[1], yy[0,0] - yy[1,0]

    # regridding for plot if the projection requires
    if noreg:
        pass
    elif reg:
        # new grid
        xc2     = np.linspace(np.nanmax(xx),np.nanmin(xx),naxis_i[0])
        yc2     = np.linspace(np.nanmin(yy),np.nanmax(yy),naxis_i[1])
        xx_new, yy_new = np.meshgrid(xc2, yc2)

        # nan --> 0
        data[np.where(np.isnan(data))] = 0.
        #print np.max(data)

        # 2D --> 1D
        xinp    = xx.reshape(xx.size)
        yinp    = yy.reshape(yy.size)


        # regrid
        print ('regriding...')


        # internal expression
        data_reg = griddata((xinp, yinp), data.reshape(data.size), (xx_new, yy_new), method='cubic',rescale=True)
        data_reg = data_reg.reshape((data.shape))


        # renew data & axes
        data = data_reg
        xx   = xx_new
        yy   = yy_new

    return data, xx, yy



def Idistmap(fitsdata, data=None, header=None, ax=None, outname=None, imscale=[], outformat='pdf', color=True, cmap='Greys',
             colorbar=False, cbaroptions=np.array(['right','5%','0%','Jy/beam']), vmin=None,vmax=None,
             contour=True, clevels=np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]), ccolor='k', mask=None,
             xticks=[], yticks=[], relativecoords=True, csize=9, scalebar=np.empty(0),
             cstar=True, prop_star=np.array(['1','0.5','white']), logscale=False, bcolor='k',figsize=(11.69,8.27),
             tickcolor='k',axiscolor='k',labelcolor='k',coord_center=None, plot_beam = True, interpolation=None,
             noreg=True, inmode='fits'):
    '''
    Make a figure from single image.

    Args
    fitsdata (fits file): Input fitsdata. It must be image data having 3 or 4 axes.
    outname (str): Output file name. Not including file extension.
    outformat (str): Extension of the output file. Default is eps.
    imscale (ndarray): Image scale [arcsec]. Input as np.array([xmin,xmax,ymin,ymax]).
    color (bool): If True, image will be described in color scale.
    cmap: Choose colortype of color scale
    colorbar (bool): If True, color bar will be showen. Default False.
    cbaroptions: Set colorbar options.
    vmin: Cut off of color scale. Abusolute value.
    contour (bool): If True, contour will be drawn.
    clevels (ndarray): Set contour levels. Abusolute value.
    ccolor: Set contour color.
    xticks, yticks: Optional setting. If input ndarray, set xticks and yticsk as well as input.
    relativecoords (bool): If True, the coordinate is shown in relativecoordinate. Default True.
    csize: Font size.
    scalebar: Optional setting. Input ndarray([barx, bary, barlength, textx, texty, text ]).
     barx and bary are the position where scalebar will be putted. [arcsec].
     barlength is the length of the scalebar in the figure, so in arcsec.
     textx and texty are the position where a label of scalebar will be putted. [arcsec].
     text is a text which represents the scale of the scalebar.
    cstar (bool): If True, central star position will be marked by cross.
    inmode: 'fits' or 'data'. If 'data' is selected, header must be provided. Default 'fits'.
    '''

    ### setting output file name & format
    if (outformat == formatlist).any():
        outname = outname + '.' + outformat
    else:
        print ('ERROR\tIdistmap: Outformat is wrong.')
        return


    ### reading fits files
    if inmode == 'fits':
        # read file
        data, header = fits.getdata(fitsdata,header=True)

        # get coordinate
        data, xx, yy = fits_deprojection(fitsdata, noreg=noreg)
    elif inmode == 'data':
        if data is None:
            print ("inmode ='data' is selected. data must be provided.")
            return
        elif header is None:
            print ("inmode ='data' is selected. header must be provided.")
            return

        # get coordinate
        data, xx, yy = fits_deprojection('self', data=data, header=header, noreg=noreg, inmode=inmode)
    else:
        print ("inmode is incorrect. Must be choosen from 'fits' or 'data'.")



    # number of axis
    naxis    = header['NAXIS']
    if naxis < 2:
        print ('ERROR\tIdistmap: NAXIS of fits is < 2 although It must be > 2.')
        return

    naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
    label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
    refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
    refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
    alpha_0  = refval_i[0] # degree
    delta_0  = refval_i[1]

    if 'CDELT1' in header:
        del_i    = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree

    # beam size (degree)
    if 'BMAJ' in header:
        bmaj     = header['BMAJ'] # degree
        bmin     = header['BMIN'] # degree
        bpa      = header['BPA']  # degree
        if 'BUNIT' in header:
            unit = header['BUNIT']
        else:
            unit = 'deg'
    else:
        plot_beam = False
        bmaj, bmin, bpa = [0,0,0]

    # rest frequency (Hz)
    if 'RESTFRQ' in header:
        restfreq = header['RESTFRQ']
    elif 'RESTFREQ' in header:
        restfreq = header['RESTFREQ']
    elif 'FREQ' in header:
        restfreq = header['FREQ']

    if 'LONPOLE' in header:
        phi_p = header['LONPOLE']
    else:
        phi_p = 180.


    # coordinates
    # read projection type
    try:
        projection = label_i[0].replace('RA---','')
    except:
        print ('Cannot read information about projection from fits file.')
        print ('Set projection SIN for radio interferometric data.')
        projection = 'SIN'

    # rotation of pixel coordinates
    if 'PC1_1' in header:
        pc1_1 = header['PC1_1']
        pc1_2 = header['PC1_2']
        pc2_1 = header['PC2_1']
        pc2_2 = header['PC2_2']
        pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
        pc_ij = pc_ij*np.array([del_i[0],del_i[1]])
    elif 'CD1_1' in header:
        if 'CD1_2' in header:
            pc1_1 = header['CD1_1']
            pc1_2 = header['CD1_2']
            pc2_1 = header['CD2_1']
            pc2_2 = header['CD2_2']
            pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
        else:
            pc1_1 = header['CD1_1']
            pc1_2 = 0.0
            pc2_1 = 0.0
            pc2_2 = header['CD2_2']
            pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
    else:
        print ('CAUTION\tIdistmap: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
        pc_ij = np.array([[1.,0.],[0.,1.]])
        pc_ij = pc_ij*np.array([del_i[0],del_i[1]])


    # coordinate style & center
    # coordinate center
    if coord_center:
        # ra, dec
        refra, refdec = coord_center.split(' ')
        ref           = SkyCoord(refra, refdec, frame='icrs')
        refra_deg     = ref.ra.degree                     # in degree
        refdec_deg    = ref.dec.degree                    # in degree
        cc            = np.array([refra_deg, refdec_deg]) # absolute coordinate of image center
    else:
        refra_deg  = alpha_0
        refdec_deg = delta_0
        cc         = np.array([refra_deg, refdec_deg]) # absolute coordinate of image center

    # coordinate style
    if relativecoords:
        offra  = alpha_0 + (refra_deg-alpha_0)*np.cos(np.radians(refdec_deg))
        offdec = refdec_deg
        arcsec = True
        cc     = np.array([0.,0.])

        xlabel = 'RA offset (arcsec)'
        ylabel = 'DEC offset (arcsec)'
    else:
        print ('WARNING: Abusolute coordinates are still in development.\
         RA (deg) value is rescaled and not correct.\n\
         relativecoords = True is recommended.')
        offra  = 0.
        offdec = 0.
        arcsec = False
        # units of axes are in degree
        # setting parameters used to plot
        xlabel = label_i[0]
        ylabel = label_i[1]



    # check data axes
    if len(data.shape) == 2:
        pass
    elif len(data.shape) == 3:
        data = data[0,:,:]
    elif len(data.shape) == 4:
        data = data[0,0,:,:]
    else:
        print ('Error\tsingleim_to_fig: Input fits size is not corrected.\
         It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
        return


    # unit: arcsec or deg
    if arcsec:
        xx     = xx*3600.
        yy     = yy*3600.
        offra  = offra*3600.
        offdec = offdec*3600.
        bmaj   = bmaj*3600.
        bmin   = bmin*3600.
        cc     = cc*3600.

    # figure extent
    xmin   = xx[0,0] - offra
    xmax   = xx[-1,-1] - offra
    ymin   = yy[0,0] - offdec
    ymax   = yy[-1,-1] - offdec
    del_x  = xx[1,1] - xx[0,0]
    del_y  = yy[1,1] - yy[0,0]
    extent = (xmin-0.5*del_x, xmax+0.5*del_x, ymin-0.5*del_y, ymax+0.5*del_y)
    #print (extent)

    # image scale
    if len(imscale) == 0:
        figxmin, figxmax, figymin, figymax = extent  # arcsec

        if figxmin < figxmax:
            cp = figxmax
            figxmax = figxmin
            figxmin = cp

        if figymin > figymax:
            cp = figymax
            figymax = figymin
            figymin = cp
    elif len(imscale) == 4:
        figxmax, figxmin, figymin, figymax = imscale # arcsec

        if arcsec:
            pass
        else:
            figxmin = figxmin/3600. # arcsec --> degree
            figxmax = figxmax/3600.
            figymin = figymin/3600.
            figymax = figymax/3600.

        # set cc as image center
        figxmin = figxmin + cc[0]
        figxmax = figxmax + cc[0]
        figymin = figymin + cc[1]
        figymax = figymax + cc[1]
    else:
        print ('ERROR\tIdistmap: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')


    ### ploting
    # setting figure
    if ax is not None:
        pass
    else:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111)

    plt.rcParams['font.size'] = csize
    #print 'start plot'

    # mask
    if mask:
        #d_formasking                         = data
        #d_formasking[np.isnan(d_formasking)] = 0.
        index_mask                           = np.where(data < mask)
        data[index_mask]                     = np.nan


    # set colorscale
    if vmax:
        pass
    else:
        vmax = np.nanmax(data)

    # logscale
    if logscale:
        norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


    # showing in color scale
    if color:
        # color image
        imcolor = ax.imshow(data, cmap=cmap, origin='lower', extent=extent, norm=norm, interpolation=interpolation, rasterized=True)
        #imcolor = ax.pcolor(xx,yy,data, cmap=cmap, vmin=vmin,vmax=vmax)
        # color bar
        #print 'plot color'
        if colorbar:
            cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax     = divider.append_axes(cbar_loc, cbar_wd, pad=cbar_pad)
            cbar    = fig.colorbar(imcolor, cax = cax)
            cbar.set_label(cbar_lbl)

    if contour:
        imcont02 = ax.contour(data, colors=ccolor, origin='lower', levels=clevels,linewidths=1, extent=(xmin,xmax,ymin,ymax))
        #print 'plot contour'


    # set axes
    ax.set_xlim(figxmin,figxmax)
    ax.set_ylim(figymin,figymax)
    ax.set_xlabel(xlabel,fontsize=csize)
    ax.set_ylabel(ylabel, fontsize=csize)
    if (len(xticks) != 0) and (len(yticks) != 0):
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
    else:
        pass
    ax.set_aspect(1)
    ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, labelsize=csize, color=tickcolor, labelcolor=labelcolor, pad=9)

    # plot beam size
    if plot_beam:
        bmin_plot, bmaj_plot = ax.transLimits.transform((bmin,bmaj)) - ax.transLimits.transform((0,0))   # data --> Axes coordinate
        beam = patches.Ellipse(xy=(0.1, 0.1), width=bmin_plot, height=bmaj_plot, fc=bcolor, angle=bpa, transform=ax.transAxes)
        ax.add_patch(beam)

    # central star position
    if cstar:
        if len(prop_star) == 3:
            ll, lw, cl = prop_star
            ll = float(ll)
            lw = float(lw)
            if relativecoords:
                pos_cstar = np.array([0,0])
            else:
                pos_cstar = cc
        elif len(prop_star) == 4:
            ll, lw, cl, pos_cstar = prop_star
            ll = float(ll)
            lw = float(lw)
            ra_st, dec_st = pos_cstar.split(' ')
            radec_st     = SkyCoord(ra_st, dec_st, frame='icrs')
            ra_stdeg      = radec_st.ra.degree                     # in degree
            dec_stdeg     = radec_st.dec.degree                    # in degree
            if relativecoords:
                pos_cstar = np.array([(ra_stdeg - cc[0])*3600., (dec_stdeg - cc[1])*3600.])
            else:
                pos_cstar = np.array([ra_stdeg, dec_stdeg])
        else:
            print ('ERROR\tIdistmap: prop_star must be size of 3 or 4.')
            return

        if arcsec:
            pass
        else:
            ll = ll/3600.

        cross01 = patches.Arc(xy=(pos_cstar[0],pos_cstar[1]), width=ll, height=1e-9, lw=lw, color=cl,zorder=11)
        cross02 = patches.Arc(xy=(pos_cstar[0],pos_cstar[1]), width=1e-9, height=ll, lw=lw, color=cl,zorder=12)
        ax.add_patch(cross01)
        ax.add_patch(cross02)

    # scale bar
    if len(scalebar) == 0:
        pass
    elif len(scalebar) == 8:
        barx, bary, barlength, textx, texty, text, colors, barcsize = scalebar

        barx      = float(barx)
        bary      = float(bary)
        barlength = float(barlength)
        textx     = float(textx)
        texty     = float(texty)

        scale   = patches.Arc(xy=(barx,bary), width=barlength, height=0.001, lw=2, color=colors,zorder=10)
        ax.add_patch(scale)
        ax.text(textx,texty,text,color=colors,fontsize=barcsize,horizontalalignment='center',verticalalignment='center')
    else:
        print ('scalebar must be 8 elements. Check scalebar.')

    plt.savefig(outname, transparent = True)

    return ax



### moment maps
def multiIdistmap(fitsdata, ax=None, outname=None, imscale=[], outformat='pdf', cmap='Greys',
             colorbar=False, cbaroptions=np.array(['right','5%','0%','Jy/beam']), vmin=None, vmax=None,
             clevels=np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]), ccolor='k',mask=None,
             xticks=np.empty, yticks=np.empty, relativecoords=True, csize=9, scalebar=np.empty(0),
             cstar=True, prop_star=np.array(['1','0.5','white']), locsym=0.1, bcolor='k',logscale=False,
             coord_center=None,tickcolor='k',axiscolor='k',labelcolor='k',figsize=(11.69,8.27)):
    '''
    Make a figure from multi images.

    Args
    fitsdata (fits file): Input fitsdata. It must be image data having 3 or 4 axes.
    outname (str): Output file name. Not including file extension.
    outformat (str): Extension of the output file. Default is eps.
    imscale (ndarray): Image scale [arcsec]. Input as np.array([xmin,xmax,ymin,ymax]).
    color (bool): If True, image will be described in color scale.
    cmap: Choose colortype of color scale
    colorbar (bool): If True, color bar will be showen. Default False.
    cbaroptions: Set colorbar options.
    vmin: Cut off of color scale. Abusolute value.
    contour (bool): If True, contour will be drawn.
    clevels (ndarray): Set contour levels. Abusolute value.
    ccolor: Set contour color.
    mask(float): Mask fits02 where fits01 < mask.
    xticks, yticks: Optional setting. If input ndarray, set xticks and yticsk as well as input.
    relativecoords (bool): If True, the coordinate is shown in relativecoordinate. Default True.
    csize: Font size.
    scalebar: Optional setting. Input ndarray([barx, bary, barlength, textx, texty, text ]).
     barx and bary are the position where scalebar will be putted. [arcsec].
     barlength is the length of the scalebar in the figure, so in arcsec.
     textx and texty are the position where a label of scalebar will be putted. [arcsec].
     text is a text which represents the scale of the scalebar.
    cstar (bool): If True, central star position will be marked by cross.
    locsym: Set locations of beamsize.
    '''

    ### setting output file name & format
    if (outformat == formatlist).any():
        outname = outname + '.' + outformat
    else:
        print ('ERROR\tsingleim_to_fig: Outformat is wrong.')
        return


    ### setting initial parameters
    # for data
    data   = None
    data02 = None
    data03 = None
    data04 = None

    # for header
    header = None
    hd02   = None
    hd03   = None
    hd04   = None

    ### reading fits files
    if type(fitsdata) == tuple:
        nfits = len(fitsdata)
        if nfits == 1:
            # read fits
            data, header = fits.getdata(fitsdata[0],header=True)

            # data shape
            if len(data.shape) == 2:
                pass
            elif len(data.shape) == 3:
                data = data[0,:,:]
            elif len(data.shape) == 4:
                data = data[0,0,:,:]
            else:
                print ('Error\tmultiIdistmap: Input fits size is not corrected.\
                 It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
                return
        elif nfits == 2:
            # read fits
            data, header = fits.getdata(fitsdata[0],header=True)
            data02, hd02 = fits.getdata(fitsdata[1],header=True)
            #print len(data.shape)

            # data shape
            if len(data.shape) == 2:
                pass
            elif len(data.shape) == 3:
                data = data[0,:,:]
            elif len(data.shape) == 4:
                data = data[0,0,:,:]
            else:
                print ('Error\tmultiIdistmap: Input fits size is not corrected.\
                 It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
                return

            # data02 shape
            if len(data02.shape) == 2:
                pass
            elif len(data02.shape) == 3:
                data02 = data02[0,:,:]
            elif len(data02.shape) == 4:
                data02 = data02[0,0,:,:]
            else:
                #print 'here'
                print ('Error\tmultiIdistmap: Input fits size is not corrected.\
                 It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
                return
        else:
            print ('Cannot deal with images more than 3. Now under development.')
            print ('Only use the first and second one, and ignore rest.')
            data, header = fits.getdata(fitsdata[0],header=True)
            data02, hd02 = fits.getdata(fitsdata[1],header=True)

            # data shape
            if len(data.shape) == 2:
                pass
            elif len(data.shape) == 3:
                data = data[0,:,:]
            elif len(data.shape) == 4:
                data = data[0,0,:,:]
            else:
                print ('Error\tmultiIdistmap: Input fits size is not corrected.\
                 It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
                return

            # data02 shape
            if len(data02.shape) == 2:
                pass
            elif len(data02.shape) == 3:
                data02 = data02[0,:,:]
            elif len(data.shape) == 4:
                data02 = data02[0,0,:,:]
            else:
                print ('Error\tmultiIdistmap: Input fits size is not corrected.\
                 It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
                return
    else:
        print ('Input type is wrong. Input type must be tuple.')
        return

    # reading header info.
    xlabel   = header['CTYPE1']
    ylabel   = header['CTYPE2']
    #try:
        #restfreq = header['RESTFRQ'] # Hz
    #except:
        #restfreq = header['RESTFREQ'] # Hz
    refval_x = header['CRVAL1']
    refval_y = header['CRVAL2']
    refpix_x = int(header['CRPIX1'])
    refpix_y = int(header['CRPIX2'])
    del_x_deg = header['CDELT1']
    del_y_deg = header['CDELT2']
    del_x    = header['CDELT1'] # deg --> arcsec
    del_y    = header['CDELT2']
    nx       = header['NAXIS1']
    ny       = header['NAXIS2']
    bmaj     = header['BMAJ']
    bmin     = header['BMIN']
    bpa      = header['BPA']  # [deg]
    unit     = header['BUNIT']
    phi_p    = header['LONPOLE']
    print ('x, y axes are ', xlabel, ' and ', ylabel)
    try:
        projection = xlabel.replace('RA---','')
    except:
        print ('Cannot read information about projection from fits file.')
        print ('Set projection SIN for radio interferometric data.')
        projection = 'SIN'


    # pixel to coordinate
    # 1. pixels --> (x,y)
    # edges of the image
    xmin = (1 - refpix_x)*del_x - 0.5*del_x
    xmax = (nx - refpix_x)*del_x + 0.5*del_x
    ymin = (1 - refpix_y)*del_y - 0.5*del_y
    ymax = (ny - refpix_y)*del_y + 0.5*del_y
    #print xmin, xmax, ymin, ymax


    # 2. (x,y) --> (phi, theta): native coordinate
    # correct projection effect, and then put into polar coordinates
    # For detail, look into Mark R. Calabretta and Eric W. Greisen (A&A, 2002)
    if projection == 'SIN':
        #print 'projection: SIN'
        phi_min = np.arctan2(xmin,-ymin)*180./np.pi                               # at xmin, ymin
        the_min = np.arccos(np.sqrt(xmin*xmin + ymin*ymin)*np.pi/180.)*180./np.pi # at xmin, ymin
        phi_max = np.arctan2(xmax,-ymax)*180./np.pi                               # at xmax, ymax
        the_max = np.arccos(np.sqrt(xmax*xmax + ymax*ymax)*np.pi/180.)*180./np.pi # at xmax, ymax
        alpha_0 = refval_x
        delta_0 = refval_y
        alpha_p = alpha_0
        delta_p = delta_0
        the_0   = 90. # degree
        phi_0   = 0.
        print (phi_min, phi_max, the_min, the_max)
    elif projection == 'SFL':
        # (ra, dec) of reference position is (0,0) in (phi, theta) and (x,y)
        # (0,0) is on a equatorial line, and (0, 90) is the pole in a native spherical coordinate
        #print 'projection: SFL'
        cos = np.cos(np.radians(ymin))
        phi_min = xmin/cos
        the_min = ymin
        cos = np.cos(np.radians(ymax))
        phi_max = xmax/cos
        the_max = ymax
        alpha_0 = refval_x
        delta_0 = refval_y
        the_0   = 0.
        phi_0   = 0.
        alpha_p = None
        delta_p = None
    else:
        print ('ERROR: Input value of projection is wrong. Can be only SIN or SFL now.')
        pass


    # 3. (phi, theta) --> (ra, dec) (sky plane)
    # Again, for detail, look into Mark R. Calabretta and Eric W. Greisen (A&A, 2002)
    if relativecoords and (coord_center is None):
        # no transform from native coordinate to cerestial coordinate
        #extent = (phi_min*60*60, phi_max*60*60, the_min*60*60, the_max*60*60)
        extent = (xmin*60*60, xmax*60*60, ymin*60*60, ymax*60*60)
        bmaj = bmaj*60.*60.
        bmin = bmin*60.*60.
        pos_cstar = (0,0)
        xlabel = 'RA offset (arcsec)'
        ylabel = 'DEC offset (arcsec)'
        #print extent
    else:
        #print 'Now only relative coordinate is available.'
        # (alpha_p, delta_p): cerestial coordinate of the native coordinate pole
        # In SFL projection, reference point is not polar point

        # parameters
        sin_th0 = np.sin(np.radians(the_0))
        cos_th0 = np.cos(np.radians(the_0))
        sin_del0 = np.sin(np.radians(delta_0))
        cos_del0 = np.cos(np.radians(delta_0))

        # delta_p
        if delta_p:
            pass
        else:
            argy    = sin_th0
            argx    = cos_th0*np.cos(np.radians(phi_p-phi_0))
            arg     = np.arctan2(argy,argx)
            #print arg

            cos_inv  = np.arccos(sin_del0/(np.sqrt(1. - cos_th0*cos_th0*np.sin(phi_p - phi_0)*np.sin(phi_p - phi_0))))

            delta_p = (arg + cos_inv)*180./np.pi

            if (-90. > delta_p) or (delta_p > 90.):
                delta_p = (arg - cos_inv)*180./np.pi

            if (-90. > delta_p) or (delta_p > 90.):
                print ('No valid delta_p. Use value in LATPOLE.')
                delta_p = header['LATPOLE']

        sin_delp = np.sin(np.radians(delta_p))
        cos_delp = np.cos(np.radians(delta_p))

        # alpha_p
        if alpha_p:
            pass
        else:
            sin_alpha_p = np.sin(np.radians(phi_p - phi_0))*cos_th0/cos_del0
            cos_alpha_p = sin_th0 - sin_delp*sin_del0/(cos_delp*cos_del0)
            #print sin_alpha_p, cos_alpha_p
            #print np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
            alpha_p = alpha_0 - np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
            #print alpha_p


        # ra, dec of map edges
        # ra min, ra value at (1,1) in pixel
        sin_th = np.sin(np.radians(the_min))
        cos_th = np.cos(np.radians(the_min))

        argy = -cos_th*np.sin(np.radians(phi_min-phi_p))
        argx = sin_th*cos_delp - cos_th*sin_delp*np.cos(np.radians(phi_min-phi_p))
        alpha_min = alpha_p + np.arctan2(argy,argx)*180./np.pi

        if (alpha_min < 0.):
            alpha_min = alpha_min + 360.
        elif (alpha_min > 360.):
            alpha_min = alpha_min - 360.

        # ra max, ra value at (nx,ny) in pixel
        sin_th = np.sin(np.radians(the_max))
        cos_th = np.cos(np.radians(the_max))
        argy = -cos_th*np.sin(np.radians(phi_max-phi_p))
        argx = sin_th*cos_delp - cos_th*sin_delp*np.cos(np.radians(phi_max-phi_p))
        alpha_max = alpha_p + np.arctan2(argy,argx)*180./np.pi

        if (alpha_max < 0.):
            alpha_max = alpha_max + 360.
        elif (alpha_max > 360.):
            alpha_max = alpha_max - 360.

        #print alpha_min, alpha_max

        # dec min, dec value at (1,1) in pixel
        sin_th = np.sin(np.radians(the_min))
        cos_th = np.cos(np.radians(the_min))
        in_sin = sin_th*sin_delp+cos_th*cos_delp*np.cos(np.radians(phi_min-phi_p))
        del_min = np.arcsin(in_sin)*180./np.pi

        # dec max, dec value at (nx,ny) in pixel
        sin_th = np.sin(np.radians(the_max))
        cos_th = np.cos(np.radians(the_max))
        in_sin = sin_th*sin_delp+cos_th*cos_delp*np.cos(np.radians(phi_max-phi_p))
        del_max = np.arcsin(in_sin)*180./np.pi

        #print del_min, del_max

        extent = (alpha_min,alpha_max,del_min,del_max)

        pos_cstar = (refval_x,refval_y)
        #print extent
        #return



    # set coordinate center
    if coord_center:
        # ra, dec
        refra, refdec = coord_center.split(' ')
        ref           = SkyCoord(refra, refdec, frame='icrs')
        refra_deg     = ref.ra.degree   # in degree
        refdec_deg    = ref.dec.degree  # in degree
        if relativecoords:
            sin_delref = np.sin(np.radians(refdec_deg))
            cos_delref = np.cos(np.radians(refdec_deg))
            argx = sin_delref*cos_delp - cos_delref*sin_delp*np.cos(np.radians(refra_deg-alpha_p))
            argy = -cos_delref*np.sin(np.radians(refra_deg-alpha_p))
            arg  = np.arctan2(argy,argx)

            phi_ref  = phi_p + arg*180./np.pi

            in_sin = sin_delref*sin_delp + cos_delref*cos_delp*np.cos(np.radians(refra_deg-alpha_p))

            the_ref = np.arcsin(in_sin)*180./np.pi

            # (phi, theta) --> (x,y)
            if projection == 'SIN':
                rxy = np.cos(np.radians(the_ref))*180./np.pi
                x_ref = rxy*np.sin(np.radians(phi_ref))
                y_ref = -rxy*np.cos(np.radians(phi_ref))
            elif projection == 'SFL':
                x_ref = phi_ref*np.cos(np.radians(the_ref))
                y_ref = the_ref

            refval_x, refval_y = [-x_ref*60.*60., -y_ref*60.*60.]
            #print refval_x, refval_y

            extent = (xmin*60*60+refval_x, xmax*60*60+refval_x, ymin*60*60+refval_y, ymax*60*60+refval_y)
            #print extent

            bmaj = bmaj*60.*60.
            bmin = bmin*60.*60.
            pos_cstar = (0,0)

            xlabel = 'RA offset (arcsec)'
            ylabel = 'DEC offset (arcsec)'
            #print refval_x, refval_y
        else:
            # no meaning but...
            refval_x, refval_y = [refra_deg,refdec_deg]
            pos_cstar = (refval_x,refval_y)

    # set colorscale
    if vmax:
        pass
    else:
        vmax = np.nanmax(data)

    # logscale
    if logscale:
        norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)

    # setting parameters used to plot
    if len(imscale) == 0:
        figxmin, figxmax, figymin, figymax = extent
    elif len(imscale) == 4:
        figxmax, figxmin, figymin, figymax = imscale
    else:
        print ('ERROR\tchannelmap: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')

    # mask
    if mask:
        d_formasking                         = data02
        d_formasking[np.isnan(d_formasking)] = 0.
        index_mask                           = np.where(d_formasking < mask)
        data[index_mask]                     = np.nan

    ### ploting
    # setting figure
    if ax is not None:
        pass
    else:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111)

    plt.rcParams.update({'font.size':csize})

    # showing in color scale
    if data is not None:
        # color image
        imcolor = ax.imshow(data, origin='lower', cmap=cmap, extent=extent,norm=norm, rasterized=True)
        # color bar
        if colorbar:
            cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax     = divider.append_axes(cbar_loc, cbar_wd, pad=cbar_pad)
            cbar    = fig.colorbar(imcolor, cax = cax)
            cbar.set_label(cbar_lbl)

    if data02 is not None:
        imcont02 = ax.contour(data02, colors=ccolor, origin='lower',extent=extent, levels=clevels,linewidths=1)

    # set axes
    ax.set_xlim(figxmin,figxmax)
    ax.set_ylim(figymin,figymax)
    ax.set_xlabel(xlabel,fontsize=csize)
    ax.set_ylabel(ylabel, fontsize=csize)
    if xticks != np.empty and yticks != np.empty:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
    else:
        pass
    ax.set_aspect(1)
    ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, labelsize=csize, pad=9, color=tickcolor, labelcolor=labelcolor)

    # plot beam size
    bmin_plot, bmaj_plot = ax.transLimits.transform((0,bmaj)) - ax.transLimits.transform((bmin,0))   # data --> Axes coordinate
    beam = patches.Ellipse(xy=(0.1, 0.1), width=bmin_plot, height=bmaj_plot, fc=bcolor, angle=bpa, transform=ax.transAxes)
    ax.add_patch(beam)

    # central star position
    if cstar:
        ll,lw, cl = prop_star
        ll = float(ll)
        lw = float(lw)

        cross01 = patches.Arc(xy=pos_cstar, width=ll, height=0.001, lw=lw, color=cl,zorder=11) # xy=(refval_x,refval_y)
        cross02 = patches.Arc(xy=pos_cstar, width=0.001, height=ll, lw=lw, color=cl,zorder=12)
        ax.add_patch(cross01)
        ax.add_patch(cross02)

    # scale bar
    if len(scalebar) == 0:
        pass
    elif len(scalebar) == 8:
        barx, bary, barlength, textx, texty, text, colors, barcsize = scalebar

        barx      = float(barx)
        bary      = float(bary)
        barlength = float(barlength)
        textx     = float(textx)
        texty     = float(texty)

        scale   = patches.Arc(xy=(barx,bary), width=barlength, height=0.001, lw=2, color=colors,zorder=10)
        ax.add_patch(scale)
        ax.text(textx,texty,text,color=colors,fontsize=barcsize,horizontalalignment='center',verticalalignment='center')
    else:
        print ('scalebar must consist of 8 elements. Check scalebar.')

    plt.savefig(outname, transparent = True)

    return ax



### channel map
def channelmap(fitsdata, grid=None, outname=None, outformat='pdf', imscale=[1], color=False,cbaron=False,cmap='Greys', vmin=None, vmax=None,
                contour=True, clevels=np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]), ccolor='k',
                nrow=5, ncol=5,velmin=None, velmax=None, nskip=1,
                xticks=np.empty, yticks=np.empty, relativecoords=True, vsys=None, csize=14, scalebar=np.empty(0),
                cstar=True, prop_star=np.array(['1','0.5','red']), logscale=False, tickcolor='k',axiscolor='k',
                labelcolor='k',cbarlabel=None, txtcolor='k', bcolor='k', figsize=(11.69,8.27),
                cbarticks=None,coord_center=None,noreg=True,sbar_vertical=False, cbaroptions=np.array(['right','5%','0%'])):
    '''
    Make channel maps from single image.

    Args
    fitsdata: Input fitsdata. It must be an image cube having 3 or 4 axes.
    outname: Output file name. Not including file extension.
    outformat: Extension of the output file. Default is eps.
    imscale: scale to be shown (arcsec). It must be given as [xmin, xmax, ymin, ymax].
    color (bool): If True, images will be shown in colorscale. Default is False.
        cmap: color of the colorscale.
        vmin: Minimum value of colorscale. Default is None.
        vmax: Maximum value of colorscale. Default is the maximum value of the image cube.
        logscale (bool): If True, the color will be shown in logscale.
    contour (bool): If True, images will be shown with contour. Default is True.
        clevels (ndarray): Contour levels. Input will be treated as absolute values.
        ccolor: color of contour.
    nrow, ncol: the number of row and column of the channel map.
    relativecoords (bool): If True, the channel map will be produced in relative coordinate. Abusolute coordinate mode is (probably) coming soon.
    velmin, velmax: Minimum and maximum velocity to be shown.
    vsys: Systemic velicity [km/s]. If no input value, velocities will be described in LSRK.
    csize: Caracter size. Default is 9.
    cstar: If True, a central star or the center of an image will be shown as a cross.
    locsym: A factor to decide locations of symbols (beam and velocity label). It must be 0 - 1.
    tickcolor, axiscolor, labelcolor, txtcolor: Colors for each stuffs.
    scalebar (array): If it is given, scalebar will be drawn. It must be given as [barx, bary, bar length, textx, texty, text].
                       Barx, bary, textx, and texty are locations of a scalebar and a text in arcsec.
    nskip: the number of channel skipped
    '''

    ### setting output file name & format
    #nmap = 1
    if (outformat == formatlist).any():
        #outfile = outname + '_nmap{0:02d}'.format(nmap) + '.' + outformat
        outfile = outname + '.' + outformat
    else:
        print ('ERROR\tchannelmap: Outformat is wrong.')
        return


    ### reading fits files
    data, header = fits.getdata(fitsdata,header=True)


    # number of axis
    naxis    = header['NAXIS']
    if naxis < 3:
        print ('ERROR\tchannelmap: NAXIS of fits is < 3 although It must be > 3.')
        return

    naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
    label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
    refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
    refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
    if 'CDELT1' in header:
        del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree

    # beam size (degree)
    if 'BMAJ' in header:
        bmaj     = header['BMAJ'] # degree
        bmin     = header['BMIN'] # degree
        bpa      = header['BPA']  # degree
        if 'BUNIT' in header:
            unit = header['BUNIT']
        else:
            unit = 'deg'
    else:
        plot_beam = False
        bmaj, bmin, bpa = [0,0,0]

    # rest frequency (Hz)
    if 'RESTFRQ' in header:
        restfreq = header['RESTFRQ']
    elif 'RESTFREQ' in header:
        restfreq = header['RESTFREQ']
    elif 'FREQ' in header:
        restfreq = header['FREQ']

    if 'LONPOLE' in header:
        phi_p = header['LONPOLE']
    else:
        phi_p = 180.


    # coordinates
    # read projection type
    try:
        projection = label_i[0].replace('RA---','')
    except:
        print ('Cannot read information about projection from fits file.')
        print ('Set projection SIN for radio interferometric data.')
        projection = 'SIN'

    # rotation of pixel coordinates
    if 'PC1_1' in header:
        pc1_1 = header['PC1_1']
        pc1_2 = header['PC1_2']
        pc2_1 = header['PC2_1']
        pc2_2 = header['PC2_2']
        pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
        pc_ij = pc_ij*np.array([del_i[0],del_i[1]])
    elif 'CD1_1' in header:
        if 'CD1_2' in header:
            pc1_1 = header['CD1_1']
            pc1_2 = header['CD1_2']
            pc2_1 = header['CD2_1']
            pc2_2 = header['CD2_2']
            pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
        else:
            pc1_1 = header['CD1_1']
            pc1_2 = 0.0
            pc2_1 = 0.0
            pc2_2 = header['CD2_2']
            pc_ij = np.array([[pc1_1,pc1_2],[pc2_1,pc2_2]])
    else:
        print ('CAUTION\tchannelmap: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
        pc_ij = np.array([[1.,0.],[0.,1.]])
        pc_ij = pc_ij*np.array([del_i[0],del_i[1]])


    data, xx, yy = fits_deprojection(fitsdata, noreg=noreg)
    alpha_0 = refval_i[0] # degree
    delta_0 = refval_i[1]


    # frequency --> velocity
    del_v    = del_i[2]
    refval_v = refval_i[2]
    refpix_v = refpix_i[2]
    nchan    = naxis_i[2]
    if label_i[2] == 'VRAD' or label_i[2] == 'VELO':
        print ('The third axis is ', label_i[2])
        del_v    = del_v*1.e-3
        refval_v = refval_v*1.e-3
    else:
        print ('The third axis is [FREQUENCY]')
        print ('Convert frequency to velocity')
        del_v    = - del_v*clight/restfreq       # delf --> delv [cm/s]
        del_v    = del_v*1.e-5                   # cm/s --> km/s
        refval_v = clight*(1.-refval_v/restfreq) # radio velocity c*(1-f/f0) [cm/s]
        refval_v = refval_v*1.e-5                # cm/s --> km/s
        #print refval_v


    # coordinate style & center
    # coordinate center
    if coord_center:
        # ra, dec
        refra, refdec = coord_center.split(' ')
        ref           = SkyCoord(refra, refdec, frame='icrs')
        refra_deg     = ref.ra.degree                     # in degree
        refdec_deg    = ref.dec.degree                    # in degree
        cc            = np.array([refra_deg, refdec_deg]) # absolute coordinate of image center
    else:
        refra_deg  = alpha_0
        refdec_deg = delta_0
        cc         = np.array([refra_deg, refdec_deg]) # absolute coordinate of image center

    # coordinate style
    if relativecoords:
        offra  = alpha_0 + (refra_deg-alpha_0)*np.cos(np.radians(refdec_deg))
        offdec = refdec_deg
        arcsec = True
        cc     = np.array([0.,0.])

        xlabel = 'RA offset (arcsec)'
        ylabel = 'DEC offset (arcsec)'
    else:
        print ('WARNING\tchannelmap: Abusolute coordinates are still in development.\
         RA (deg) value is rescaled and not correct.\n\
         relativecoords = True is recommended.')
        offra  = 0.
        offdec = 0.
        arcsec = False
        # units of axes are in degree
        # setting parameters used to plot
        xlabel = label_i[0]
        ylabel = label_i[1]


    # check data axes
    if naxis == 3:
        pass
    elif naxis == 4:
        data = data[0,:,:,:]
    else:
        print ('Error\tchannelmap: Input fits shape is not corrected.\
            It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
        return


    # unit: arcsec or deg
    if arcsec:
        xx     = xx*3600.
        yy     = yy*3600.
        offra  = offra*3600.
        offdec = offdec*3600.
        bmaj   = bmaj*3600.
        bmin   = bmin*3600.
        cc     = cc*3600.

    # figure extent
    xmin   = xx[0,0] - offra
    xmax   = xx[-1,-1] - offra
    ymin   = yy[0,0] - offdec
    ymax   = yy[-1,-1] - offdec
    del_x  = xx[1,1] - xx[0,0]
    del_y  = yy[1,1] - yy[0,0]
    extent = (xmin-0.5*del_x, xmax+0.5*del_x, ymin-0.5*del_y, ymax+0.5*del_y)
    #print extent

    # image scale
    if len(imscale) == 0:
        figxmin, figxmax, figymin, figymax = extent  # arcsec

        if figxmin < figxmax:
            cp = figxmax
            figxmax = figxmin
            figxmin = cp

        if figymin > figymax:
            cp = figymax
            figymax = figymin
            figymin = cp
    elif len(imscale) == 4:
        figxmax, figxmin, figymin, figymax = imscale # arcsec

        if arcsec:
            pass
        else:
            figxmin = figxmin/3600. # arcsec --> degree
            figxmax = figxmax/3600.
            figymin = figymin/3600.
            figymax = figymax/3600.

        # set cc as image center
        figxmin = figxmin + cc[0]
        figxmax = figxmax + cc[0]
        figymin = figymin + cc[1]
        figymax = figymax + cc[1]
    else:
        print ('ERROR\tchannelmap: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')


    # setting velocity axis in relative velocity
    if vsys:
        refval_v = refval_v - vsys
        #print refval_v


    if del_v < 0:
        del_v = - del_v
        data  = data[::-1,:,:]
        refpix_v = nchan - refpix_v + 1

    # set colorscale
    if vmax:
        pass
    else:
        vmax = np.nanmax(data)

    # logscale
    if logscale:
        norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


    # plot
    # setting figure
    plt.rcParams['font.size'] = csize

    if grid:
    	pass
    else:
    	fig = plt.figure(figsize=figsize)
    	# setting colorbar
    	if cbaron:
        	cbar_mode = 'single'
    	else:
        	cbar_mode= None

	    # setting grid
    	cbar_loc, cbar_wd, cbar_pad = cbaroptions
    	grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
    		axes_pad=0,share_all=True,cbar_mode=cbar_mode,
            cbar_location=cbar_loc,cbar_size=cbar_wd,cbar_pad=cbar_pad)

    # setting parameters used to plot
    if len(imscale) == 1:
        figxmin, figxmax, figymin, figymax = extent
    elif len(imscale) == 4:
        figxmax, figxmin, figymin, figymax = imscale
    else:
        print ('ERROR\tchannelmap: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')
    i, j, gridi = [0,0,0]
    gridimax    = nrow*ncol-1

    nroop    = nchan//nskip + 1
    for k in range(0,nroop):
        ichan = k*nskip
        if ichan >= nchan:
            continue
        # select channel
        dataim = data[ichan,:,:]

        # velocity at nchan
        vnchan = refval_v + (ichan + 1 - refpix_v)*del_v

        # check whether vnchan in setted velocity range
        if velmax is not None:
            if vnchan < velmin or vnchan > velmax:
                continue
        elif velmin is not None:
            if vnchan < velmin:
                continue
        else:
            pass

        # each plot
        #ax  = fig.add_subplot(gs[i,j])
        ax = grid[gridi]
        print ('channel ', '%s'%ichan, ', velocity: ', '%4.2f'%vnchan, ' km/s')

        # showing in color scale
        if color:
            imcolor = ax.imshow(dataim, cmap=cmap, origin='lower', extent=extent,norm=norm, rasterized=True)

        if contour:
            imcont  = ax.contour(dataim, colors=ccolor, origin='lower',extent=extent, levels=clevels, linewidths=0.5)

        # set axes
        ax.set_xlim(figxmin,figxmax)
        ax.set_ylim(figymin,figymax)
        ax.spines["bottom"].set_color(axiscolor)
        ax.spines["top"].set_color(axiscolor)
        ax.spines["left"].set_color(axiscolor)
        ax.spines["right"].set_color(axiscolor)
        if xticks != np.empty and yticks != np.empty:
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
        else:
            pass
        ax.set_aspect(1)
        ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, color=tickcolor, labelcolor=labelcolor, pad=9, labelsize=csize)

        # velocity
        #vlabel = AnchoredText('%03.2f'%vnchan,loc=2,frameon=False)
        #ax.add_artist(vlabel)
        vlabel = '%3.2f'%vnchan
        ax.text(0.1, 0.9,vlabel,color=txtcolor,size=csize,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)

        # only on the bottom corner pannel
        if i == nrow-1 and j == 0:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.xaxis.label.set_color(labelcolor)
            ax.yaxis.label.set_color(labelcolor)

            # plot beam size
            #beam_test = patches.Ellipse(xy=(5, -5), width=bmin, height=bmaj, fc='red', angle=-bpa, alpha=0.5)
            #ax.add_patch(beam_test)
            bmin_plot, bmaj_plot = ax.transLimits.transform((0,bmaj)) - ax.transLimits.transform((bmin,0))   # data --> Axes coordinate
            beam = patches.Ellipse(xy=(0.1, 0.1), width=bmin_plot, height=bmaj_plot, fc=bcolor, angle=bpa, transform=ax.transAxes)
            ax.add_patch(beam)

            # scale bar
            if len(scalebar) == 0:
                pass
            elif len(scalebar) == 8:
                barx, bary, barlength, textx, texty, text, barcolor, barcsize = scalebar

                barx      = float(barx)
                bary      = float(bary)
                barlength = float(barlength)
                textx     = float(textx)
                texty     = float(texty)

                if sbar_vertical:
                    ax.vlines(barx, bary - barlength*0.5,bary + barlength*0.5, color=barcolor, lw=2, zorder=10)
                else:
                    ax.hlines(bary, barx - barlength*0.5,barx + barlength*0.5, color=barcolor, lw=2, zorder=10)

                #scale   = patches.Arc(xy=(barx,bary), width=barlength, height=0.001, lw=2, color=colors,zorder=10)
                #ax.add_patch(scale)
                ax.text(textx,texty,text,color=barcolor,fontsize=barcsize,horizontalalignment='center',verticalalignment='center')
            else:
                print ('scalebar must consist of 8 elements. Check scalebar.')
        else:
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        # central star position
        if cstar:
            ll,lw, cl = prop_star
            ll = float(ll)
            lw = float(lw)

            cross01 = patches.Arc(xy=(0,0), width=ll, height=0.001, lw=lw, color=cl, zorder=11)
            cross02 = patches.Arc(xy=(0,0), width=0.001, height=ll, lw=lw, color=cl, zorder=12)
            ax.add_patch(cross01)
            ax.add_patch(cross02)

        # counts
        j     = j + 1
        gridi = gridi+1

        if j == ncol:
            j = 0
            i = i + 1


        if i == nrow:
            #gs.tight_layout(fig,h_pad=0,w_pad=0)
            #plt.subplots_adjust(wspace=0., hspace=0.)
            if color and cbaron:
                # With cbar_mode="single", cax attribute of all axes are identical.
                cbar = ax.cax.colorbar(imcolor,ticks=cbarticks)
                ax.cax.toggle_label(True)
                cbar.ax.yaxis.set_tick_params(color=tickcolor) # tick color
                cbar.ax.spines["bottom"].set_color(axiscolor)  # axes color
                cbar.ax.spines["top"].set_color(axiscolor)
                cbar.ax.spines["left"].set_color(axiscolor)
                cbar.ax.spines["right"].set_color(axiscolor)
                if cbarlabel:
                    cbar.ax.set_ylabel(cbarlabel, color=labelcolor) # label

            plt.savefig(outfile, transparent = True)
            #plt.clf()
            #nmap      = nmap+1
            #outfile   = outname + '_nmap{0:02d}'.format(nmap) + '.' + outformat
            #fig       = plt.figure(figsize=figsize)
            #grid      = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),axes_pad=0,share_all=True,cbar_mode=cbar_mode)
            #i,j,gridi = [0,0,0]

    if color:
        # With cbar_mode="single", cax attribute of all axes are identical.
        cbar = ax.cax.colorbar(imcolor,ticks=cbarticks)
        ax.cax.toggle_label(True)
        cbar.ax.yaxis.set_tick_params(color=tickcolor) # tick color
        cbar.ax.spines["bottom"].set_color(axiscolor)  # axes color
        cbar.ax.spines["top"].set_color(axiscolor)
        cbar.ax.spines["left"].set_color(axiscolor)
        cbar.ax.spines["right"].set_color(axiscolor)
        if cbarlabel:
            cbar.ax.set_ylabel(cbarlabel,color=labelcolor) # label

    if gridi != gridimax+1 and gridi != 0:
        while gridi != gridimax+1:
            #print gridi
            ax = grid[gridi]
            ax.spines["right"].set_color("none")  # right
            ax.spines["left"].set_color("none")   # left
            ax.spines["top"].set_color("none")    # top
            ax.spines["bottom"].set_color("none") # bottom
            ax.axis('off')
            gridi = gridi+1

        #gs.tight_layout(fig,h_pad=0,w_pad=0)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(outfile, transparent = True)

    '''
    # convert N maps into one map
    # ImageMagick is needed
    try:
        outfile = outname + '_nmap*' + '.' + outformat
        sumfile = outname + '.pdf'
        cmdop   = ' -density 600x600 '
        cmd = 'convert' + cmdop + outfile + ' ' + sumfile
        subprocess.call(cmd,shell=True)
    except:
        print 'Cannot convert N maps into one pdf file.'
        print 'Install ImageMagick if you want.'
    '''

    return grid




### channel map
def mltichannelmap(fits01, fits02, outname=None, outformat='pdf', imscale=None, cmap='Greys', vmin=None, vmax=None,
                imoption='clcn', clevels=np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]), ccolor='k',
                nrow=5, ncol=5,velmin=None, velmax=None, nskip=1,colorbar=True,
                xticks=np.empty(0), yticks=np.empty(0), relativecoords=True, vsys=None, csize=9, scalebar=np.empty,
                cstar=True, locsym=0.2, logscale=False, tickcolor='k',axiscolor='k',labelcolor='k',cbarlabel=None, txtcolor='k',
                bcolor='k', figsize=(11.69,8.27), ccolor2='red', lw2=0.5,alpha2=1, prop_star=np.array(['1','0.5','red'])):
    '''
    Make channel maps from images.

    Args
    fitsdata: Input fitsdata. It must be an image cube having 3 or 4 axes.
    imtype (str): Choose which image will be drawn with contour or color. 'cl' means color, 'cn' means contour.
    outname: Output file name. Not including file extension.
    outformat: Extension of the output file. Default is eps.
    imscale: scale to be shown (arcsec). It must be given as [xmin, xmax, ymin, ymax].
    color (bool): If True, images will be shown in colorscale. Default is False.
        cmap: color of the colorscale.
        vmin: Minimum value of colorscale. Default is None.
        vmax: Maximum value of colorscale. Default is the maximum value of the image cube.
        logscale (bool): If True, the color will be shown in logscale.
    contour (bool): If True, images will be shown with contour. Default is True.
        clevels (ndarray): Contour levels. Input will be treated as absolute values.
        ccolor: color of contour.
    nrow, ncol: the number of row and column of the channel map.
    relativecoords (bool): If True, the channel map will be produced in relative coordinate. Abusolute coordinate mode is (probably) coming soon.
    velmin, velmax: Minimum and maximum velocity to be shown.
    vsys: Systemic velicity [km/s]. If no input value, velocities will be described in LSRK.
    csize: Caracter size. Default is 9.
    cstar: If True, a central star or the center of an image will be shown as a cross.
    locsym: A factor to decide locations of symbols (beam and velocity label). It must be 0 - 1.
    tickcolor, axiscolor, labelcolor, txtcolor: Colors for each stuffs.
    scalebar (array): If it is given, scalebar will be drawn. It must be given as [barx, bary, bar length, textx, texty, text].
                       Barx, bary, textx, and texty are locations of a scalebar and a text in arcsec.
    '''

    ### setting output file name & format
    #nmap = 1
    if (outformat == formatlist).any():
        #outfile = outname + '_nmap{0:02d}'.format(nmap) + '.' + outformat
        outfile   = outname + '.' + outformat
    else:
        print ('ERROR\tsingleim_to_fig: Outformat is wrong.')
        return


    ### reading fits files
    data01, hd01 = fits.getdata(fits01,header=True)
    data02, hd02 = fits.getdata(fits02,header=True)

    # reading hd01 info.
    xlabel   = hd01['CTYPE1']
    ylabel   = hd01['CTYPE2']
    vlabel   = hd01['CTYPE3']
    try:
        restfreq = hd01['RESTFRQ'] # Hz
    except:
        restfreq = hd01['RESTFREQ'] # Hz
    refval_x = hd01['CRVAL1']*60.*60. # deg --> arcsec
    refval_y = hd01['CRVAL2']*60.*60.
    refval_v = hd01['CRVAL3']
    refpix_x = int(hd01['CRPIX1'])
    refpix_y = int(hd01['CRPIX2'])
    refpix_v = int(hd01['CRPIX3'])
    del_x    = hd01['CDELT1']*60.*60. # deg --> arcsec
    del_y    = hd01['CDELT2']*60.*60.
    del_v    = hd01['CDELT3']
    nx       = hd01['NAXIS1']
    ny       = hd01['NAXIS2']
    nchan    = hd01['NAXIS3']
    bmaj     = hd01['BMAJ']*60.*60.
    bmin     = hd01['BMIN']*60.*60.
    bpa      = hd01['BPA']  # [deg]
    unit     = hd01['BUNIT']
    print ('x, y axes are ', xlabel, ' and ', ylabel)


    # frequency --> velocity
    if vlabel == 'VRAD':
        print ('The third axis is ', vlabel)
        del_v = del_v*1.e-3
        refval_v = refval_v*1.e-3
    else:
        print ('The third axis is [FREQUENCY]')
        print ('Convert frequency to velocity')
        del_v    = - del_v*clight/restfreq       # delf --> delv [cm/s]
        del_v    = del_v*1.e-5                   # cm/s --> km/s
        refval_v = clight*(1.-refval_v/restfreq) # radio velocity c*(1-f/f0) [cm/s]
        refval_v = refval_v*1.e-5                # cm/s --> km/s
        #print refval_v


    # setting axes in relative coordinate
    if relativecoords:
        refval_x, refval_y = [0,0]
        xlabel = 'RA offset (arcsec; J2000)'
        ylabel = 'Dec offset (arcsec; J2000)'
    else:
        pass
    xmin = refval_x + (1 - refpix_x)*del_x - 0.5*del_x
    xmax = refval_x + (nx - refpix_x)*del_x + 0.5*del_x
    ymin = refval_y + (1 - refpix_y)*del_y - 0.5*del_y
    ymax = refval_y + (ny - refpix_y)*del_y + 0.5*del_y
    extent01 = (xmin,xmax,ymin,ymax)

    # setting velocity axis in relative velocity
    if vsys:
        refval_v = refval_v - vsys
        #print refval_v



    # reading hd02 info.
    xlabel02   = hd02['CTYPE1']
    ylabel02   = hd02['CTYPE2']
    vlabel02   = hd02['CTYPE3']
    try:
        restfreq02 = hd02['RESTFRQ'] # Hz
    except:
        restfreq02 = hd02['RESTFREQ'] # Hz
    refval_x02 = hd02['CRVAL1']*60.*60. # deg --> arcsec
    refval_y02 = hd02['CRVAL2']*60.*60.
    refval_v02 = hd02['CRVAL3']
    refpix_x02 = int(hd02['CRPIX1'])
    refpix_y02 = int(hd02['CRPIX2'])
    refpix_v02 = int(hd02['CRPIX3'])
    del_x02    = hd02['CDELT1']*60.*60. # deg --> arcsec
    del_y02    = hd02['CDELT2']*60.*60.
    del_v02    = hd02['CDELT3']
    nx02       = hd02['NAXIS1']
    ny02       = hd02['NAXIS2']
    nchan02    = hd02['NAXIS3']
    bmaj02     = hd02['BMAJ']*60.*60.
    bmin02     = hd02['BMIN']*60.*60.
    bpa02      = hd02['BPA']  # [deg]
    unit02     = hd02['BUNIT']
    print ('x, y axes are ', xlabel02, ' and ', ylabel02)


    # frequency --> velocity
    if vlabel02 == 'VRAD':
        print ('The third axis is ', vlabel)
        del_v02 = del_v02*1.e-3
        refval_v02 = refval_v02*1.e-3
    else:
        print ('The third axis is [FREQUENCY]')
        print ('Convert frequency to velocity')
        del_v02    = - del_v02*clight/restfreq02       # delf --> delv [cm/s]
        del_v02    = del_v02*1.e-5                     # cm/s --> km/s
        refval_v02 = clight*(1.-refval_v02/restfreq02) # radio velocity c*(1-f/f0) [cm/s]
        refval_v02 = refval_v02*1.e-5                  # cm/s --> km/s
        #print refval_v


    # setting axes in relative coordinate
    if relativecoords:
        refval_x02, refval_y02 = [0,0]
        xlabel02 = 'RA offset (arcsec; J2000)'
        ylabel02 = 'Dec offset (arcsec; J2000)'
    else:
        pass
    xmin02 = refval_x02 + (1 - refpix_x02)*del_x02 - 0.5*del_x02
    xmax02 = refval_x02 + (nx02 - refpix_x02)*del_x02 + 0.5*del_x02
    ymin02 = refval_y02 + (1 - refpix_y02)*del_y02 - 0.5*del_y02
    ymax02 = refval_y02 + (ny02 - refpix_y02)*del_y02 + 0.5*del_y02
    extent02 = (xmin02,xmax02,ymin02,ymax02)


    # check data axes
    if len(data01.shape) == 4:
        pass
    elif len(data01.shape) == 3:
        data01 = data01.reshape((1,nchan,nx,ny))
    else:
        print ('Error\tsingleim_to_fig: Input fits size is not corrected.\
         It is allowed only to have 4 axes. Check the shape of the fits file.')
        return

    # check data axes
    if len(data02.shape) == 4:
        pass
    elif len(data02.shape) == 3:
        data02 = data02.reshape((1,nchan02,nx02,ny02))
    else:
        print ('Error\tsingleim_to_fig: Input fits size is not corrected.\
         It is allowed only to have 4 axes. Check the shape of the fits file.')
        return

    # set colorscale
    if vmax:
        pass
    else:
        vmax = np.nanmax(data01[0,:,:,:])

    # logscale
    if logscale:
        norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)

    ### ploting
    # setting figure
    plt.rcParams['font.size'] = csize
    fig = plt.figure(figsize=figsize)
    # setting colorbar
    if colorbar:
        cbar_mode = 'single'
    else:
        cbar_mode = None

    # setting grid
    grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
            axes_pad=0,share_all=True,cbar_mode=cbar_mode)

    # setting parameters used to plot
    i, j, gridi = [0,0,0]
    gridimax    = nrow*ncol-1
    figxmin, figxmax, figymin, figymax = imscale


    nroop = nchan//nskip + 1
    for k in range(0,nroop):
        # select channel
        ichan   = k*nskip
        if ichan >= nchan:
            break

        # velocity at nchan
        vnchan = refval_v + (ichan + 1 - refpix_v)*del_v

        # check whether vnchan in setted velocity range
        if velmax:
            if vnchan < velmin or vnchan > velmax:
                continue
        elif velmin:
            if vnchan < velmin:
                continue
        else:
            pass

        # each plot
        #ax  = fig.add_subplot(gs[i,j])
        ax = grid[gridi]
        print ('channel ', '%s'%ichan, ', velocity: ', '%4.2f'%vnchan, ' km/s')


        if imoption == 'clcn':
            color   = data01[0,ichan,:,:]
            contour = data02[0,ichan,:,:]
            # showing in color scale
            imcolor = ax.imshow(color, cmap=cmap, origin='lower', extent=extent01,norm=norm)
            imcont  = ax.contour(contour, colors=ccolor, origin='lower',extent=extent02, levels=clevels, linewidths=0.5)
        elif imoption == 'cncl':
            color   = data02[0,ichan,:,:]
            contour = data01[0,ichan,:,:]
            imcolor = ax.imshow(color, cmap=cmap, origin='lower', extent=extent02,norm=norm)
            imcont  = ax.contour(contour, colors=ccolor, origin='lower',extent=extent01, levels=clevels, linewidths=0.5)
        elif imoption == 'cncn':
            contour01 = data01[0,ichan,:,:]
            contour02 = data02[0,ichan,:,:]
            imcont01  = ax.contour(contour01, colors=ccolor, origin='lower',extent=extent01, levels=clevels, linewidths=0.5)
            imcont02  = ax.contour(contour02, colors=ccolor2, origin='lower',extent=extent02, levels=clevels, linewidths=lw2,alpha=alpha2)
            colorbar  = False

        if colorbar:
            cbar = ax.cax.colorbar(imcolor)
            ax.cax.toggle_label(True)
            cbar.ax.yaxis.set_tick_params(color=tickcolor) # tick color
            cbar.ax.spines["bottom"].set_color(axiscolor)  # axes color
            cbar.ax.spines["top"].set_color(axiscolor)
            cbar.ax.spines["left"].set_color(axiscolor)
            cbar.ax.spines["right"].set_color(axiscolor)
            if cbarlabel:
                cbar.ax.set_ylabel(cbarlabel, color=labelcolor) # label
            colorbar = False

        # set axes
        ax.set_xlim(figxmax,figxmin)
        ax.set_ylim(figymin,figymax)
        ax.spines["bottom"].set_color(axiscolor)
        ax.spines["top"].set_color(axiscolor)
        ax.spines["left"].set_color(axiscolor)
        ax.spines["right"].set_color(axiscolor)
        if len(xticks) == 0 and len(yticks) == 0:
            pass
        elif len(xticks) != 0 and len(yticks) == 0:
            ax.set_xticks(xticks)
        elif len(xticks) == 0 and len(yticks) != 0:
            ax.set_yticks(yticks)
        else:
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
        ax.set_aspect(1)
        ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, color=tickcolor, labelcolor=labelcolor, pad=9, labelsize=csize)

        # velocity
        #vlabel = AnchoredText('%03.2f'%vnchan,loc=2,frameon=False)
        #ax.add_artist(vlabel)
        vlabel = '%03.2f'%vnchan
        ax.text(0.1, 0.9,vlabel,color=txtcolor,size=csize,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)

        # only on the bottom corner pannel
        if i == nrow-1 and j == 0:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.xaxis.label.set_color(labelcolor)
            ax.yaxis.label.set_color(labelcolor)

            # plot beam size
            beam = patches.Ellipse(xy=(figxmax-locsym*figxmax, figymin-locsym*figymin), width=bmin, height=bmaj, fc=bcolor, angle=-bpa)
            ax.add_patch(beam)
            #beam = AnchoredEllipse(ax.transData,width=bmin, height=bmaj, angle=-bpa,loc=3, pad=0.5, borderpad=0.4, frameon=False)
            #ax.add_artist(beam)

            # scale bar
            if scalebar is np.empty:
                pass
            else:
                barx, bary, barlength = scalebar[0,1,2]
                textx, texty, text    = scalebar[3,4,5]

                barx      = float(barx)
                bary      = float(bary)
                barlength = float(barlength)
                textx     = float(textx)
                texty     = float(texty)

                scale   = patches.Arc(xy=(barx,bary), width=barlength, height=0., lw=2, color='k',zorder=10)
                ax.add_patch(scale)
                ax.text(textx,texty,text)
        else:
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        # central star position
        if cstar:
            ll,lw, cl = prop_star
            ll = float(ll)
            lw = float(lw)

            cross01 = patches.Arc(xy=(refval_x,refval_y), width=ll, height=0.001, lw=lw, color=cl, zorder=11)
            cross02 = patches.Arc(xy=(refval_x,refval_y), width=0.001, height=ll, lw=lw, color=cl, zorder=12)
            ax.add_patch(cross01)
            ax.add_patch(cross02)

        # counts
        j     = j + 1
        gridi = gridi+1

        if j == ncol:
            j = 0
            i = i + 1

        if i == nrow:
            fig.savefig(outfile, transparent = True)
            fig.clf()
            #nmap      = nmap+1
            #outfile   = outname + '_nmap{0:02d}'.format(nmap) + '.' + outformat
            fig       = plt.figure(figsize=figsize)
            grid      = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
                axes_pad=0,share_all=True,cbar_mode=cbar_mode)
            i,j,gridi = [0,0,0]


    '''
    # With cbar_mode="single", cax attribute of all axes are identical.
    cbar = ax.cax.colorbar(imcolor)
    ax.cax.toggle_label(True)
    cbar.ax.yaxis.set_tick_params(color=tickcolor) # tick color
    cbar.ax.spines["bottom"].set_color(axiscolor)  # axes color
    cbar.ax.spines["top"].set_color(axiscolor)
    cbar.ax.spines["left"].set_color(axiscolor)
    cbar.ax.spines["right"].set_color(axiscolor)
    if cbarlabel:
        cbar.ax.set_ylabel(cbarlabel,color=labelcolor) # label
    '''

    if gridi != gridimax+1 and gridi != 0:
        while gridi != gridimax+1:
            #print gridi
            ax = grid[gridi]
            ax.spines["right"].set_color("none")  # right
            ax.spines["left"].set_color("none")   # left
            ax.spines["top"].set_color("none")    # top
            ax.spines["bottom"].set_color("none") # bottom
            ax.axis('off')
            gridi = gridi+1

        #gs.tight_layout(fig,h_pad=0,w_pad=0)
        #plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(outfile, transparent = True)

    '''
    # convert N maps into one map
    # ImageMagick is needed
    try:
        outfile = outname + '_nmap*' + '.' + outformat
        sumfile = outname + '.pdf'
        cmdop   = ' -density 600x600 '
        cmd = 'convert' + cmdop + outfile + ' ' + sumfile
        subprocess.call(cmd,shell=True)
    except:
        print 'Cannot convert N maps into one pdf file.'
        print 'Install ImageMagick if you want.'
    '''

    return ax



### PV diagram
def pvdiagram(self,outname,data=None,header=None,ax=None,outformat='pdf',color=True,cmap='Greys',
    vmin=None,vmax=None,vsys=None,contour=True,clevels=None,ccolor='k',
    vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
    lw=1,clip=None,plot_res=True,inmode='fits'):

    # setting for figures
    plt.rcParams['font.size'] = fontsize           # fontsize

    # output file
    if (outformat == formatlist).any():
        outname = outname + '.' + outformat
    else:
        print ('ERROR\tsingleim_to_fig: Outformat is wrong.')
        return


    # reading fits data
    if inmode == 'fits':
        data, header = fits.getdata(self,header=True)
    elif inmode == 'data':
        if data is None:
            print ("inmode ='data' is selected. data must be provided.")
            return
        elif header is None:
            print ("inmode ='data' is selected. header must be provided.")
            return
    else:
        print ("inmode is incorrect. Must be choosen from 'fits' or 'data'.")


    # figures
    if ax:
        pass
    else:
        fig = plt.figure(figsize=(11.69,8.27)) # figsize=(11.69,8.27)
        ax  = fig.add_subplot(111)


    # header info.
    naxis      = int(header['NAXIS'])
    noff       = int(header['NAXIS1'])
    nvel       = int(header['NAXIS2'])
    bmaj       = header['BMAJ']*60.*60. # in arcsec
    bmin       = header['BMIN']*60.*60. # in arcsec
    bpa        = header['BPA']          # [deg]
    offlabel   = header['CTYPE1']
    vellabel   = header['CTYPE2']
    thirdlabel = header['BUNIT']
    offunit    = header['CUNIT1']
    restfreq   = header['RESTFRQ'] # Hz
    refval_off = header['CRVAL1']  # in arcsec
    refval_vel = header['CRVAL2']
    refval_vel = clight*(restfreq - refval_vel)/restfreq # Hz --> radio velocity [cm/s]
    refval_vel = refval_vel*1.e-5         # cm/s --> km/s
    refpix_off = header['CRPIX1']
    refpix_vel = header['CRPIX2']
    del_off    = header['CDELT1']  # in arcsec
    del_vel    = header['CDELT2']
    del_vel    = - clight*del_vel/restfreq # Hz --> cm/s
    del_vel    = del_vel*1.e-5             # cm/s --> km/s
    #print refval_vel, del_vel
    #print refval_off, del_off


    # check unit
    if offunit == 'degree' or offunit == 'deg':
        refval_off = refval_off*60.*60.
        del_off    = del_off*60.*60.


    # relative velocity or LSRK
    offlabel = 'offset (arcsec)'
    if vrel:
        refval_vel = refval_vel - vsys
        vellabel   = 'relative velocity (km/s)'
        vcenter    = 0
    else:
        vellabel = 'LSRK velocity (km/s)'
        vcenter  = vsys


    # set extent of an image
    offmin = refval_off + (1 - refpix_off)*del_off - del_off*0.5
    offmax = refval_off + (noff - refpix_off)*del_off + del_off*0.5
    velmin = refval_vel + (1 - refpix_vel)*del_vel - del_vel*0.5
    velmax = refval_vel + (nvel - refpix_vel)*del_vel +del_vel*0.5



    # set axes
    if x_offset:
        data   = data[0,:,:]
        extent = (offmin,offmax,velmin,velmax)
        xlabel = offlabel
        ylabel = vellabel
        hline_params = [vsys,offmin,offmax]
        vline_params = [0.,velmin,velmax]
        res_x = bmaj
        res_y = del_vel
    else:
        data   = np.rot90(data[0,:,:])
        extent = (velmin,velmax,offmin,offmax)
        xlabel = vellabel
        ylabel = offlabel
        hline_params = [0.,velmin,velmax]
        vline_params = [vcenter,offmin,offmax]
        res_x = del_vel
        res_y = bmaj


    # set colorscale
    if vmax:
        pass
    else:
        vmax = np.nanmax(data)


    # logscale
    if logscale:
        norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


    # clip data at some value
    data_color = copy.copy(data)
    if clip:
        data_color[np.where(data < clip)] = np.nan


    # plot images
    if color:
        imcolor = ax.imshow(data_color, cmap=cmap, origin='lower', extent=extent,norm=norm)

    if contour:
        imcont  = ax.contour(data, colors=ccolor, origin='lower',extent=extent, levels=clevels, linewidths=lw)


    # axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # set xlim, ylim
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])


    # lines showing offset 0 and relative velocity 0
    xline = plt.hlines(hline_params[0], hline_params[1], hline_params[2], ccolor, linestyles='dashed', linewidths = 0.5)
    yline = plt.vlines(vline_params[0], vline_params[1], vline_params[2], ccolor, linestyles='dashed', linewidths = 0.5)
    ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)

    # plot resolutions
    if plot_res:
        # x axis
        #print (res_x, res_y)
        res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5)) -  ax.transLimits.transform((0, 0)) # data --> Axes coordinate
        ax.errorbar(0.1, 0.1, xerr=res_x_plt, yerr=res_y_plt, color=ccolor, capsize=3, capthick=1., elinewidth=1., transform=ax.transAxes)

    # aspect ratio
    change_aspect_ratio(ax, ratio)


    # save figure
    plt.savefig(outname, transparent=True)

    return ax