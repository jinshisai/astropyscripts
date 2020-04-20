### cut and extract image cube within a specified region



### import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import mpl_toolkits.axes_grid1
from scipy import optimize
from scipy import stats
from astrconsts import *
import matplotlib.figure as figure
from scipy.ndimage.interpolation import rotate
import mathfuncs as mf
import imhandle
import pyfigures as pyfg
from scipy import interpolate
from mpl_toolkits.axes_grid1 import ImageGrid
from copy import copy
from astropy.convolution import convolve_fft
import math
import readfits
from astropy.io import fits
from datetime import datetime
import os



### main
def main(self, region, outname=None, values=False):
	# reading fits files
	data, header = fits.getdata(self,header=True)

	# output name
	if outname:
		pass
	else:
		outname = self.replace('.fits', '.croped.fits')

	# reading header info.
	xlabel   = header['CTYPE1']
	ylabel   = header['CTYPE2']
	try:
	    restfreq = header['RESTFRQ']    # Hz
	except:
	    restfreq = header['RESTFREQ']   # Hz
	refval_x = header['CRVAL1']         # deg
	refval_y = header['CRVAL2']
	refval_v = header['CRVAL3']
	refpix_x = int(header['CRPIX1'])
	refpix_y = int(header['CRPIX2'])
	refpix_v = int(header['CRPIX3'])
	del_x    = header['CDELT1']         # deg
	del_y    = header['CDELT2']
	del_v    = header['CDELT3']
	nx       = header['NAXIS1']
	ny       = header['NAXIS2']
	nchan    = header['NAXIS3']
	bmaj     = header['BMAJ']           # deg
	bmin     = header['BMIN']           # deg
	bpa      = header['BPA']            # [deg]
	unit     = header['BUNIT']

	# frequency --> velocity
	#print 'The third axis is [FREQUENCY]'
	#print 'Convert frequency to velocity'
	#del_v    = - del_v*clight/restfreq       # delf --> delv [cm/s]
	#del_v    = del_v*1.e-5                   # cm/s --> km/s
	#refval_v = clight*(1.-refval_v/restfreq) # radio velocity c*(1-f/f0) [cm/s]
	#refval_v = refval_v*1.e-5                # cm/s --> km/s
	#print del_v

	vpixmin, vpixmax, xpixmin, xpixmax, ypixmin, ypixmax = region
	nv_new = vpixmax - vpixmin +1
	nx_new = xpixmax - xpixmin +1
	ny_new = ypixmax - ypixmin +1
	print xpixmax, xpixmin, nx_new

	# axes
	vmin     = refval_v + (1 - refpix_v)*del_v
	vmax     = refval_v + (nchan - refpix_v)*del_v
	xmin     = refval_x + (1 - refpix_x)*del_x
	xmax     = refval_x + (nx - refpix_x)*del_x
	ymin     = refval_y + (1 - refpix_y)*del_y
	ymax     = refval_y + (ny - refpix_y)*del_y
	xaxis    = np.arange(xmin,xmax+del_x,del_x)
	yaxis    = np.arange(ymin,ymax+del_y,del_y)
	vaxis    = np.arange(vmin,vmax+del_v,del_v)

	# crop data
	data   = data[:,vpixmin-1:vpixmax,xpixmin-1:xpixmax,ypixmin-1:ypixmax] # 0 start in python grammer, pixmax is not included
	vaxis  = vaxis[vpixmin-1:vpixmax]
	xaxis  = xaxis[xpixmin-1:xpixmax]
	yaxis  = yaxis[ypixmin-1:ypixmax]
	extent = (xmin - 0.5*del_x, xmax + 0.5*del_x, ymin - 0.5*del_y, ymax+0.5*del_y)


	# output
	hdout = header
	today = datetime.today()
	today = today.strftime("%Y/%m/%d %H:%M:%S")

	# write into new header
	del hdout['HISTORY']
	hdout['DATE']     = (today, 'FITS file was croped')
	hdout['CRPIX1']   = refpix_x - xpixmin +1
	hdout['CRPIX2']   = refpix_y - ypixmin +1
	hdout['CRPIX3']   = 1
	hdout['CRVAL3']   = vaxis[0]

	if os.path.exists(outname):
		os.system('rm -r '+outname)

	print 'writing fits file...'
	fits.writeto(outname, data, header=hdout)

	return


if __name__ == '__main__':
	fitsimage = 'tmc1a.c18o21.contsub.rbp05.mlt100.clean.image.reg.smoothed.fits'
	region    = [7,58,251,750,251,750]
	main(fitsimage, region)