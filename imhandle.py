### import modules
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import mathfuncs as mf
from astropy.io import fits
from datetime import datetime
import os
from scipy.interpolate import griddata
from time import sleep
from tqdm import tqdm
from astrfuncs import IvTOJT
from astrfuncs import TbTOIv



### constants
clight     = 2.99792458e10 # light speed [cm s^-1]



### function
def imrotate(image, angle=0):
	'''
    Rotate the input image

    Args:
  	image: input image in 2d array
  	angle (float): Rotational Angle. Anti-clockwise direction will be positive (same to the Position Angle). in deg.
	'''

	# check whether the array includes nan
	# nan --> 0 for interpolation
	if np.isnan(image).any() == False:
		pass
	elif np.isnan(image).any() == True:
		print ('CAUTION\timrotate: Input image array includes nan. Replace nan with 0 for interpolation when rotate image.')
		image[np.isnan(image)] = 0.

	# rotate image
	nx = image.shape[0]
	ny = image.shape[1]
	newimage = scipy.ndimage.rotate(image, -angle)

	# resampling
	mx = newimage.shape[0]
	my = newimage.shape[1]
	outimage = newimage[my//2 - ny//2:my//2 - ny//2 + ny, mx//2 - nx//2:mx//2 - nx//2 + nx]

	return outimage


def imdeproject(fitsdata, pa=0., inc=0., deg=True):
	'''
	Deprojected image by inclination angle.

	image: input image in fitsdata
	pa: Angle of major axis. Anti-clockwise direction is positive.
	inc: Inclination angle.
	deg (bool): If True pa and inc will be treated in degree.\
	 If False pa and inc will be treated in radian. Default True.
	'''

	### reading fits files
	data, header = fits.getdata(fitsdata,header=True)

    # check data axes
	if len(data.shape) == 3:
		data = data[0,:,:]
	elif len(data.shape) == 4:
		data = data[0,0,:,:]
	else:
		print ('Error\timdeproject: Input fits size is not corrected.\
			It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
		return

	# reading header info.
	xlabel   = header['CTYPE1']
	ylabel   = header['CTYPE2']
	try:
	    restfreq = header['RESTFRQ'] # Hz
	except:
	    restfreq = header['RESTFREQ'] # Hz
	refval_x = header['CRVAL1']*60.*60. # deg --> arcsec
	refval_y = header['CRVAL2']*60.*60.
	refpix_x = int(header['CRPIX1'])
	refpix_y = int(header['CRPIX2'])
	del_x    = header['CDELT1']*60.*60. # deg --> arcsec
	del_y    = header['CDELT2']*60.*60.
	nx       = header['NAXIS1']
	ny       = header['NAXIS2']
	bmaj     = header['BMAJ']*60.*60. # deg --> arcsec
	bmin     = header['BMIN']*60.*60.
	bpa      = header['BPA']  # [deg]
	unit     = header['BUNIT']
	print ('x, y axes are ', xlabel, ' and ', ylabel)

	# degree --> radian
	if deg:
		pa = np.radians(pa)
		inc = np.radians(inc)
	else:
		pass

	tanpa  = np.tan(pa)
	cospa  = np.cos(pa)
	sinpa  = np.sin(pa)
	cosinc = np.cos(inc)

	# image center
	xcent = refpix_x
	ycent = refpix_y

	b = ycent - xcent*tanpa
	return


def circular_slice(image,deltapa=np.linspace(-180,180,10),istokes=0,ifreq=0):
	'''
	Produce figure where position angle (x) vs radius (y) vs intensity (color)

	image: fits data
	deltpa: delta of position angle to cut
	'''

	### reading fits files
	data, header = fits.getdata(image,header=True)

    # check data axes
	if len(data.shape) == 3:
		data = data[istokes,:,:]
	elif len(data.shape) == 4:
		data = data[istokes,ifreq,:,:]
	else:
		print ('Error\tciruclar_slice: Input fits size is not corrected.\
			It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
		return

    # header info.
	nx       = header['NAXIS1']
	refpix_x = int(header['CRPIX1'])
	refpix_y = int(header['CRPIX2'])

	# set parameters
	#Nz  = nx//2
	Nz  = nx - refpix_y
	Npa = len(deltapa)

	# start slicing
	print ('circular slice...')
	cslice = np.zeros([Nz,Npa])
	for ipa in range(Npa):
		rotimage      = imrotate(data,-deltapa[ipa])
		cslice[:,ipa] = rotimage[refpix_y:,refpix_x]

	return cslice



def imcrop(self, region, outname=None, values=False):
	'''
	Crop a given region from an image cube.

	self: fits file
	region: Croped region. Must be given as [vmin, vmax, xmin, xmax, ymin, ymax] in pixel.
	outname: Output fits file name. If not given, '.fits' will be replaced '.croped.fits'.
	values(bool): Now still in development.
	'''
	# notification
	print ('Begin\timcrop')
	print ('CAUTION:\tImage center could be changed. Check the given cropping region carefully.')
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
	bpa      = header['BPA']            # deg
	unit     = header['BUNIT']

	# check data axes
	if len(data.shape) == 3:
		data = data.reshape(1,nchan,ny,nx)
	elif len(data.shape) == 4:
		pass
	else:
		print ('Error\tciruclar_slice: Input fits size is not corrected.\
     	It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
		return

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
	#print xpixmax, xpixmin, nx_new

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
	print ('region')
	print ('x range: %.1f - %.1f arcsec' %((xaxis[0] - refval_x)*60.*60., (xaxis[-1] -refval_x)*60.*60.))
	print ('y range: %.1f - %.1f arcsec' %((yaxis[0] - refval_y)*60.*60., (yaxis[-1] -refval_y)*60.*60.))
	print ('v range: %.2f - %.2f km/s' %( clight*(1.-vaxis[-0]/restfreq)*1e-5, clight*(1.-vaxis[-1]/restfreq)*1e-5))


	# output
	hdout = header
	today = datetime.today()
	today = today.strftime("%Y/%m/%d %H:%M:%S")

	# write into new header
	if 'HISTORY' in hdout:
		del hdout['HISTORY']
	hdout['DATE']     = (today, 'FITS file was croped')
	hdout['CRPIX1']   = refpix_x - xpixmin +1
	hdout['CRPIX2']   = refpix_y - ypixmin +1
	hdout['CRPIX3']   = 1
	hdout['CRVAL3']   = vaxis[0]

	if os.path.exists(outname):
		os.system('rm -r '+outname)

	print ('writing fits file...')
	fits.writeto(outname, data, header=hdout)

	return



def fits_regrid(self, fits_tmp, savefits=True, outname='imregrid.fits', overwrite=False, check_fig = False):
	# regrid fits image using spline trapolation.

	### reading fits files
	# tempreate image
	data_tmp, header = fits.getdata(fits_tmp,header=True)

	# reading header info.
	xlabel   = header['CTYPE1']
	ylabel   = header['CTYPE2']
	try:
	    restfreq = header['RESTFRQ'] # Hz
	except:
	    restfreq = header['RESTFREQ'] # Hz
	refval_x = header['CRVAL1']       # deg
	refval_y = header['CRVAL2']       # deg
	refval_v = header['CRVAL3']
	refpix_x = int(header['CRPIX1'])
	refpix_y = int(header['CRPIX2'])
	refpix_v = int(header['CRPIX3'])
	del_x    = header['CDELT1']       # deg
	del_y    = header['CDELT2']       # deg
	del_v    = header['CDELT3']
	nx       = header['NAXIS1']
	ny       = header['NAXIS2']
	nchan    = header['NAXIS3']
	bmaj     = header['BMAJ'] # deg
	bmin     = header['BMIN'] # deg
	bpa      = header['BPA']  # deg
	unit     = header['BUNIT']
	#print 'x, y axes are ', xlabel, ' and ', ylabel


	# center of corner pixels
	xmin = refval_x + (1 - refpix_x )*del_x
	xmax = refval_x + (nx - refpix_x)*del_x
	ymin = refval_y + (1 - refpix_y)*del_y
	ymax = refval_y + (ny - refpix_y)*del_y
	vmin = refval_v + (1 - refpix_v)*del_v
	vmax = refval_v + (nchan - refpix_v)*del_v
	#print xmin, xmax, ymin, ymax

	# grid
	xc             = np.linspace(xmin,xmax,nx)
	yc             = np.linspace(ymin,ymax,ny)
	vc             = np.linspace(vmin,vmax,nchan)
	xx_tmp, yy_tmp = np.meshgrid(xc, yc)
	#xx_tmp, yy_tmp, vv_temp = np.meshgrid(xc, yc, vc)


	# image which will be regrided
	data, hd2 = fits.getdata(self,header=True)
	dataxysize = data[0,0,:,:].size
	dataxyvsize = data[0,:,:,:].size

	# reading header info.
	xlabel2   = hd2['CTYPE1']
	ylabel2   = hd2['CTYPE2']
	try:
	    restfreq2 = hd2['RESTFRQ'] # Hz
	except:
	    restfreq2 = hd2['RESTFREQ'] # Hz
	refval_x2 = hd2['CRVAL1']       # deg
	refval_y2 = hd2['CRVAL2']       # deg
	refval_v2 = hd2['CRVAL3']
	refpix_x2 = int(hd2['CRPIX1'])
	refpix_y2 = int(hd2['CRPIX2'])
	refpix_v2 = int(hd2['CRPIX3'])
	del_x2    = hd2['CDELT1']       # deg
	del_y2    = hd2['CDELT2']       # deg
	del_v2    = hd2['CDELT3']
	nx2       = hd2['NAXIS1']
	ny2       = hd2['NAXIS2']
	nchan2    = hd2['NAXIS3']
	bmaj2     = hd2['BMAJ']         # deg
	bmin2     = hd2['BMIN']         # deg
	bpa2      = hd2['BPA']          # deg
	unit2     = hd2['BUNIT']
	#print 'x, y axes are ', xlabel2, ' and ', ylabel2


	# center of corner pixels
	xmin2 = refval_x2 + (1 - refpix_x2)*del_x2
	xmax2 = refval_x2 + (nx2 - refpix_x2)*del_x2
	ymin2 = refval_y2 + (1 - refpix_y2)*del_y2
	ymax2 = refval_y2 + (ny2 - refpix_y2)*del_y2
	vmin2 = refval_v2 + (1 - refpix_v2)*del_v2
	vmax2 = refval_v2 + (nchan2 - refpix_v2)*del_v2
	#print xmin2, xmax2, ymin2, ymax2


	# grid
	xc2     = np.linspace(xmin2,xmax2,nx2)
	yc2     = np.linspace(ymin2,ymax2,ny2)
	vc2     = np.linspace(vmin2,vmax2,nchan2)
	xx_reg, yy_reg = np.meshgrid(xc2, yc2)
	#xx_reg, yy_reg, vv_reg = np.meshgrid(xc2, yc2, vc2)

	# nan --> 0
	data[np.where(np.isnan(data))] = 0.
	#print np.max(data)

	# 2D --> 1D
	xinp    = xx_reg.reshape(xx_reg.size)
	yinp    = yy_reg.reshape(yy_reg.size)
	#vinp    = vv_reg.reshape(vv_reg.size)

	# output array
	data_out = np.zeros(data_tmp.shape)

	# regrid
	print ('regriding...')
	'''
	# roop
	for ichan in xrange(nchan2):
		dinp = data[0,ichan,:,:].reshape(data[0,0,:,:].size)

		# regrid
		print 'channle: %.0f/%.0f'%(ichan+1,nchan2)

		data_reg = griddata((xinp, yinp), dinp, (xx_tmp, yy_tmp), method='cubic',rescale=True)
		data_reg = data_reg.reshape((1,1,ny,nx))
		data_out[0,ichan,:,:] = data_reg
	'''

	# internal expression
	channel_range = tqdm(range(nchan2))
	data_reg = np.array([ griddata((xinp, yinp), data[0,i,:,:].reshape(dataxysize), (xx_tmp, yy_tmp), method='cubic',rescale=True) for i in channel_range ])
	#data_reg = griddata((xinp, yinp, vinp), data[0,:,:,:].reshape(dataxyvsize), (xx_tmp, yy_tmp, vv_temp), method='linear',rescale=True)
	data_reg = data_reg.reshape((data_tmp.shape))

	# new header
	hd_new = header
	try:
		hd_new['RESTFRQ'] = restfreq2 # Hz
	except:
		hd_new['RESTFREQ'] = restfreq2 # Hz

	hd_new['BMAJ'] = hd2['BMAJ']
	hd_new['BMIN'] = hd2['BMIN']
	hd_new['BPA']  = hd2['BPA']

	# write fits
	if savefits:
		if overwrite:
			try:
				os.system('rm ' +outname)
			except:
				pass
		fits.writeto(outname, data_reg, header=hd_new)


	# check plot
	# figure
	if check_fig:
		fig = plt.figure(figsize=(11.69, 8.27))
		ax  = fig.add_subplot(111)

		# image size
		figxmax, figxmin, figymin, figymax = imscale


		# images
		#imcolor   = ax.imshow(redata[0,0,:,:], cmap=cmap, origin='lower', extent=(xmin, xmax, ymin, ymax))
		imcontour     = ax.contour(data[0,0,:,:], origin='lower', colors='k', levels=clevels, linewidths=3., extent=(xmin2, xmax2, ymin2, ymax2), alpha=0.5)
		imreg_contour = ax.contour(data_reg[0,0,:,:], origin='lower', colors='red', levels=clevels, linewidths=1., extent=(xmin, xmax, ymin, ymax), alpha=0.5)

		# images for test
		#imrot     = ax.contour(datarot, origin='lower', colors='cyan', levels=clevels, extent=(xmin, xmax, ymin, ymax))
		#imcontour = ax.contour(xx, yy, data[0,0,:,:], origin='lower', colors='white', levels=clevels)
		#ax.contour(xxparot, yyinc, data[0,0,:,:], origin='lower', colors='cyan', levels=clevels, extent=(np.nanmax(xxparot), np.nanmin(xxparot), np.nanmin(yyinc), np.nanmax(yyinc)))


		# plot beam size
		beam    = patches.Ellipse(xy=(0.9,-0.9),transform=ax.transAxes, width=bmin*60.*60., height=bmaj*60.*60., fc='k', angle=-bpa)
		ax.add_patch(beam)

		# axes
		ax.set_xlim(figxmin,figxmax)
		ax.set_ylim(figymin,figymax)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_aspect(1)
		ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)

		#fig.savefig(outfigname, transparent = True)
		plt.show()

	return data_reg, hd_new


def fits_IvTOTb(self, outname=None, overwrite=False):
	'''
	Convert intensity to brightness temperature.
	Calling a function IvTOJT from astrofuncs.

	# nu0  = header['RESTFRQ'] rest frequency [Hz]
    # bmaj = header['BMAJ'] # major beam size [deg]
    # bmin = header['BMIN'] # minor beam size [deg]
    # Iv: intensity [Jy/beam]
    # C2: coefficient to convert beam to str
	'''

	### reading fits files
	# tempreate image
	data, header = fits.getdata(self,header=True)

	# output name
	if outname:
		pass
	else:
		outname = self.replace('.fits', '.tb.fits')

	# reading header info.
	xlabel   = header['CTYPE1']
	ylabel   = header['CTYPE2']
	try:
	    restfreq = header['RESTFRQ'] # Hz
	except:
	    restfreq = header['RESTFREQ'] # Hz
	refval_x = header['CRVAL1']       # deg
	refval_y = header['CRVAL2']       # deg
	refval_v = header['CRVAL3']
	refpix_x = int(header['CRPIX1'])
	refpix_y = int(header['CRPIX2'])
	refpix_v = int(header['CRPIX3'])
	del_x    = header['CDELT1']       # deg
	del_y    = header['CDELT2']       # deg
	del_v    = header['CDELT3']
	nx       = header['NAXIS1']
	ny       = header['NAXIS2']
	nchan    = header['NAXIS3']
	bmaj     = header['BMAJ'] # deg
	bmin     = header['BMIN'] # deg
	bpa      = header['BPA']  # deg
	unit     = header['BUNIT']

	# convert
	data_Tb = IvTOJT(restfreq, bmaj, bmin, data)

	# output
	# new header
	hd_new = header
	hd_new['BUNIT'] = 'K'
	hd_new['BTYPE'] = 'Tb'

	# write fits
	if overwrite:
		try:
			os.system('rm ' +outname)
		except:
			pass

	fits.writeto(outname, data_Tb, header=hd_new)



def fits_TbTOIv(self, outname=None, overwrite=False):
	'''
	Convert intensity to brightness temperature.
	Calling a function IvTOJT from astrofuncs.

	# nu0  = header['RESTFRQ'] rest frequency [Hz]
	# bmaj = header['BMAJ'] # major beam size [deg]
	# bmin = header['BMIN'] # minor beam size [deg]
	# Iv: intensity [Jy/beam]
	# C2: coefficient to convert beam to str
	'''

	### reading fits files
	# tempreate image
	data, header = fits.getdata(self,header=True)

	# output name
	if outname:
		pass
	else:
		outname = self.replace('.fits', '_Iv.fits')

	# reading header info.
	xlabel   = header['CTYPE1']
	ylabel   = header['CTYPE2']
	try:
	    restfreq = header['RESTFRQ'] # Hz
	except:
	    restfreq = header['RESTFREQ'] # Hz
	refval_x = header['CRVAL1']       # deg
	refval_y = header['CRVAL2']       # deg
	refval_v = header['CRVAL3']
	refpix_x = int(header['CRPIX1'])
	refpix_y = int(header['CRPIX2'])
	refpix_v = int(header['CRPIX3'])
	del_x    = header['CDELT1']       # deg
	del_y    = header['CDELT2']       # deg
	del_v    = header['CDELT3']
	nx       = header['NAXIS1']
	ny       = header['NAXIS2']
	nchan    = header['NAXIS3']
	bmaj     = header['BMAJ'] # deg
	bmin     = header['BMIN'] # deg
	bpa      = header['BPA']  # deg
	unit     = header['BUNIT']

	# convert
	data_Iv = TbTOIv(data, restfreq, bmaj, bmin)

	# output
	# new header
	hd_new = header
	hd_new['BUNIT'] = 'Jy/beam'
	hd_new['BTYPE'] = 'Intensity'

	# write fits
	if overwrite:
		try:
			os.system('rm ' +outname)
		except:
			pass

	fits.writeto(outname, data_Iv, header=hd_new)



def fits_getaxes(self, velocity=True, relative=True, inmode='fits'):
	'''
	Return axes of the input fits file.

	self (str): input fits file.
	inmode (str): If 'fits', self (input strings) will be treated as input fits file.
	 If 'data', input values of data and header will be used. Default 'fits'. Put strings as self,
	 even if inmode='data' to privent errors.
	data (array): data of a fits file.
	header (array): header of a fits file.
	noreg (bool): If False, regrid will be done. Default True.
	 For some projections, this will be needed to draw maps with exact coordinates.
	'''
	# reading fits files
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
	else:
		print ('WARNING: Cannot find restfrequency in the header.')
		restfreq = None

	if 'LONPOLE' in header:
		phi_p = header['LONPOLE']
	else:
		phi_p = 180.

	if 'LATPOLE' in header:
		the_p = header['LATPOLE']
	else:
		the_p = None


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
		pc_ij = np.array([
			[header['PC%i_%i'%(i+1,j+1)] for j in range(naxis)]
			 for i in range(naxis)])
		pc_ij = pc_ij*del_i
	elif 'CD1_1' in header:
		pc_ij = np.array([[header['CD%i_%i'%(i+1,j+1)] if 'CD%i_%i'%(i+1,j+1) in header else 0.
		 for j in range(naxis)]
		  for i in range(naxis)])
	else:
	    print ('CAUTION: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
	    pc_ij = np.array([
	    	[1. if i ==j else 0. for j in range(naxis)] for i in range(naxis)])
	    pc_ij = pc_ij*del_i


	# get axes
	fits_axes = np.array([np.dot(pc_ij, (i+1 - refpix_i))\
	 for i in range(np.max(naxis_i))]).T # +1 in i+1 comes from 0 start index in python
	fits_axes = np.array([fits_axes[i,:naxis_i[i]] for i in range(naxis)]) + refval_i
	#xaxis = xaxis[:naxis_i[0]]
	#yaxis = yaxis[:naxis_i[1]]

	if relative:
		fits_axes[0] = fits_axes[0] - refval_i[0]
		fits_axes[1] = fits_axes[1] - refval_i[1]

	if velocity:
		if naxis <= 2:
			print ('ERROR: The number of axes of the fits file is <= 2. No velocity axis.')
		else:
			if label_i[2] == 'VRAD' or label_i[2] == 'VELO':
				print ('The third axis is ', label_i[2])
				del_v    = del_v*1.e-3    # m/s --> km/s
				refval_v = refval_v*1.e-3 # m/s --> km/s
			else:
				print ('The third axis is ', label_i[2])
				fits_axes[2] = clight*(1.-fits_axes[2]/restfreq) # radio velocity c*(1-f/f0) [cm/s]
				fits_axes[2] = fits_axes[2]*1.e-5                # cm/s --> km/s

	return fits_axes


if __name__ == '__main__':
	### Test the function, fits_rotate, using a Gaussian function

	# propaty of Gaussian function
	x            = np.arange(-10,11,0.1)
	y            = np.arange(-10,11,0.1)
	xx, yy       = np.meshgrid(x,y)
	meanx        = 0
	meany        = 0
	sigx         = 3
	sigy         = 1
	amp          = 10.
	pa           = 10.

	# normal Gaussian & rotated Gaussian
	gauss        = mf.gaussian2D(xx,yy,amp,meanx,meany,sigx,sigy)
	gauss_rotate = mf.gaussian2D(xx,yy,amp,meanx,meany,sigx,sigy,pa=pa)

	# rotate an array using fits_rotate function
	gauss_test   = imrotate(gauss, angle=10.)
	#print gauss_test.shape


	# plot the results
	# ax01: using meshgrid
	fig = plt.figure()
	ax  = fig.add_subplot(121)

	ax.pcolor(xx,yy,gauss, cmap='Greys')
	ax.contour(xx,yy,gauss, colors='k')
	ax.contour(xx,yy,gauss_rotate, colors='blue')
	ax.contour(xx,yy,gauss_test, colors='red')
	ax.set_xlim(-10,10)
	ax.set_ylim(-10,10)
	ax.set_aspect(1)


	# ax02: using imshow
	ax2  = fig.add_subplot(122)

	ax2.imshow(gauss, cmap='Greys', origin='lower',extent=(-10,10,-10,10))
	ax2.contour(gauss, colors='k', origin='lower',extent=(-10,10,-10,10))
	ax2.contour(gauss_rotate, colors='blue',origin='lower',extent=(-10,10,-10,10))
	ax2.contour(gauss_test, colors='red',origin='lower',extent=(-10,10,-10,10))
	ax2.set_xlim(-10,10)
	ax2.set_ylim(-10,10)
	ax2.set_aspect(1)
	plt.show()