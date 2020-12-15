# -*- coding: utf-8 -*-
'''
Made and developed by J. Sai.

email: jn.insa.sai@gmail.com
'''



### Modules
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.wcs
from astropy.wcs import WCS
import matplotlib.pyplot as plt



### Constants (in cgs)
clight     = 2.99792458e10 # light speed [cm s^-1]




### Imfits
class Imfits():
	'''
	Read a fits file, store the information, and draw maps.
	'''


	def __init__(self, infile, pv=False, relativecoords=True):
		self.file = infile
		self.data, self.header = fits.getdata(infile, header=True)

		if pv:
			self.read_pvfits()
		else:
			self.read_header()
			self.get_coordinates(relative=relativecoords)
			#self.fits_deprojection(relativecoords=relativecoords)

		self.ifpv = pv


	def read_header(self, velocity=True):
		'''
		Get axes of a fits file. x axis and y axis will be in intermediate coordinates.

		Args:
			fitsname: input fits file name

		Return:
			xaxis, yaxis, vaxis, saxis
		'''
		# reading fits files
		header = self.header
		#data, header = fits.getdata(fitsname,header=True)


		# number of axis
		naxis    = header['NAXIS']
		if naxis < 2:
			print ('ERROR\treadfits: NAXIS of fits is < 2.')
			return
		self.naxis = naxis

		naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
		label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
		refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
		refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
		if 'CDELT1' in header:
			del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree
		self.naxis_i  = naxis_i
		self.label_i  = label_i
		self.refpix_i = refpix_i
		self.refval_i = refval_i

		# beam size (degree)
		if 'BMAJ' in header:
			bmaj     = header['BMAJ'] # degree
			bmin     = header['BMIN'] # degree
			bpa      = header['BPA']  # degree
			self.beam = np.array([bmaj*3600., bmin*3600., bpa]) # Get in arcsec
		else:
			self.beam = None


		# rest frequency (Hz)
		if 'RESTFRQ' in header:
			restfreq = header['RESTFRQ']
		elif 'RESTFREQ' in header:
			restfreq = header['RESTFREQ']
		elif 'FREQ' in header:
			restfreq = header['FREQ']
		else:
			restfreq = None
		self.restfreq = restfreq

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
		self.projection = projection

		# rotation of pixel coordinates
		if 'PC1_1' in header:
			pc_ij = np.array([
				[header['PC%i_%i'%(i+1,j+1)]
				if 'PC%i_%i'%(i+1,j+1) in header else 0.
				for j in range(naxis)] for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])
		elif 'CD1_1' in header:
			pc_ij = np.array([
			[header['CD%i_%i'%(i+1,j+1)]
			if 'CD%i_%i'%(i+1,j+1) in header else 0.
			for j in range(naxis)] for i in range(naxis)])
		else:
			print ('CAUTION\tchannelmap: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
			pc_ij = np.array([
				[1. if i==j else 0. for j in range(naxis)]
				 for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])


		# axes
		axes = np.array([np.dot(pc_ij, (i+1 - refpix_i))\
		 for i in range(np.max(naxis_i))]).T      # +1 in i+1 comes from 0 start index in python


		# x & y (RA & DEC)
		xaxis = axes[0]
		xaxis = xaxis[:naxis_i[0]]                # offset, relative
		yaxis = axes[1]
		yaxis = yaxis[:naxis_i[1]]                # offset, relative
		self.delx = xaxis[1] - xaxis[0]
		self.dely = yaxis[1] - yaxis[0]

		# frequency & stokes
		if naxis >= 3:
			# frequency
			vaxis = axes[2]
			vaxis = vaxis[:naxis_i[2]] + refval_i[2]  # frequency, absolute

			if naxis == 4:
				# stokes
				saxis = axes[3]
				saxis = saxis[:naxis_i[3]]
			else:
				saxis = np.array([0.])
		else:
			vaxis = np.array([0.])
			saxis = np.array([0.])


		# frequency --> velocity
		if len(vaxis) > 1:
			if velocity:
				if label_i[2] == 'VRAD' or label_i[2] == 'VELO':
					print ('The third axis is ', label_i[2])
					# m/s --> km/s
					vaxis    = vaxis*1.e-3 # m/s --> km/s
					#del_v    = del_v*1.e-3
					#refval_v = refval_v*1.e-3
					#vaxis    = vaxis*1.e-3
				else:
					print ('The third axis is ', label_i[2])
					print ('Convert frequency to velocity')
					# frequency (Hz) --> radio velocity (km/s)
					vaxis = clight*(1.-vaxis/restfreq)*1.e-5

				if len(vaxis) >= 2:
					self.delv = vaxis[1] - vaxis[0]
				else:
					self.delv = 1.

		axes = np.array([xaxis, yaxis, vaxis, saxis])
		self.axes  = axes
		self.vaxis = vaxis


	# Read fits file of Poistion-velocity (PV) diagram
	def read_pvfits(self):
		'''
		Read fits file of pv diagram produced by CASA.
		'''
		# read header
		header = self.header

		# number of axis
		naxis    = header['NAXIS']
		if naxis < 2:
			print ('ERROR\treadfits: NAXIS of fits is < 2.')
			return
		self.naxis = naxis

		naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
		label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
		refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
		refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
		if 'CDELT1' in header:
			del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree
		self.naxis_i  = naxis_i
		self.label_i  = label_i
		self.refpix_i = refpix_i
		self.refval_i = refval_i

		# beam size (degree)
		if 'BMAJ' in header:
			bmaj     = header['BMAJ'] # degree
			bmin     = header['BMIN'] # degree
			bpa      = header['BPA']  # degree
			self.beam = np.array([bmaj*3600., bmin*3600., bpa])
		else:
			self.beam = None


		# Info. of P.A.
		if 'PA' in header:
			self.pa = header['PA']
		elif 'P.A.' in header:
			self.pa = header['P.A.']
		else:
			self.pa = None

		# Resolution along offset axis
		if self.pa:
			# an ellipse of the beam
			# (x/bmin)**2 + (y/bmaj)**2 = 1
			# y = x*tan(theta)
			# --> solve to get resolution in the direction of pv cut with P.A.=pa
			bmaj, bmin, bpa = self.beam
			del_pa = pa - bpa
			del_pa = del_pa*np.pi/180. # radian
			term_sin = (np.sin(del_pa)/bmin)**2.
			term_cos = (np.cos(del_pa)/bmaj)**2.
			res_off  = np.sqrt(1./(term_sin + term_cos))
			self.res_off = res_off
		else:
			self.res_off = None


		# rest frequency (Hz)
		if 'RESTFRQ' in header:
			restfreq = header['RESTFRQ']
		elif 'RESTFREQ' in header:
			restfreq = header['RESTFREQ']
		elif 'FREQ' in header:
			restfreq = header['FREQ']
		else:
			restfreq = None
		self.restfreq = restfreq


		# get axes
		# rotation of pixel coordinates
		if 'PC1_1' in header:
			pc_ij = np.array([
				[header['PC%i_%i'%(i+1,j+1)]
				if 'PC%i_%i'%(i+1,j+1) in header else 0.
				for j in range(naxis)] for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])
		elif 'CD1_1' in header:
			pc_ij = np.array([
			[header['CD%i_%i'%(i+1,j+1)]
			if 'CD%i_%i'%(i+1,j+1) in header else 0.
			for j in range(naxis)] for i in range(naxis)])
		else:
			print ('CAUTION\tchannelmap: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
			pc_ij = np.array([
				[1. if i==j else 0. for j in range(naxis)]
				 for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])

		# axes
		axes = np.array([np.dot(pc_ij, (i+1 - refpix_i))\
		 for i in range(np.max(naxis_i))]).T # +1 in i+1 comes from 0 start index in python

		# x & v axes
		xaxis = axes[0]
		vaxis = axes[1]
		xaxis = xaxis[:naxis_i[0]]               # offset
		vaxis = vaxis[:naxis_i[1]] + refval_i[1] # frequency, absolute

		# check unit of offest
		if 'CUNIT1' in header:
			unit_i = np.array([header['CUNIT'+str(i+1)] for i in range(naxis)]) # degree
			if unit_i[0] == 'degree' or unit_i[0] == 'deg':
				# degree --> arcsec
				xaxis    = xaxis*3600.
				del_i[0] = del_i[0]*3600.
		else:
			print ('WARNING: No unit information in the header.\
				Assume units of arcesc and Hz for offset and frequency axes, respectively.')

		# frequency --> velocity
		if label_i[1] == 'VRAD' or label_i[1] == 'VELO':
			vaxis    = vaxis*1.e-3 # m/s --> km/s
			#del_v    = del_v*1.e-3
			#refval_v = refval_v*1.e-3
		else:
			print ('Convert frequency to velocity')
			vaxis    = clight*(1.-vaxis/restfreq) # radio velocity c*(1-f/f0) [cm/s]
			vaxis    = vaxis*1.e-5                # cm/s --> km/s
			#del_i[1] = -del_i[1]*clight/restfreq  # delf --> delv [cm/s]
			#del_i[1] = del_i[1]*1.e-5             # cm/s --> km/s

		axes_out = np.array([xaxis, vaxis])
		if naxis >= 2:
			saxis = axes[2]
			saxis = saxis[:naxis_i[2]]
			axes_out = np.array([xaxis, vaxis, saxis])


		# get delta
		delx = xaxis[1] - xaxis[0]
		delv = vaxis[1] - vaxis[0]

		self.axes  = axes_out
		self.xaxis = xaxis
		self.vaxis = vaxis
		self.delx  = delx
		self.delv  = delv


	# Get sky coordinates with astropy
	def get_coordinates(self, relative=True):
		'''
		Get sky coordinates.
		'''
		# Get wcs
		wcs = WCS(self.file)
		self.wcs = wcs

		naxis_i  = self.naxis_i
		refpix_i = self.refpix_i
		refval_i = self.refval_i
		nx = naxis_i[0]
		ny = naxis_i[1]
		ref_x = refval_i[0]
		ref_y = refval_i[1]

		xaxis = np.arange(0,nx,1)
		yaxis = np.arange(0,ny,1)
		xx, yy = np.meshgrid(xaxis, yaxis)
		sc     = astropy.wcs.utils.pixel_to_skycoord(xx, yy, wcs)

		# Sky coordinates
		xx = sc.ra.deg
		yy = sc.dec.deg

		# Relative coordinates
		if relative:
			# RA
			xx = xx - ref_x
			xx = xx*np.cos(np.radians(yy))

			# DEC
			yy = yy - ref_y

			ctype = 'relative'
		else:
			ctype = 'absolute'

		self.xx = xx
		self.yy = yy
		self.cc = np.array([ref_x, ref_y])
		self.ctype = ctype


	# Deproject fits coordinates
	def fits_deprojection(self, noreg=True, relativecoords=True):
		'''
		Deproject coordinates of a fits file. Steps of transformation is as below:
		 1. Projected coordinates (x, y) --> native polar coordinates (phi, theta)
		 2. (phi, theta) --> (ra, dec)

		Args:
			noreg (bool): If False, regrid will be done. Default True.
		 	 For some projections, this will be needed to draw maps with exact coordinates.
		 	use (str): 'relative' or 'absolute'
		'''
		header = self.header


		if 'LONPOLE' in header:
			phi_p = header['LONPOLE']
		else:
			phi_p = 180.

		if 'LATPOLE' in header:
			the_p = header['LATPOLE']
		else:
			the_p = None

		# x & y (RA & DEC)
		xaxis, yaxis, vaxis, saxis = self.axes
		projection                 = self.projection

		# 1. get intermidiate coordinates before the projection correction
		xx, yy = np.meshgrid(xaxis, yaxis)
		#print xx[0,0],xx[-1,-1],yy[0,0],yy[-1,-1]


		# 2. (x,y) --> (phi, theta): native coordinates
		#  Correct projection effect, and then put them into polar coordinates
		#  For detail, look into Mark R. Calabretta and Eric W. Greisen (A&A, 2002)
		#   and Mark Calabretta (1992)
		#  Here variables are as following:
		#	phi, theta: (phi, theta) in polar coordinates
		#	alpha_0, delta_0: Celestial longitude and latitude of the fiducial point
		#	alpha_p, delta_p: Celestial longitude and latitude of the native pole
		#	phi_0, the_0: Native longitude and latitude of the fiducial point
		#   phi_p, the_p: Native longitude and latitude of the celestial pole

		# SIN: Slant orthographic
		if projection == 'SIN':
			# print ('projection: SIN')

			# exact solution of phi & theta
			# xi & eta
			xi  = header['PV1_1'] if 'PV1_1' in header else 0.
			eta = header['PV1_2'] if 'PV1_2' in header else 0.

			# eq. (38) and (44) in Calabretta and Greisen (A&A, 2002),
			#  where X' & Y' are replaced with xi and eta
			xx_rad = xx*np.pi/180.
			yy_rad = yy*np.pi/180.
			a = xi*xi + eta*eta + 1.
			b = xi*(xx_rad - xi) + eta*(yy_rad - eta)
			c = (xx_rad - xi)**2 + (yy_rad - eta)**2 - 1.

			sol_1 = np.arcsin((-b + np.sqrt(b*b - a*c))/a)*180./np.pi
			sol_2 = np.arcsin((-b - np.sqrt(b*b - a*c))/a)*180./np.pi

			if np.any(np.abs(90. - sol_1) <= np.abs(90. - sol_2)):
				theta = sol_1
			else:
				theta = sol_2

			sin_th = np.sin(theta*np.pi/180.)
			phi    = np.arctan2((xx_rad - xi*(1. - sin_th)), -(yy_rad - eta*(1. - sin_th) ))*180./np.pi

			# approximate solution
			#phi   = np.arctan2(xx,-yy)*180./np.pi
			#theta = np.arccos(np.sqrt(xx*xx + yy*yy)*np.pi/180.)*180./np.pi
			#print phi
			#print theta

			# values for converstion from (phi, theta) to (ra, dec)
			alpha_0 = self.refval_i[0] # degree
			delta_0 = self.refval_i[1]
			alpha_p = alpha_0
			delta_p = delta_0
			the_0   = 90.
			phi_0   = 0.
			reg     = False
		elif projection == 'SFL':
			# (ra, dec) of reference position is (0,0) in (phi, theta) and (x,y)
			# (0,0) is on a equatorial line, and (0, 90) is the pole in a native spherical coordinate
			print ('projection: SFL')

			# values for converstion from (phi, theta) to (ra, dec)
			alpha_0 = self.refval_i[0]
			delta_0 = self.refval_i[1]
			phi_0   = header['PV1_1'] if 'PV1_1' in header else 0.
			the_0   = header['PV1_2'] if 'PV1_2' in header else 0.
			alpha_p = None
			delta_p = None
			reg     = True

			# phi & theta
			xoff0 = phi_0 * np.cos(the_0*np.pi/180.)
			yoff0 = the_0
			cos   = np.cos(np.radians(yy))
			phi   = (xx - xoff0)/cos # deg
			theta = yy + yoff0       # deg
			#print (the_0)

			if the_p:
				pass
			else:
				the_p = 90.
		elif projection == 'GLS':
			print ('WARNING\tfits_deprojection: The projection GFL is treated as a projection SFL.')
			# values for converstion from (phi, theta) to (ra, dec)
			alpha_0 = self.refval_i[0]
			delta_0 = self.refval_i[1]
			phi_0   = header['PV1_1'] if 'PV1_1' in header else 0.
			the_0   = header['PV1_2'] if 'PV1_2' in header else 0.
			alpha_p = None
			delta_p = None
			reg     = True

			# phi & theta
			xoff0 = phi_0 * np.cos(the_0*np.pi/180.)
			yoff0 = the_0
			cos   = np.cos(np.radians(yy))
			phi   = (xx - xoff0)/cos # deg
			theta = yy + yoff0       # deg
			#print (the_0)
			if the_p:
				pass
			else:
				the_p = 90.
		elif projection == 'TAN':
			#print 'projection: TAN'
			phi   = np.arctan2(xx,-yy)*180./np.pi
			theta = np.arctan2(180.,np.sqrt(xx*xx + yy*yy)*np.pi)*180./np.pi

			# values for converstion from (phi, theta) to (ra, dec)
			alpha_0 = self.refval_i[0]
			delta_0 = self.refval_i[1]
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
		# (alpha_p, delta_p): cerestial coordinate of the native coordinate pole
		# In SFL projection, reference point is not polar point

		# parameters
		sin_th0  = np.sin(np.radians(the_0))
		cos_th0  = np.cos(np.radians(the_0))
		sin_del0 = np.sin(np.radians(delta_0))
		cos_del0 = np.cos(np.radians(delta_0))


		# spherical coordinate rotation or not
		if phi_0 == 0. and the_0 == 90.:
			# case of spherical coordinate rotation
			sin_delp = np.sin(np.radians(delta_p))
			cos_delp = np.cos(np.radians(delta_p))
			pass
		else:
			# with non-polar (phi0, and theta0)
			# we have to derive delta_p and alpha_p
			argy    = sin_th0
			argx    = cos_th0*np.cos(np.radians(phi_p-phi_0))
			arg     = np.arctan2(argy,argx)
			#print (arg)

			cos_inv  = np.arccos(sin_del0/(np.sqrt(1. - cos_th0*cos_th0*np.sin(np.radians(phi_p - phi_0))*np.sin(np.radians(phi_p - phi_0)))))

			sol_1 = (arg + cos_inv)*180./np.pi
			sol_2 = (arg - cos_inv)*180./np.pi

			if np.any(np.abs(the_p - sol_1) <= np.abs(the_p - sol_2)):
				delta_p = sol_1
			else:
				delta_p = sol_2

			sin_delp = np.sin(np.radians(delta_p))
			cos_delp = np.cos(np.radians(delta_p))

			if delta_p == 90.:
				alpha_p = alpha_0 + phi_p - phi_0 - 180.
			elif delta_p == -90.:
				alpha_p = alpha_0 - phi_p + phi_0
			else:
				sin_alpha_p = np.sin(np.radians(phi_p - phi_0))*cos_th0/cos_del0
				cos_alpha_p = (sin_th0 - sin_delp*sin_del0)/(cos_delp*cos_del0)
				#print sin_alpha_p, cos_alpha_p
				#print np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
				alpha_p = alpha_0 - np.arctan2(sin_alpha_p,cos_alpha_p)*180./np.pi
				#print (alpha_p)


		# (phi, theta) --> (ra, dec) finally
		sin_th = np.sin(np.radians(theta))
		cos_th = np.cos(np.radians(theta))

		in_sin = sin_th*sin_delp + cos_th*cos_delp*np.cos(np.radians(phi-phi_p))
		delta  = np.arcsin(in_sin)*180./np.pi

		argy  = -cos_th*np.sin(np.radians(phi-phi_p))
		argx  = sin_th*cos_delp - cos_th*sin_delp*np.cos(np.radians(phi-phi_p))
		alpha = alpha_p + np.arctan2(argy,argx)*180./np.pi
		#print (alpha)

		alpha[np.where(alpha < 0.)]   = alpha[np.where(alpha < 0.)] + 360.
		alpha[np.where(alpha > 360.)] = alpha[np.where(alpha > 360.)] - 360.

		# coordinate type: relative or absolute
		if relativecoords:
			alpha = (alpha - alpha_0)*np.cos(np.radians(delta))
			delta = delta - delta_0
			self.ctype = 'relative'
			print ('Coordinates: relative.')
		else:
			self.ctype = 'absolute'
			print ('Coordinates: absolute.')

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
		#print (xx[0,0],yy[0,0])
		#print xx[0,-1],yy[0,-1]
		#print xx[-1,0],yy[-1,0]
		#print xx[-1,-1],yy[-1,-1]
		#print del_i[0], del_x, xx[0:naxis_i[1]-1,0:naxis_i[1]-1] - xx[1:naxis_i[1],1:naxis_i[1]]
		#print del_i[1], yy[0,0] - yy[1,0]

		# regridding for plot if the projection requires
		if noreg:
			pass
		else:
			from scipy.interpolate import griddata
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

		self.xx = xx
		self.yy = yy
		self.cc = np.array([alpha_0, delta_0])

		return


	def shift_coord_center(self, coord_center, relativecoords=True):
		'''
		Shift the coordinate center.

		Args:
			coord_center: Put an coordinate for the map center.
			   The shape must be '00h00m00.00s 00d00m00.00s', or
			   'hh:mm:ss.ss dd:mm:ss.ss'. RA and DEC must be separated
			   by space.
		'''
		# module
		from astropy.coordinates import SkyCoord

		# ra, dec
		c_ra, c_dec = coord_center.split(' ')
		cc          = SkyCoord(c_ra, c_dec, frame='icrs')
		cra_deg     = cc.ra.degree                   # in degree
		cdec_deg    = cc.dec.degree                  # in degree
		new_cent    = np.array([cra_deg, cdec_deg])  # absolute coordinate of the new image center

		# current coordinates
		alpha = self.xx
		delta = self.yy
		ra_0, dec_0 = self.cc # deg


		# calculate shift in absolute coordinates
		if self.ctype == 'absolute':
			pass
		elif self.ctype == 'relative':
			# to absolute
			delta = delta + dec_0
			alpha = ra_0 + alpha/np.cos(np.radians(delta))


		# shift of the center
		#  now new_cent is the center of the relative coordinates
		alpha = (alpha - cra_deg)*np.cos(np.radians(delta))
		delta = delta - cdec_deg

		self.xx = alpha
		self.yy = delta
		self.cc = new_cent
		self.ctype = 'relative'


	# ------------------ for plot ---------------------
	def draw_Idistmap(self, data=None, ax=None, outname=None, imscale=[], outformat='pdf', color=True, cmap='Greys',
		colorbar=False, cbaroptions=np.array(['vertical','40','0','Jy/beam']), vmin=None,vmax=None,
		contour=True, clevels=np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]), ccolor='k', mask=None,
		xticks=[], yticks=[], relativecoords=True, csize=18, scalebar=[],
		cstar=True, prop_star=np.array(['1','0.5','white']), logscale=False, bcolor='k',figsize=(11.69,8.27),
		tickcolor='k',axiscolor='k',labelcolor='k',coord_center=None, map_center=None, plot_beam = True, interpolation=None,
		noreg=True, inmode=None, exact_coord=False, arcsec=True):
		'''
		Draw a map from an image cube. You can overplot maps by giving ax where other maps are drawn.

		Usage (examples)
		Draw a map:
		Idistmap('object.co.fits', outname='test', imscale=[-10,10,-10,10], color=True, cmap='Greys')

		Overplot:
		rms = 10. # rms
		ax = Idistmap('object01.fits', outname='test', imscale=[-10,10,-10,10], color=True)
		ax = Idistmap('object02.fits', outname='test', ax=ax, color=False,
		contour=True, clevels=np.array([3,6,9,12])*rms) # put ax=ax and the same outname

		# If you want to overplot something more
		ax.plot(0, 0, marker='*', size=40)        # overplot
		plt.savefig('test.pdf', transparent=True) # use the same name

		Make a map editting data:
		fitsdata = 'name.fits'
		data, hd = fits.getdata(fitsdata, header=True) # get data
		data = data*3.                                 # edit data
		Idistmap(fitsdata, data=data, header=hd, inmode='data') # set inmode='data'


		Args
		fitsdata (fits file): Input fitsdata. It must be image data having 3 or 4 axes.
		data (array): data of a fits file.
		header: header of a fits file.
		outname (str): Output file name. Do not include file extension.
		outformat (str): Extension of the output file. Default is pdf.
		imscale (ndarray): Image scale [arcsec]. Input as np.array([xmin,xmax,ymin,ymax]).
		color (bool): If True, image will be described in color scale.
		cmap: Choose colortype of the color scale.
		colorbar (bool): If True, color bar will be put in a map. Default False.
		cbaroptions: Detailed setting for colorbar. np.array(['position','width','pad','label']).
		vmin, vmax: Minimun and maximun values in the color scale. Put abusolute values.
		contour (bool): If True, contour will be drawn.
		clevels (ndarray): Set contour levels. Put abusolute values.
		ccolor: Set contour color.
		xticks, yticks: Optional setting. If input ndarray, xticks and yticsk will be set as the input.
		relativecoords (bool): If True, the coordinate is shown in relativecoordinate. Default True.
		Absolute coordinate is currently now not supported.
		csize: Font size.
		cstar (bool): If True, a cross denoting central stellar position will be drawn.
		prop_star: Detailed setting for the cross showing stellar position.
		np.array(['length','width','color']) or np.array(['length','width','color', 'coordinates']).
		logscale (bool): If True, the color scale will be in log scale.
		coord_center (str): Put an coordinate for the map center. The shape must be '00h00m00.00s 00d00m00.00s', or
		'hh:mm:ss.ss dd:mm:ss.ss'. RA and DEC must be separated by space.
		figsize (tapule): figure size. Default A4 size.
		plot_beam (bool): If True, an ellipse denoting the beam size will be drawn.
		bcolor: color for the ellipse for the beam.
		tickcolor, axiscolor, labelcolor: color set for the map.
		interpolation (str): The color map is shown with interpolation.
		noreg (bool): If False, coordinates will be regrided when the deprojection is calculated.
		scalebar: Optional setting. Input ndarray([barx, bary, barlength, textx, texty, text ]).
		barx and bary are the position where scalebar will be putted. [arcsec].
		barlength is the length of the scalebar in the figure, so in arcsec.
		textx and texty are the position where a label of scalebar will be putted. [arcsec].
		text is a text which represents the scale of the scalebar.
		cstar (bool): If True, central star position will be marked by cross.
		inmode: 'fits' or 'data'. If 'data' is selected, header must be provided. Default 'fits'.
		'''
		# modules
		import matplotlib.figure as figure
		import matplotlib as mpl
		#from mpl_toolkits.mplot3d import axes3d
		from astropy.coordinates import SkyCoord
		import matplotlib.patches as patches

		# format
		formatlist = np.array(['eps','pdf','png','jpeg'])

		# properties of plots
		#mpl.use('Agg')
		plt.rcParams['font.family']     ='Arial' # font (Times New Roman, Helvetica, Arial)
		plt.rcParams['xtick.direction'] = 'in'   # directions of x ticks ('in'), ('out') or ('inout')
		plt.rcParams['ytick.direction'] = 'in'   # directions of y ticks ('in'), ('out') or ('inout')
		plt.rcParams['font.size']       = csize  # fontsize
		#plt.rcParams['xtick.major.width'] = 1.0 # x ticks width
		#plt.rcParams['ytick.major.width'] = 1.0 # y ticks width
		#plt.rcParams['axes.linewidth'] = 1.0    # edge linewidth

		# setting output file name & format
		if (outformat == formatlist).any():
			outname = outname + '.' + outformat
		else:
			print ('ERROR\tIdistmap: Outformat is wrong.')
			return

		if inmode == 'data':
			if data is None:
				print ("inmode ='data' is selected. data must be provided.")
				return

			naxis = len(data.shape)
		else:
			data  = self.data
			naxis = self.naxis

		if naxis < 2:
			print ('ERROR\tIdistmap: NAXIS < 2. It must be >= 2.')
			return

		if self.beam is None:
			plot_beam = False
		else:
			bmaj, bmin, bpa = self.beam

		# coordinate style
		xx = self.xx
		yy = self.yy
		cc = self.cc
		if relativecoords:
			if coord_center:
				self.shift_coord_center(coord_center)
				xx = self.xx
				yy = self.yy
				cc = self.cc
			else:
				if self.ctype == 'absolute':
					x0, y0 = cc
					xx = (xx - x0)*np.cos(np.radians(yy))
					yy = yy - y0
			xlabel = 'RA offset (arcsec)'
			ylabel = 'DEC offset (arcsec)'
		else:
			print ('WARNING: Abusolute coordinates are still in development.')
			xlabel = self.label_i[0]
			ylabel = self.label_i[1]


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

		# figure extent
		xmin   = xx[0,0]
		xmax   = xx[-1,-1]
		ymin   = yy[0,0]
		ymax   = yy[-1,-1]
		del_x  = xx[1,1] - xx[0,0]
		del_y  = yy[1,1] - yy[0,0]
		extent = (xmin-0.5*del_x, xmax+0.5*del_x, ymin-0.5*del_y, ymax+0.5*del_y)
		#print (extent)

		# image scale
		if len(imscale) == 0:
			figxmin, figxmax, figymin, figymax = extent  # arcsec

			if figxmin < figxmax:
				cp      = figxmax
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
		else:
			print ('ERROR\tIdistmap: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')


		# !!!!! plot !!!!!
		# setting figure
		if ax is not None:
			pass
		else:
			fig = plt.figure(figsize=figsize)
			ax  = fig.add_subplot(111)


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


		# color map
		if color:
			if exact_coord:
				imcolor = ax.pcolor(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
			else:
				imcolor = ax.imshow(data, cmap=cmap, origin='lower', extent=extent, norm=norm, interpolation=interpolation, rasterized=True)

			# color bar
			if colorbar:
				cbar_loc, cbar_wd, cbar_pad, cbar_lbl = cbaroptions
				#divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
				#cax     = divider.append_axes(cbar_loc, cbar_wd, pad=cbar_pad)
				cbar    = fig.colorbar(imcolor, ax = ax, orientation=cbar_loc,
				 aspect=float(cbar_wd), pad=float(cbar_pad))
				cbar.set_label(cbar_lbl)

		# contour map
		if contour:
			if exact_coord:
				imcont = ax.contour(xx, yy, data, colors=ccolor, origin='lower', levels=clevels, linewidths=1)
			else:
				imcont = ax.contour(data, colors=ccolor, origin='lower', levels=clevels,linewidths=1, extent=(xmin,xmax,ymin,ymax))


		# set axes
		ax.set_xlim(figxmin,figxmax)
		ax.set_ylim(figymin,figymax)
		ax.set_xlabel(xlabel,fontsize=csize)
		ax.set_ylabel(ylabel, fontsize=csize)
		if len(xticks) != 0:
			ax.set_xticks(xticks)
			ax.set_xticklabels(xticks)

		if  len(yticks) != 0:
			ax.set_yticks(yticks)
			ax.set_yticklabels(yticks)

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

			#cross01 = patches.Arc(xy=(pos_cstar[0],pos_cstar[1]), width=ll, height=1e-9, lw=lw, color=cl,zorder=11)
			#cross02 = patches.Arc(xy=(pos_cstar[0],pos_cstar[1]), width=1e-9, height=ll, lw=lw, color=cl,zorder=12)
			#ax.add_patch(cross01)
			#ax.add_patch(cross02)
			ax.hlines(pos_cstar[1], pos_cstar[0]-ll*0.5, pos_cstar[0]+ll*0.5, lw=lw, color=cl,zorder=11)
			ax.vlines(pos_cstar[0], pos_cstar[1]-ll*0.5, pos_cstar[1]+ll*0.5, lw=lw, color=cl,zorder=11)


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

			#scale   = patches.Arc(xy=(barx,bary), width=barlength, height=0.001, lw=2, color=colors,zorder=10)
			#ax.add_patch(scale)
			ax.hlines(bary, barx - barlength*0.5, barx + barlength*0.5, lw=2, color=colors,zorder=10)
			ax.text(textx, texty, text, color=colors, fontsize=barcsize,
			 horizontalalignment='center', verticalalignment='center')
		else:
			print ('scalebar must be 8 elements. Check scalebar.')

		plt.savefig(outname, transparent = True)

		return ax

	# Channel maps
	def draw_channelmaps(self, grid=None, data=None, outname=None, outformat='pdf', imscale=[1], color=False, cbaron=False, cmap='Greys', vmin=None, vmax=None,
		contour=True, clevels=np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]), ccolor='k',
		nrow=5, ncol=5,velmin=None, velmax=None, nskip=1,
		xticks=np.empty, yticks=np.empty, relativecoords=True, vsys=None, csize=14, scalebar=np.empty(0),
		cstar=True, prop_star=np.array(['1','0.5','red']), logscale=False, tickcolor='k',axiscolor='k',
		labelcolor='k',cbarlabel=None, txtcolor='k', bcolor='k', figsize=(11.69,8.27),
		cbarticks=None, coord_center=None, noreg=True, arcsec=True, sbar_vertical=False,
		cbaroptions=np.array(['right','5%','0%']), inmode='fits', vlabel_on=True):
		'''
		Make channel maps from a fits file.


		Usage (examples)
		 Draw a map:
		  channelmap('object.co.fits', outname='test', imscale=[-10,10,-10,10],
		   color=True, cmap='Greys', velmin=5.2, velmax=12.5, nrow=5, ncol=8)

		 Overplot:
		  rms = 10. # rms
		  grid = channelmap('object.co.fits', outname='test', imscale=[-10,10,-10,10], color=True)
		  grid = channelmap('object.13co.fits', outname='test', grid=grid, color=False,
		   contour=True, clevels=np.array([3,6,9,12])*rms) # put grid=grid, the same outname

		  # If you want to overplot something more
		  grid[nrow*(ncol-1)].plot(0, 0, marker='*', size=40) # overplot
		  plt.savefig('test.pdf', transparent=True)           # use the same name


		Args
		fitsdata: Input fitsdata. It must be an image cube having 3 or 4 axes.
		outname: Output file name. Do not include file extension.
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
		prop_star: Detailed setting for the cross showing stellar position.
		 np.array(['length','width','color']) or np.array(['length','width','color', 'coordinates']).
		logscale (bool): If True, the color scale will be in log scale.
		coord_center (str): Put an coordinate for the map center. The shape must be '00h00m00.00s 00d00m00.00s', or
		 'hh:mm:ss.ss dd:mm:ss.ss'. RA and DEC must be separated by space.
		locsym: Removed. A factor to decide locations of symbols (beam and velocity label). It must be 0 - 1.
		tickcolor, axiscolor, labelcolor, txtcolor: Colors for the maps.
		scalebar (array): If it is given, scalebar will be drawn. It must be given as [barx, bary, bar length, textx, texty, text].
		                   Barx, bary, textx, and texty are locations of a scalebar and a text in arcsec.
		nskip: the number of channel skipped
		'''

		# modules
		import matplotlib.figure as figure
		import matplotlib as mpl
		#from mpl_toolkits.mplot3d import axes3d
		from astropy.coordinates import SkyCoord
		import matplotlib.patches as patches
		from mpl_toolkits.axes_grid1 import ImageGrid

		# format
		formatlist = np.array(['eps','pdf','png','jpeg'])

		# properties of plots
		#mpl.use('Agg')
		plt.rcParams['font.family']     ='Arial' # font (Times New Roman, Helvetica, Arial)
		plt.rcParams['xtick.direction'] = 'in'   # directions of x ticks ('in'), ('out') or ('inout')
		plt.rcParams['ytick.direction'] = 'in'   # directions of y ticks ('in'), ('out') or ('inout')
		plt.rcParams['font.size']       = csize  # fontsize
		#plt.rcParams['xtick.major.width'] = 1.0 # x ticks width
		#plt.rcParams['ytick.major.width'] = 1.0 # y ticks width
		#plt.rcParams['axes.linewidth'] = 1.0    # edge linewidth


		# Setting output file name & format
		if (outformat == formatlist).any():
			#outfile = outname + '_nmap{0:02d}'.format(nmap) + '.' + outformat
			outfile = outname + '.' + outformat
		else:
			print ('ERROR\tdraw_channelmaps: Outformat is wrong.')
			return

		if inmode == 'data':
			if data is None:
				print ("inmode ='data' is selected. data must be provided.")
				return

			naxis = len(data.shape)
		else:
			data  = self.data
			header = self.header
			naxis = self.naxis

		# number of axis
		if naxis < 3:
			print ('ERROR\tdraw_channelmaps: NAXIS of fits is < 3 although It must be > 3.')
			return

		if self.beam is None:
			plot_beam = False
		else:
			bmaj, bmin, bpa = self.beam


		# Coordinates
		xx = self.xx
		yy = self.yy
		cc = self.cc
		if relativecoords:
			if coord_center:
				self.shift_coord_center(coord_center)
				xx = self.xx
				yy = self.yy
				cc = self.cc
			else:
				if self.ctype == 'absolute':
					x0, y0 = cc
					xx = (xx - x0)*np.cos(np.radians(yy))
					yy = yy - y0
			xlabel = 'RA offset (arcsec)'
			ylabel = 'DEC offset (arcsec)'
		else:
			print ('WARNING: Abusolute coordinates are still in development.')
			xlabel = self.label_i[0]
			ylabel = self.label_i[1]


		# check data axes
		if len(data.shape) == 3:
			pass
		elif len(data.shape) == 4:
			data = data[0,:,:,:]
		else:
			print ('Error\tdraw_channelmaps: Input fits size is not corrected.\
			 It is allowed only to have 3 or 4 axes. Check the shape of the fits file.')
			return

		# unit: arcsec or deg
		if arcsec:
			xx     = xx*3600.
			yy     = yy*3600.


		# Get velocity axis
		vaxis = self.vaxis
		delv  = self.delv
		nchan = self.naxis_i[2]

		if delv < 0:
			delv  = - delv
			vaxis = vaxis[::-1]
			data  = data[::-1,:,:]


		# Figure extent
		xmin   = xx[0,0]
		xmax   = xx[-1,-1]
		ymin   = yy[0,0]
		ymax   = yy[-1,-1]
		del_x  = xx[1,1] - xx[0,0]
		del_y  = yy[1,1] - yy[0,0]
		extent = (xmin-0.5*del_x, xmax+0.5*del_x, ymin-0.5*del_y, ymax+0.5*del_y)
		#print (extent)

		# Image scale
		if len(imscale) == 0:
			figxmin, figxmax, figymin, figymax = extent  # arcsec

			if figxmin < figxmax:
				cp      = figxmax
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
		else:
			print ('ERROR\tIdistmap: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')


		# Relative velocity
		if vsys:
			vaxis = vaxis - vsys


		# Set colorscale
		if vmax:
			pass
		else:
			vmax = np.nanmax(data)

		# Logscale
		if logscale:
			norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
		else:
			norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


		# !!! Plot !!!
		# Setting colorbar
		if color and cbaron:
			cbar_mode = 'single'
		else:
			cbar_mode= None


		if grid:
			pass
		else:
			fig = plt.figure(figsize=figsize)
			 # Setting grid
			cbar_loc, cbar_wd, cbar_pad = cbaroptions
			grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
				axes_pad=0,share_all=True,cbar_mode=cbar_mode,
				cbar_location=cbar_loc,cbar_size=cbar_wd,cbar_pad=cbar_pad,
				label_mode='1')


		# Setting parameters used to plot
		if len(imscale) == 1:
			figxmin, figxmax, figymin, figymax = extent
		elif len(imscale) == 4:
			figxmax, figxmin, figymin, figymax = imscale
		else:
			print ('ERROR\tdraw_channelmaps: Input imscale is wrong. Must be [xmin, xmax, ymin, ymax]')

		# Counter
		i, j, gridi = [0,0,0]
		gridimax    = nrow*ncol-1

		nroop = nchan//nskip + 1
		ax    = None

		# Loop
		for k in range(0,nroop):
			ichan = k*nskip
			if ichan >= nchan:
				continue

			if gridi > gridimax:
				continue

			# Select channel
			Sv = data[ichan,:,:]

			# velocity at nchan
			v_i = vaxis[ichan]

			# Check whether v_i is within a selected velocity range
			if velmax:
				if v_i > velmax:
					continue

			if velmin:
				if v_i < velmin:
					continue

			# Axis
			ax = grid[gridi]
			print ('Channel ', '%s'%ichan, ', velocity: ', '%4.2f'%v_i, ' km/s')

			# showing in color scale
			if color:
				imcolor = ax.imshow(Sv, cmap=cmap, origin='lower', extent=extent,norm=norm, rasterized=True)

			if contour:
				imcont  = ax.contour(Sv, colors=ccolor, origin='lower',extent=extent, levels=clevels, linewidths=0.5)

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
			ax.tick_params(which='both', direction='in',bottom=True,
			 top=True, left=True, right=True, color=tickcolor,
			  labelcolor=labelcolor, pad=9, labelsize=csize)

			# Velocity label
			if vlabel_on:
				vlabel = '%3.2f'%v_i
				ax.text(0.1, 0.9,vlabel,color=txtcolor,size=csize,
				 horizontalalignment='left', verticalalignment='top',
				  transform=ax.transAxes)

			# On the bottom-left corner pannel
			if i == nrow-1 and j == 0:
				# Labels
				ax.set_xlabel(xlabel)
				ax.set_ylabel(ylabel)
				ax.xaxis.label.set_color(labelcolor)
				ax.yaxis.label.set_color(labelcolor)

				# Plot beam
				bmin_plot, bmaj_plot = ax.transLimits.transform((0,bmaj)) - ax.transLimits.transform((bmin,0))   # data --> Axes coordinate
				beam = patches.Ellipse(xy=(0.1, 0.1), width=bmin_plot, height=bmaj_plot, fc=bcolor, angle=bpa, transform=ax.transAxes)
				ax.add_patch(beam)

				# Scale bar
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

					ax.text(textx,texty,text,color=barcolor,fontsize=barcsize,horizontalalignment='center',verticalalignment='center')
				else:
					print ('scalebar must consist of 8 elements. Check scalebar.')
			#else:
			#ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

			# Central star position
			if cstar:
				ll,lw, cl = prop_star
				ll = float(ll)
				lw = float(lw)

				if relativecoords:
					pos_cstar = np.array([0,0])
				else:
					pos_cstar = self.cc

				ax.hlines(pos_cstar[1], pos_cstar[0]-ll*0.5, pos_cstar[0]+ll*0.5, lw=lw, color=cl,zorder=11)
				ax.vlines(pos_cstar[0], pos_cstar[1]-ll*0.5, pos_cstar[1]+ll*0.5, lw=lw, color=cl,zorder=11)

			# Counts
			j     = j + 1
			gridi = gridi+1

			if j == ncol:
				j = 0
				i = i + 1


		if color and cbaron and ax:
			# With cbar_mode="single", cax attribute of all axes are identical.
			cax = grid.cbar_axes[0]
			cbar = plt.colorbar(imcolor, ticks=cbarticks, cax=cax)
			#cbar = cax.colorbar(imcolor,ticks=cbarticks)
			cax.toggle_label(True)
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

		plt.savefig(outfile, transparent = True)

		return grid


	# Draw pv diagram
	def draw_pvdiagram(self,outname,data=None,header=None,ax=None,outformat='pdf',color=True,cmap='Greys',
		vmin=None,vmax=None,vsys=0,contour=True,clevels=None,ccolor='k', pa=None,
		vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
		lw=1,clip=None,plot_res=True,inmode='fits',xranges=[], yranges=[],
		ln_hor=True, ln_var=True):
		'''
		Draw a PV diagram.

		Args:
		 - outname:
		'''

		# Modules
		import copy
		import matplotlib as mpl

		# format
		formatlist = np.array(['eps','pdf','png','jpeg'])

		# properties of plots
		#mpl.use('Agg')
		plt.rcParams['font.family']     ='Arial' # font (Times New Roman, Helvetica, Arial)
		plt.rcParams['xtick.direction'] = 'in'   # directions of x ticks ('in'), ('out') or ('inout')
		plt.rcParams['ytick.direction'] = 'in'   # directions of y ticks ('in'), ('out') or ('inout')
		plt.rcParams['font.size']       = fontsize  # fontsize

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


		# output file
		if (outformat == formatlist).any():
			outname = outname + '.' + outformat
		else:
			print ('ERROR\tsingleim_to_fig: Outformat is wrong.')
			return

		# Input
		if inmode == 'data':
			if data is None:
				print ("inmode ='data' is selected. data must be provided.")
				return
			naxis = len(data.shape)
		else:
			data   = self.data
			header = self.header
			naxis  = self.naxis


		# figures
		if ax:
			pass
		else:
			fig = plt.figure(figsize=(11.69,8.27)) # figsize=(11.69,8.27)
			ax  = fig.add_subplot(111)

		# Read
		xaxis = self.xaxis
		vaxis = self.vaxis
		delx  = self.delx
		delv  = self.delv
		nx    = len(xaxis)
		nv    = len(vaxis)

		# Beam
		bmaj, bmin, bpa = self.beam

		if self.res_off:
			res_off = self.res_off
		else:
			# Resolution along offset axis
			if self.pa:
				pa = self.pa

			if pa:
				# an ellipse of the beam
				# (x/bmin)**2 + (y/bmaj)**2 = 1
				# y = x*tan(theta)
				# --> solve to get resolution in the direction of pv cut with P.A.=pa
				del_pa = pa - bpa
				del_pa = del_pa*np.pi/180. # radian
				term_sin = (np.sin(del_pa)/bmin)**2.
				term_cos = (np.cos(del_pa)/bmaj)**2.
				res_off  = np.sqrt(1./(term_sin + term_cos))
			else:
				res_off = bmaj

		# relative velocity or LSRK
		offlabel = r'$\mathrm{Offset\ (arcsec)}$'
		if vrel:
			vaxis   = vaxis - vsys
			vlabel  = r'$\mathrm{Relative\ velocity\ (km\ s^{-1})}$'
			vcenter = 0
		else:
			vlabel  = r'$\mathrm{LSR\ velocity\ (km\ s^{-1})}$'
			vcenter = vsys


		# set extent of an image
		offmin = xaxis[0] - delx*0.5
		offmax = xaxis[-1] + delx*0.5
		velmin = vaxis[0] - delv*0.5
		velmax = vaxis[-1] + delv*0.5


		# set axes
		if x_offset:
			data   = data[0,:,:]
			extent = (offmin,offmax,velmin,velmax)
			xlabel = offlabel
			ylabel = vlabel
			hline_params = [vsys,offmin,offmax]
			vline_params = [0.,velmin,velmax]
			res_x = res_off
			res_y = delv
		else:
			data   = np.rot90(data[0,:,:])
			extent = (velmin,velmax,offmin,offmax)
			xlabel = vlabel
			ylabel = offlabel
			hline_params = [0.,velmin,velmax]
			vline_params = [vcenter,offmin,offmax]
			res_x = delv
			res_y = res_off


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
		if len(xranges) == 0:
			ax.set_xlim(extent[0],extent[1])
		elif len(xranges) == 2:
			xmin, xmax = xranges
			ax.set_xlim(xmin, xmax)
		else:
			print ('WARRING: Input xranges is wrong. Must be [xmin, xmax].')
			ax.set_xlim(extent[0],extent[1])

		if len(yranges) == 0:
			ax.set_ylim(extent[2],extent[3])
		elif len(yranges) == 2:
			ymin, ymax = yranges
			ax.set_ylim(ymin, ymax)
		else:
			print ('WARRING: Input yranges is wrong. Must be [ymin, ymax].')
			ax.set_ylim(extent[2],extent[3])


		# lines showing offset 0 and relative velocity 0
		if ln_hor:
			xline = plt.hlines(hline_params[0], hline_params[1], hline_params[2], ccolor, linestyles='dashed', linewidths = 1.)
		if ln_var:
			yline = plt.vlines(vline_params[0], vline_params[1], vline_params[2], ccolor, linestyles='dashed', linewidths = 1.)

		ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)

		# plot resolutions
		if plot_res:
			# x axis
			#print (res_x, res_y)
			res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5)) -  ax.transLimits.transform((0, 0)) # data --> Axes coordinate
			ax.errorbar(0.1, 0.1, xerr=res_x_plt, yerr=res_y_plt, color=ccolor, capsize=3, capthick=1., elinewidth=1., transform=ax.transAxes)

		# aspect ratio
		if ratio:
			change_aspect_ratio(ax, ratio)

		# save figure
		plt.savefig(outname, transparent=True)

		return ax
