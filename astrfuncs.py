### modules
import numpy as np
from astrconsts import *



# Jy/beam --> K (when optically thick)
def IvTOTex(nu0, bmaj, bmin, Iv):
    # nu0 = header['RESTFRQ'] rest frequency [Hz]
    # bmaj = header['BMAJ'] # major beam size [deg]
    # bmin = header['BMIN'] # minor beam size [deg]
    # Iv: intensity [Jy/beam]
    # C1: coefficient to convert Iv to Tex
    # C2: coefficient to convert beam to str

    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    C1=2.*hp*(nu0*nu0*nu0)/(clight*clight*1.e6) # in MKS

    # Jy/beam -> Jy/str
    # Omg_beam (str) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/str]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str
    Istr = Iv/bTOstr # Jy/beam --> Jy/str
    Istr = Istr*1.0e-26 # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
    Istr = Istr*1.e7*1.e-4 # MKS --> cgs

    Tex = (hp*nu0/kb)/(np.log((C1/(Istr)+1.))) # no approximation [K]
    return Tex


# equivalent brightness temperature
def IvTOJT(nu0, bmaj, bmin, Iv):
    # nu0 = header['RESTFRQ'] rest frequency [Hz]
    # bmaj = header['BMAJ'] # major beam size [deg]
    # bmin = header['BMIN'] # minor beam size [deg]
    # Iv: intensity [Jy/beam]
    # C2: coefficient to convert beam to str

    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    # Jy/beam -> Jy/sr
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str
    Istr = Iv/bTOstr # Jy/beam --> Jy/str
    Istr = Istr*1.0e-26 # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
    Istr = Istr*1.e7*1.e-4 # MKS --> cgs

    JT = (clight*clight/(2.*nu0*nu0*kb))*Istr # equivalent brightness temperature
    return JT


# convert Icgs --> Iv Jy/pixel
def IcgsTOjpp(Icgs, px, py ,dist):
    '''
    Convert Intensity in cgs to in Jy/beam

    Icgs: intensity in cgs unit [erg s-1 cm-2 Hz-1 str-1]
    psize: pixel size (au)
    dist: distance to the object (pc)
    '''

    # cgs --> Jy/str
    Imks = Icgs*1.e-7*1.e4   # cgs --> MKS
    Istr = Imks*1.0e26       # MKS --> Jy/str, 1 Jy = 10^-26 Wm-2Hz-1

    # Jy/sr -> Jy/pixel
    px = np.radians(px/dist/3600.) # au --> radian
    py = np.radians(py/dist/3600.) # au --> radian
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area  = px*py
    Ijpp            = Istr*one_pixel_area # Iv (Jy per pixel)
    return Ijpp


# convert Iv Jy/beam  --> Iv Jy/pixel
def IbeamTOjpp(Ibeam, bmaj, bmin, px, py , au=False, dist=140.):
    '''
    Convert Intensity in cgs to in Jy/beam

    Ibeam: intensity in Jy/beam
    bmaj, bmin: beam size (degree)
    psize: pixel size (default in degree). If au=True, they will be treated in units of au.
    dist: distance to the object (pc)
    '''

    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    C2     = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str

    # Jy/beam --> Jy/str
    Istr = Ibeam/bTOstr # Jy/beam --> Jy/str


    # Jy/sr -> Jy/pixel
    if au:
        px = np.radians(px/dist/3600.) # au --> radian
        py = np.radians(py/dist/3600.) # au --> radian
    else:
        px = np.radians(px) # deg --> rad
        py = np.radians(py) # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area  = np.abs(px*py)
    Ijpp            = Istr*one_pixel_area # Iv (Jy per pixel)
    return Ijpp


# Convert Tb to Iv
def TbTOIv(Tb, nu0, bmaj, bmin):
    # nu0 = header['RESTFRQ'] rest frequency [Hz]
    # bmaj = header['BMAJ'] # major beam size [deg]
    # bmin = header['BMIN'] # minor beam size [deg]
    # Iv: intensity [Jy/beam]
    # C2: coefficient to convert beam to str

    # a conversion factor
    bmaj   = bmaj*np.pi/180.       # deg --> radian
    bmin   = bmin*np.pi/180.       # deg --> radian
    C2     = np.pi/(4.*np.log(2.)) # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2          # beam --> str

    # K --> Iv(in cgs, /str)
    Istr = ((2.*nu0*nu0*kb)/(clight*clight))*Tb

    # Jy/str --> Jy/beam
    Istr = Istr*1.e-7*1.e4 # cgs --> MKS
    Istr = Istr*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Iv   = Istr*bTOstr     # Jy/str --> Jy/beam

    return Iv


# partition function
def Pfunc(EJ, gJ, J, Tk):
    # EJ: energy at energy level J
    # gJ: statistical weight
    # J: energy level
    # Tk: kinetic energy
    Z = 0.0
    for j in J:
        Z = Z + gJ[j]*np.exp(-EJ[j]/Tk)
        #Z = Z + (2.*j+1.)*np.exp(-EJ[j]/Tk)
    return Z


# Planck function
def Bv(T,v):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    v = v * 1.e9 # GHz --> Hz
    exp=np.exp((hp*v)/(kb*T))-1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp
    return Bv


# Jy/beam
def Bv_Jybeam(T,v,bmaj,bmin):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    bmaj, bmin: beamsize [arcsec]
    '''

    # units
    v = v * 1.e9 # GHz --> Hz
    bmaj = np.radians(bmaj/3600.) # arcsec --> radian
    bmin = np.radians(bmin/3600.) # arcsec --> radian

    # coefficient for unit convertion
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str


    exp=np.exp((hp*v)/(kb*T))-1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp

    # cgs --> Jy/beam
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Bv = Bv*bTOstr     # Jy/str --> Jy/beam
    return Bv



# Rayleigh-Jeans approx.
def BvRJ(T,v):
    # T: temprature [K]
    # v: frequency [GHz]
    v = v * 1.e9 # GHz --> Hz
    Bv = 2.*v*v*kb*T/(clight*clight)
    return Bv