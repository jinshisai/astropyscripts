Astropyscripts
-----------------------
Python scripts for astronomy. A main function is Imfits to read and handle fits files easily.


**Required modules**
- sys
- numpy
- pandas
- astropy
- matplotlib


**Contanct**  
E-mail: jn.insa.sai@gmail.com  
Jinshi Sai (Insa Choi)
Department of Astronomy, the University of Tokyo


Features
---------------------
The main function Imfits, a class in imfits.py, is to read fits files and contain header and image information in variables to call the information easily. This is made for fits images especially at (sub)millimeter wavelengths obtained like ALMA, ACA, SMA, JCMT, IRAM-30m and so on. Not guaranteed but could be applied to other fits data like at IR wavelengths.

A input fits file is expected to have four axes; RA, Dec, frequency and Stokes axes. Or it can also read fits files of position-velocity (PV) diagrams. The script is well tested For fits files exported from [CASA](https://casa.nrao.edu).

All functions in pyfigures.py, which are to draw maps from a fits file, were implemented into imfits.py.

Others are just for convenience.
- astrconsts: Contains constants often used in astronomy.
- astrfuncs: Contains functions often used in astronomy.
- imcrop: Crop a specific region from a fits file.
- imhandle: Handle fits files.
- mathfuncs: Contains mathematical functions often used in astronomy.
- pyfigures: A script to draw maps from fits files. Now absorbed into imfits.


How to use
---------------

Imfits read a fits file.

```python
from imfits import Imfits

infile   = 'fitsname.fits' # fits file
fitsdata = Imfits(infile)  # Read information
```

Then, you can call the data and the header information easily.

```python
data  = fitsdata.data  # Call data array
xaxis = fitsdata.xaxis # Call x axis
nx    = fitsdata.nx    # Size of xaxis
```

You can also draw maps.

```python
# Moment maps
fitsdata.draw_Idistmap(outname='map', outformat='pdf')
# Just an example.
# Put more options for the function to work correctly.
```

Add ```pv=True``` as an option to read a fits file of a position-velocity (PV) diagram.

```python
from imfits import Imfits

infile   = 'pvfits.fits'           # fits file
fitsdata = Imfits(infile, pv=True) # Read information
```