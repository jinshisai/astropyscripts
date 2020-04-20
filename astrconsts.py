### Often Use Constants
import numpy as np

### constants

kb     = 1.38064852e-16  # Boltzmann constant [erg K^-1]
hp     = 6.626070040e-27 # Planck constant [erg s]
sigsb  = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
clight = 2.99792458e10   # light speed [cm s^-1]
NA     = 6.022140857e23  # mol^-1
Ggrav  = 6.67428e-8      # gravitational constant [dyn cm^2 g^-2]
Msun   = 1.9884e33       # solar mass [g]
Lsun   = 3.85e26         # solar total luminosity [W]
Lsun   = 3.85e26*1.e7    # solar total luminosity [erg/s]
Rsun   = 6.960e10        # solar radius [cm]
pi     = np.pi
mH     = 1.672621898e-24 # proton mass [g]
Mearth = 3.0404e-6*Msun  # Earth mass [g]
Mjup   = 9.5479e-4*Msun  # Jupiter mass [g]

# distance
# AU --> km 1.50e8
# AU --> cm 1.50e13

auTOkm = 1.495978707e8  # AU --> km
auTOcm = 1.495978707e13 # AU --> cm
auTOpc = 4.85e-6        # au --> pc
pcTOau = 2.06e5         # pc --> au
pcTOcm = 3.09e18        # pc --> cm

# units
# J = 1e7 erg
