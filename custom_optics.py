import numpy as np
import numpy as xp
import astropy.units as u
from poppy import AnalyticOpticalElement,utils
from poppy.poppy_core import BaseWavefront,PlaneType

class ShiftedGaussianAperture(AnalyticOpticalElement):
    """ Defines an ideal Gaussian apodized pupil aperture,
    or at least as much of one as can be fit into a finite-sized
    array
    The Gaussian's width must be set with either the fwhm or w parameters.
    Note that this makes an optic whose electric *field amplitude*
    transmission is the specified Gaussian; thus the intensity
    transmission will be the square of that Gaussian.
    Parameters
    ----------
    name : string
        Descriptive name
    fwhm : float, optional.
        Full width at half maximum for the Gaussian, in meters.
    w : float, optional
        Beam width parameter, equal to fwhm/(2*sqrt(ln(2))).
    pupil_diam : float, optional
        default pupil diameter for cases when it is not otherwise
        specified (e.g. displaying the optic by itself.) Default
        value is 3x the FWHM.
    """

    @utils.quantity_input(fwhm=u.meter, w=u.meter, pupil_diam=u.meter,shiftx=u.meter,shifty=u.meter)
    def __init__(self, name=None, fwhm=None, w=None, pupil_diam=None, shiftx=0, shifty=0, **kwargs):
        if fwhm is None and w is None:
            raise ValueError("Either the fwhm or w parameter must be set.")
        elif w is not None:
            self.w = w
        elif fwhm is not None:
            self.w = fwhm / (2 * np.sqrt(np.log(2)))

        if pupil_diam is None:
            pupil_diam = 3 * self.fwhm  # for creating input wavefronts
        self.pupil_diam = pupil_diam
        if name is None:
            name = "Gaussian aperture with fwhm ={0:.2f}".format(self.fwhm)

        self.shiftx = shiftx.to(u.meter).value
        self.shifty = shifty.to(u.meter).value
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, **kwargs)

    @property
    def fwhm(self):
        return self.w * (2 * np.sqrt(np.log(2)))

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the aperture.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        y, x = self.get_coordinates(wave)

        r = xp.sqrt((x-self.shiftx) ** 2 + (y-self.shifty) ** 2)

        transmission = np.exp((- (r / self.w.to(u.meter).value) ** 2))

        return transmission