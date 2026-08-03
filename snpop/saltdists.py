"""
Module related to details of SALT distributions
"""
from __future__ import absolute_import, print_function
__all__ = ['SALT2_MMDist', 'SALT2_SK16', 'sample_SALT2_SK16_hybrid',
          'Nicolas21DriftingStretch', 'Nicolas21AsymmetricStretch']
import numpy as np
from scipy.stats import norm
from tdaspop import double_gaussian, double_gaussian_pdf, GMMSampler

def  sample_SALT2_SK16_hybrid(num_sn, model1, model2, probs=[0.5, 0.5],
                              rng=np.random.RandomState(1)):
    """
    Hybrid  model scheme meant to weight different modes of intrinsic dispersion

    Parameters
    ----------
    num_sn : int
        number of samples requested
    model1 : string
        must be an available model name in `SALT2_SK16`
    model2 : string
        must be an available model name in `SALT2_SK16`
    probs : sequence of size 2
        should have two floats. (prob of being model 1, and prob of being model2)
    rng : `np.random.RandomState`
        Instance of randomstate


    Returns
    -------
    samples : `np.ndarray` of shape (num_sn, 2)
    samples of x1 and c in  samples[:, 0] and samples[:, 1] respectively.
    """
    sk1 = SALT2_SK16.from_model_name(model1)
    sk2 = SALT2_SK16.from_model_name(model2)
    
    p1 = probs[0]
    num1 = np.int(np.floor(num_sn *p1))
    num2 = np.int(num_sn - num1)

    return np.vstack((sk1.sample(num1), sk2.sample(num2)))

class SALT2_SK16(object):
    """
    Implementation of distribution of x1, and c from  Scolnic and Kessler,
    ApJ, 2016.

    Parameters
    ----------
    csigmalow : float
        sigma for lower values of the double gaussian for c
    csigmahigh : float
        sigma for higher values of the double gaussian for c
    cmode : float
        mode of the c distribution
    x1sigmalow : float
        sigma for lower values of the double gaussian for x1
    x1sigmahigh : float
        sigma for higher values of the double gaussian for x1
    x1mode : float
        mode of the x1 distribution
    rng : Instance of `np.random.RandomState`
    """
    def __init__(self, csigmalow, csigmahigh, cmode, x1sigmalow, x1sigmahigh,
                 x1mode, rng=np.random.RandomState()):
        self.csigmalow = csigmalow
        self.csigmahigh = csigmahigh
        self.cmode = cmode
        self.x1sigmalow = x1sigmalow
        self.x1sigmahigh = x1sigmahigh
        self.x1mode = x1mode
        self.rng = rng


    @classmethod
    def from_model_name(cls, model_name, rng=np.random.RandomState(1)):
        """ Implementations for models calculated from all surveys and G10,
        C11 scatter, and lowz surveys with G10 and C11 scatter. Set these
        models by name by using:

        All g10 : 'sk16_allz_g10'
        All c11 : 'sk16_allz_c11'
        Lowz g10 : 'sk16_lowz_g10'
        Lowz c11 : 'sk16_lowz_c11'
        """
        if model_name.lower() == "sk16_allz_g10":
            cm  = -0.043
            csl = 0.052
            csh = 0.107
            xm  = 0.945
            xsl = 1.553
            xsh = 0.257

        elif model_name.lower() == "sk16_allz_c11":
            cm  = -0.062
            csl = 0.032
            csh = 0.113
            xm  = 0.938
            xsl = 1.551
            xsh = 0.26

        elif model_name.lower() == "sk16_lowz_g10":
            cm  = -0.055
            csl = 0.023
            csh = 0.150
            xm  = 0.436
            xsl = 3.118
            xsh = 0.724

        elif model_name.lower() == "sk16_lowz_c11":
            cm  = -0.069
            csl = 0.023
            csh = 0.083
            xm  = 0.436
            xsl = 3.118
            xsh = 0.724
        else:
            raise NotImplementedError(f'class method for ks16 for Model name {model_name} not implemented yet')
        return cls(csl, csh, cm, xsl, xsh, xm, rng)

    def pdf(self, x):
        """
        Evaluate the pdf at x:

        Parameters
        ----------
        x : `np.ndarray`, shape (N, 2)
            parmeters describing the population, with
            x[:, 0] = c, x[:, 1] = x1

        Returns
        -------
        pdf : `np.ndarray` of shape (N, 1)
        """
        c = x[:, 0]
        x1 = x[:, 1]
        pc = double_gaussian_pdf(c, self.cmode,
                                 self.csigmalow,
                                 self.csigmahigh)
        px1 = double_gaussian_pdf(x1, self.x1mode,
                                 self.x1sigmalow,
                                 self.x1sigmahigh)

        pdf = pc * px1

        return pdf

    def sample(self, num_sn):
        """
        Draw samples from the distribution specified by the class

        Parameters
        ----------
        num_sn : int
            number of SN

        Returns
        -------
        samples : `np.ndarray` with shape = (num_sn, 2)
        Contains the x1 samples in samples[:, 0] and c samples in samples[:, 1]
        """
        c = double_gaussian(self.cmode, self.csigmalow, self.csigmahigh, size=num_sn, rng=self.rng)
        x1 = double_gaussian(self.x1mode, self.x1sigmalow, self.x1sigmahigh, size=num_sn, rng=self.rng)
        X = np.zeros(shape=(num_sn, 2))
        X[:, 0] = x1
        X[:, 1] = c
        return X  




def SALT2_MMDist(numSN, rng=None,
                 cm=-0.0474801042369, cs1=0.0965032273527, cs2=0.042844366359,
                 x1m=0.872727291354, x1s1=0.358731835038, x1s2=1.42806797468,
                 mm=10.701690617, ms1=0.334359086569, ms2=1.0750402101,
                 mBm=-19.0199168813, mc=-0.0838387899933, mt=10.,
                 cc=3.20907949118, cx1=-0.137042055737):
    """
    Generates "noise"-free mB, x1, c. Trained on JLA SALT2-4 SDSS z < 0.2 and
    SNLS z < 0.5. Very simple model with linear alpha/beta and same
    distribution irrspective of host-mass. mB needs h=0.7 distance modulus
    added to it.

    From D. Rubin

    Parameters
    ----------
    numSN : int
        number of SN parameter sets to draw
    rng : instance of `np.random.RandomState`, defaults to `None`
        random state used to draw samples. If `None`, a new
        `np.random.RandomState` is created and the draw is not reproducible
        across calls.
    """
    if rng is None:
        rng = np.random.RandomState()

    color = double_gaussian(mode=cm, sigmam=cs2, sigmap=cs1, size=numSN, rng=rng)
    x1 = double_gaussian(mode=x1m, sigmam=x1s2, sigmap=x1s1, size=numSN, rng=rng)
    mass = double_gaussian(mode=mm, sigmam=ms2, sigmap=ms1, size=numSN, rng=rng)

    mB = mBm + mc * (mass > mt) + cc * color + cx1 * x1

    return mB, x1, color, mass


class Nicolas21DriftingStretch(object):
    """
    Model for the redshift evolution of the SALT2 stretch (x1) distribution
    of SNe Ia from Nicolas et al. 2021, A&A 649, A74 (arXiv:2005.09441),
    Eqs. 1 and 2.

    The underlying stretch distribution is a mixture of a young population
    with a single high-stretch Gaussian mode, and an old population that
    shares that same mode plus an exclusive low-stretch mode (Eq. 2). The
    fraction of young SNe Ia evolves with redshift following the specific
    star formation rate (Eq. 1).

    Parameters
    ----------
    K : float, defaults to 0.87
        normalization of the young-fraction evolution (Eq. 1)
    mu1, sigma1 : float
        mean and standard deviation of the high-stretch mode shared by the
        young and old populations
    mu2, sigma2 : float
        mean and standard deviation of the low-stretch mode exclusive to
        the old population
    a : float
        weight of the high-stretch mode within the old population
    """
    def __init__(self, K=0.87, mu1=0.37, sigma1=0.61, mu2=-1.22, sigma2=0.56,
                 a=0.51):
        self.K = K
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.a = a

    @classmethod
    def from_sample(cls, sample_name='fiducial'):
        """
        Best-fit parameters from Table 2 of Nicolas et al. 2021 for the
        base drifting stretch model (Eq. 2).

        Parameters
        ----------
        sample_name : {'fiducial', 'conservative', 'snfactory'}
        """
        name = sample_name.lower()
        if name == 'fiducial':
            mu1, sigma1, mu2, sigma2, a = 0.37, 0.61, -1.22, 0.56, 0.51
        elif name == 'conservative':
            mu1, sigma1, mu2, sigma2, a = 0.38, 0.60, -1.26, 0.53, 0.47
        elif name == 'snfactory':
            mu1, sigma1, mu2, sigma2, a = 0.41, 0.55, -1.38, 0.44, 0.48
        else:
            raise NotImplementedError(f'sample_name {sample_name} not implemented')
        return cls(K=0.87, mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2, a=a)

    def delta(self, z):
        """
        Fraction of young SNe Ia at redshift `z` (Eq. 1).
        """
        z = np.asarray(z, dtype=float)
        return 1. / (self.K ** -1 * (1. + z) ** (-2.8) + 1.)

    def weights(self, z):
        """
        Mixture weights (w1, w2) of the shared high-stretch mode and the
        old-population-only low-stretch mode at redshift `z`, obtained by
        collapsing Eq. 2 into an equivalent 2-component mixture:
        w1 = delta(z) + (1 - delta(z)) * a, w2 = (1 - delta(z)) * (1 - a).
        """
        d = self.delta(z)
        w1 = d + (1. - d) * self.a
        w2 = (1. - d) * (1. - self.a)
        return w1, w2

    def pdf(self, x1, z):
        """
        Evaluate the pdf of Eq. 2 at stretch values `x1` and redshifts `z`.
        """
        w1, w2 = self.weights(z)
        return (w1 * norm.pdf(x1, self.mu1, self.sigma1)
                + w2 * norm.pdf(x1, self.mu2, self.sigma2))

    def sample(self, z, rng=None):
        """
        Draw one stretch sample per input redshift from the mixture in
        Eq. 2, using `tdaspop.GMMSampler` for the per-SN two-component
        Gaussian mixture defined by `weights` at that SN's redshift.

        Parameters
        ----------
        z : float or `np.ndarray`
            redshift(s) at which to draw stretch samples
        rng : `np.random.RandomState`, defaults to `None`
            if `None`, a new `np.random.RandomState` is created and draws
            are not reproducible across calls.

        Returns
        -------
        x1 : float or `np.ndarray`
            stretch sample(s), one per input redshift
        """
        if rng is None:
            rng = np.random.RandomState()

        zarr = np.asarray(z, dtype=float)
        scalar_input = zarr.ndim == 0
        zvals = np.atleast_1d(zarr)

        means = np.array([[self.mu1], [self.mu2]])
        covariances = np.array([[[self.sigma1 ** 2]], [[self.sigma2 ** 2]]])

        x1 = np.zeros(len(zvals))
        for i, zi in enumerate(zvals):
            w1, w2 = self.weights(zi)
            sampler = GMMSampler(weights=[w1, w2], means=means,
                                 covariances=covariances)
            X, _ = sampler.sample(size=1, rng=rng)
            x1[i] = X[0, 0]

        return x1[0] if scalar_input else x1


class Nicolas21AsymmetricStretch(object):
    """
    Per-survey asymmetric Gaussian model of the SALT2 stretch (x1)
    distribution, from Table 4 of Nicolas et al. 2021, A&A 649, A74
    (arXiv:2005.09441).

    Note: the paper's own model comparison (their Table 3 / Fig. 7) finds
    this redshift-independent, sample-based model a significantly worse
    description of the data than the drifting mixture model of Eq. 2
    (`Nicolas21DriftingStretch`); it is provided here as it was fit and
    reported per survey in Table 4.

    Parameters
    ----------
    mu0 : float
        mode of the stretch distribution
    sigmam : float
        standard deviation for x1 < mu0
    sigmap : float
        standard deviation for x1 > mu0
    """
    def __init__(self, mu0, sigmam, sigmap):
        self.mu0 = mu0
        self.sigmam = sigmam
        self.sigmap = sigmap

    @classmethod
    def from_survey(cls, survey_name):
        """
        Best-fit parameters from Table 4 of Nicolas et al. 2021.

        Parameters
        ----------
        survey_name : {'snfactory', 'sdss', 'ps1', 'snls', 'hst'}
        """
        name = survey_name.lower()
        if name == 'snfactory':
            mu0, sigmam, sigmap = 0.68, 1.34, 0.41
        elif name == 'sdss':
            mu0, sigmam, sigmap = 0.72, 1.31, 0.42
        elif name == 'ps1':
            mu0, sigmam, sigmap = 0.38, 1.01, 0.52
        elif name == 'snls':
            mu0, sigmam, sigmap = 1.22, 1.41, 0.15
        elif name == 'hst':
            mu0, sigmam, sigmap = 0.11, 0.76, 0.79
        else:
            raise NotImplementedError(f'survey_name {survey_name} not implemented')
        return cls(mu0, sigmam, sigmap)

    def pdf(self, x1):
        """
        Evaluate the pdf of the asymmetric stretch distribution at `x1`.
        """
        return double_gaussian_pdf(x1, self.mu0, self.sigmam, self.sigmap)

    def sample(self, size, rng=None):
        """
        Draw stretch samples from the asymmetric distribution.

        Parameters
        ----------
        size : int
            number of samples requested
        rng : `np.random.RandomState`, defaults to `None`
            if `None`, a new `np.random.RandomState` is created and the
            draw is not reproducible across calls.
        """
        if rng is None:
            rng = np.random.RandomState()
        return double_gaussian(mode=self.mu0, sigmam=self.sigmam,
                               sigmap=self.sigmap, size=size, rng=rng)
