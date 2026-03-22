"""
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from typing import Any, Callable, Dict, Optional, Tuple, Union
from src.math.coords import sub_angles


def plot_pattern(
    pattern_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    phi: Union[int, np.ndarray]=0, n_phi: Optional[int]=None,
    thea: Union[int, np.ndarray]=0, n_theta: Optional[int]=None,
    plot_type: str="rect_phi",
    ax: Optional[Axes]=None, ax_label: bool=True,
    **kwargs: Any
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Axes, Optional[AxesImage]]
]:
    """
    Evaluate and optionally visualize an angular radiation pattern.

    Args:
    -----
    pattern_fn: Function that computes the pattern response for arrays of 
                azimuth (`phi`) and elevation (`theta`) angles in degrees. 
                The function must return a 1-D array of responses for each 
                angle pair.
    phi:    Azimuth angles in degrees. If a scalar is given it is converted
            to a single-element array.
    n_phi:  If provided, overrides `phi` and generates `n_phi` uniformly 
            spaced azimuth angles in the range [-180, 180].
    theta:  Elevation angles in degrees. If a scalar is given it is converted
            to a single-element array.
    n_theta:    If provided, overrides `theta` and generates `n_theta`
                samples.
    plot_type:  Type of visualization. Supported values include rectangular 
                or polar cuts over azimuth or elevation, a 2-D heatmap, or 
                `"none"` to disable plotting.
    ax: Axis to draw the plot on. If not provided, a new axis is created.
    ax_label:   Whether to label axes and apply default axis limits.
    **kwargs:   Additional keyword arguments passed to the underlying 
                Matplotlib plotting function.

    Returns:
    --------
    Always returns `(phi, theta, v)` where `v` is the evaluated pattern.    \
    If plotting is enabled, also returns `(ax, image)` where `image`        \
    is the handle to the `imshow` result for 2-D plots (otherwise None).
    """
    # Create uniform angles
    if n_phi is not None:
        phi = np.linspace(-180, 180, n_phi)
    elif np.isscalar(phi):
        phi = np.array([phi], dtype=float)
    
    if n_theta is not None:
        theta = np.linspace(theta)
    elif np.isscalar(theta):
        theta = np.array([theta], dtype=float)
    
    n_phi, n_theta = len(phi), len(theta)

    # Create meshgrid of points, and compute pattern
    p_mat, t_mat = np.meshgrid(phi, theta)
    v = pattern_fn(p_mat.ravel(), t_mat.ravel()).reshape(n_theta, n_phi)

    # Get plot axis if not supplied
    if (ax is None) and (plot_type != 'none'):
        if plot_type in ('polar_phi','polar_theta'):
            ax = plt.axes(projection='polar')
        else:
            ax = plt.axes()
    
    # Regular plot
    image: Optional[AxesImage]=None
    if plot_type == 'rect_phi':
        ax.plot(phi, v.T, **kwargs)
        if ax_label:
            ax.set_xlabel('Azimuth (deg)')
            ax.set_xlim([-180, 180])
        
    elif plot_type == 'polar_phi':
        ax.plot(np.radians(phi), v.T, **kwargs)
        if ax_label:
            ax.set.xlabel('Azimuth (rad)')
    
    elif plot_type == 'rect_theta':
        ax.plot(theta, v, **kwargs)
        if ax_label:
            ax.set_xlabel('Elevation (deg)')
            ax.set_xlim([-90, 90])
    
    elif plot_type == 'rect_theta':
        ax.plot(theta, v, **kwargs)
        if ax_label:
            ax.set_xlabel('Elevation (rad)')
    
    elif plot_type == '2d':
        image = ax.imshow(np.flipud(v), extent=[
            float(np.min(phi)), float(np.max(phi)),
            float(np.min(theta)), float(np.max(theta))
        ], aspect='auto', **kwargs)

        if ax_label:
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Elevation (deg)')
    
    elif plot_type != 'none':
        raise ValueError(f"Unknown plot type `{plot_type}`")
    
    if plot_type == 'none':
        return phi, theta, v
    
    else:
        return phi, theta, v, ax, image



class ElementBase:
    """
    Base class representing an antenna or array element radiation pattern.
    Subclasses must implement the `response` method, which evaluates the
    element gain for given azimuth and elevation angles.
    """
    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the element response for the given angles. This method must
        be implemented by subclasses.

        Args:
        -----
        phi:    Azimuth angles in degrees.
        theta:  Elevation angles in degrees.

        Returns:
        --------
        Element gain (typically in dB) for each angle pair.
        """
        raise NotImplementedError("Response method not implemented")
    

    def compute_gain_mean(self, n_samples: int=1_000) -> float:
        """
        Estimate the average element gain over the sphere. The method samples 
        random angular directions and computes a weighted average of the 
        element response, accounting for the spherical surface element using 
        a `sin(theta)` weighting.

        Args:
        -----
        n_samples:  Number of random angle samples used in the Monte Carlo 
                    estimate.

        Returns:
        --------
        Estimated mean gain in dB.
        """
        # Generate random angles
        rng = np.random.default_rng()
        phi, theta = rng.uniform(-180, 180, n_samples), rng.uniform(0, 180, n_samples)

        # Compute Weights
        w = np.sin(theta * np.pi / 180)
        gain = np.average(10 ** (0.1 * self.response(phi, theta)), weights=w)
        return 10 * np.log10(gain)
    

    def plot_pattern(self, **kwargs: Any) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Axes, Optional[AxesImage]]
    ]:
        """
        Convenience wrapper for visualizing the element radiation pattern. 
        All keyword arguments are forwarded to the global `plot_pattern`
        utility using the element's `response` method.

        Returns:
        --------
        Same return values as `plot_pattern`.
        """
        return plot_pattern(self.response, **kwargs)



class ElementIsotropic(ElementBase):
    """
    Ideal isotropic antenna element. This element has uniform gain in all 
    directions, meaning its radiation pattern is constant regardless of 
    azimuth or elevation.
    """
    def __init__(self):
        """
            Initialize Element Isotropic Instance
        """
        super().__init__()
    

    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Return the isotropic gain pattern.

        Args:
        -----
        phi:    Azimuth angles in degrees.
        theta:  Elevation angles in degrees.

        Returns:
        --------
        Array of zeros representing constant 0 dB gain in all directions.
        """
        return np.zeros_like(phi)



class Element3GPP(ElementBase):
    """
    Directional antenna element based on the 3GPP antenna pattern model.
    The model describes horizontal and vertical attenuation relative to the 
    element boresight using quadratic approximations and maximum attenuation 
    limits as defined in 3GPP specifications.
    """
    def __init__(self,
        phi0: float=0.0, theta0: float=0.0, phibw: float=0.0, thetabw: float=0.0
    ):
        """
            Initialize 3GPP Antenna Instance
        """
        super().__init__()

        self.phi0, self.theta0 = phi0, theta0
        self.phibw, self.thetabw = phibw, thetabw

        self.gain_max = 0.0

        self.slav = 30.0    # Vertical side lobe
        self.Am = 30.0      # min gain

        self.calibrate()
    

    def response(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the directional gain using the 3GPP antenna model. The input 
        angles are rotated relative to the element boresight, and horizontal 
        and vertical attenuation components are applied according to the 
        specified beamwidth parameters.

        Args:
        -----
        phi:    Azimuth angles in degrees.
        theta:  Elevation angles in degrees.

        Returns:
        --------
        Element gain in dB for each input direction.
        """
        phi, theta = np.asarray(phi), np.asarray(theta)

        # Rotate the angles relative to element boresight.
        # """ Conversion from inclination to elevation angles """
        if self.phi0 != 0 or self.theta0 != 0:
            p, t = sub_angles(self.phi0, 90 - self.theta0, phi, 90 - theta)
        else:
            p, t = phi.copy(), theta.copy()
        
        # Put the phi from -180 to 180
        p = p % 360
        p = p - 360 * (p > 180)

        # Compute gains - vectorized operations
        Av = -np.minimum(12 * (t / self.thetabw) ** 2, self.slav) if self.thetabw > 0 else 0.0
        Ah = -np.minimum(12 * (p/self.phibw) ** 2, self.Am) if self.phibw > 0 else 0.0

        return self.gain_max - np.minimum(-Av - Ah, self.Am)
    

    def calibrate(self, n_samples: int=1_000) -> None:
        """
        Normalize the antenna pattern to achieve a desired average gain.

        This method estimates the mean gain of the current pattern and
        adjusts the internal maximum gain parameter so that the average
        response over the sphere becomes approximately 0 dB.

        Args:
        -----
        n_samples:  Number of samples used for the Monte Carlo estimation.
        """
        self.gain_max -= self.compute_gain_mean(n_samples)
