"""
"""
import numpy as np

from src.cfg.const import LIGHT_SPEED
from src.math.coords import spherical_to_cartesian, sub_angles
from src.simulate.antenna import plot_pattern, ElementIsotropic, ElementBase
from typing import Any, Dict, List, Optional, Tuple, Union



class ArrayBase:
    """
    Base class representing an antenna array.

    The class provides functionality for computing steering vectors,
    beamforming weights, and array radiation patterns based on the
    positions of individual elements and their element radiation pattern.
    """
    def __init__(self,
        element: Optional[ElementBase]=None, frequency: float=28e9,
        element_position: np.ndarray=np.array([[0, 0, 0]])
    ) -> None:
        """
            Initialize Array - Base Instance
        """
        self.element = element or ElementIsotropic()
        self.position = np.asarray(element_position, dtype=float)
        self.fc = frequency

        self._lam = LIGHT_SPEED / frequency
    

    def steering_vectors(self,
        phi: Union[float, np.ndarray], theta: Union[float, np.ndarray],
        include_element: bool=True, return_element_gain: bool=False
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute array steering vectors for specified directions.

        Args:
        -----
        phi:    Azimuth angle(s) in degrees.
        theta:  Elevation angle(s) in degrees.
        include_element:    If True, the element radiation pattern is 
                            included in the resulting steering vectors.
        return_element_gain:    If True, also return the element gain for 
                                each direction.

        Returns:
        --------
        Steering vector(s) for the specified direction(s). If                   \
        `return_element_gain` is True, the element gain values are returned as  \
        an additional array.
        """
        single_value = np.isscalar(phi) and np.isscalar(theta)

        phi = np.atleast_1d(phi).astype(float, copy=False)
        theta = np.atleast_1d(theta).astype(float, copy=False)

        n = len(phi)

        # Get unit vectors in the direction of the rays
        # """ Conversion elevation to inclination """
        u = spherical_to_cartesian(1.0, phi, 90.0 - theta)

        # Compute the delay along each path in wavelength
        # Using einsum for potential improved efficiency (numba alt coord use it)
        dly = np.einsum('ij,kj->ik', u, self.position) / self._lam

        # Phase rotation
        usv = np.exp(1j * 2 * np.pi * dly)

        # Add element pattern if requested
        if include_element: 
            gain = self.element.response(phi, theta)
            gain = 10 ** (0.05 * gain)
            usv *= gain[:, np.newaxis]
        else:
            gain = np.zeros(n, dtype=float)
        
        # Return appropriate format
        if return_element_gain:
            if single_value:
                return usv.ravel(), gain.item()
            return usv, gain
        
        else:
            if single_value:
                return usv.ravel()
            return usv
    

    def conjugate_beamforming(self,
        phi: Union[float, np.ndarray], theta: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Compute conjugate (matched filter) beamforming weights. The weights
        correspond to the normalized conjugate of the steering vector for the 
        specified direction.

        Args:
        -----
        phi:    Azimuth angle(s) in degrees.
        theta:  Elevation angle(s) in degrees.

        Returns:
        --------
        Beamforming weight vector(s) normalized to unit power.
        """
        single_val = np.isscalar(phi) and np.isscalar(theta)
        w = self.steering_vectors(phi, theta, include_element=False)

        wm = np.sqrt(np.sum(np.abs(w) ** 2, axis=1, keepdims=True))
        w = np.conj(w) / wm

        return w.ravel() if single_val else w
    

    def plot_pattern(
        weights: np.ndarray, include_element: bool=True, **kwargs
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, plt.Axes, Optional[plt.AxesImage]]
    ]:
        """
        Plot the array radiation pattern for a given set of weights.

        Args:
        -----
        weights:    Beamforming weights applied to the array elements.
        include_element:    Whether to include the element radiation pattern 
                            in the computed array response.
        **kwargs:   Additional arguments passed to the `plot_pattern` utility.

        Returns:
        --------
        Same return values as `plot_pattern`, including evaluated angles, \
            pattern values, and optionally the plotting objects.
        """
        w = np.asarray(weights)
        def pattern_fn(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
            sv = self.steering_vectors(phi, theta, include_element=include_element)

            # Handle broadcasting
            pattern = 20*np.log10(np.abs(sv @ w)) if sv.ndim==2 else 20*np.log10(np.abs(sv * w))
        
        return plot_pattern(pattern_fn, **kwargs)



class URA(ArrayBase):
    """
    Uniform Rectangular Array (URA).

    This array arranges antenna elements on a rectangular grid in
    two dimensions with constant spacing between neighboring elements.
    """
    def __init__(self, 
        n_antennas: Tuple[int, int],  
        separator: Optional[Tuple[float, float]] = None, 
        **kwargs: Any
    ) -> None:
        """
            Initialize Uniform Rectangular Array Instance
        """
        if separator is None:
            lam = LIGHT_SPEED / kwargs.get('frequency', 28e9)
            separator = (lam / 2, lam / 2)
        
        # Compute the antenna position
        ny, nz = n_antennas
        nat = ny * nz

        # Create indices
        y_ind = np.tile(np.arange(ny), nz)
        z_ind = np.repeat(np.arange(nz), ny)

        # Compute positions
        element_position = np.column_stack((
            np.zeros(nant_tot, dtype=float), y_ind * sep[0], z_ind * sep[1]
        ))

        # Super constructor
        super().__init__(elem_pos=elem_pos, **kwargs)



class RotatedArray(ArrayBase):
    """
    Wrapper for an antenna array with a rotated boresight direction.

    The class applies a coordinate transformation so that the underlying
    array appears rotated in azimuth and/or elevation relative to the
    global coordinate system.
    """
    def __init__(
        array: ArrayBase, phi0: float=0, theta0: float=0
    ):
        """
            Initialize Rotated Array Instance
        """
        self.array, self.phi0, self.theta0 = array,phi0,theta0
    

    def global_to_local(self,
        phi: np.ndarray, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert global angles to the local array coordinate system.

        Args:
        -----
        phi:    Global azimuth angles in degrees.
        theta:  Global elevation angles in degrees.

        Returns:
        --------
        Local azimuth and elevation angles relative to the array boresight.
        """
        phi1, theta1 = sub_angles(phi, 90 - theta, self.phi0, 90 - self.theta0)
        return phi1, theta1
    

    def steering_vectors(self,
        phi: Union[float, np.ndarray], theta: Union[float, np.ndarray], **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute steering vectors using the rotated coordinate system.

        Args:
        -----
        phi:    Global azimuth angle(s) in degrees.
        theta:  Global elevation angle(s) in degrees.
        **kwargs:   Additional arguments forwarded to the underlying array
                    `steering_vectors` method.

        Returns:
        --------
        Steering vectors computed in the rotated array frame.
        """
        p, t = self.global_to_local(np.atleast_1d(phi), np.atleast_1d(theta))
        return self.array.steering_vectors(p, t, **kwargs)


    def conjugate_beamforming(self,
        phi: Union[float,np.ndarray], theta: Union[float,np.ndarray]
    ) -> np.ndarray:
        """
        Compute steering vectors using the rotated coordinate system.

        Args:
        -----
        phi:    Global azimuth angle(s) in degrees.
        theta:  Global elevation angle(s) in degrees.
        **kwargs:   Additional arguments forwarded to the underlying array
                    `steering_vectors` method.

        Returns:
        --------
        Steering vectors computed in the rotated array frame.
        """
        p,t = self.global_to_local(np.atleast_1d(phi), np.atleast_1d(theta))
        return self.array.conjugate_beamforming(p, t)



def multi_sector_array(
    array0: ArrayBase, sector_type: str='azimuth',
    phi0: float=0.0, theta0: float=0.0,
    n_sectors: int=3
) -> List[RotatedArray]:
    """
    Create multiple sectorized arrays from a base array. Each sector is 
    implemented as a rotated version of the original array with a different 
    boresight orientation.

    Args:
    -----
    array0: Base array instance used for all sectors.
    sector_type:    Sectorization type. `'azimuth'` distributes sectors 
                    around the horizontal plane, while `'elevation'`
                    distributes them vertically.
    phi0:   Base azimuth orientation in degrees.
    theta0: Base elevation orientation in degrees.
    n_sectors:  Number of sectors to generate.

    Returns:
    --------
    List of rotated arrays representing each sector.
    """
    if sector_type == 'azimuth':
        pv = np.linspace(0, (n_sectors - 1) / n_sectors, n_sectors) * 360
        tv = np.full(n_sectors, theta0, dtype=float)
    
    elif sector_type == 'elevation':
        pv = np.full(n_sectors, phi0, dtype=float)
        tv = (np.linspace(1, n_sectors, n_sectors) / (n_sectors + 1) * 2 - 1) * 90

    else:
        raise ValueError(f"Unknown sectorization type `{sector_type}`")
    
    # Create list of arrays using list comprehension
    return [RotatedArray(array0, phi0=p, theta0=t) for p, t in zip(pv, tv)]
