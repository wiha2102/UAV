"""
src/math/coords.py
------------------
Script with mathematical operations specifically for coordination system change
back and forth between spherical coordinates and cartesian coordinates as well
as add / subtract angles.

    - Numba accelerated (may be slower for this due to additional overhead) 
      longer compilation time

This script focusing simply on simplistic mathematical operations.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange, vectorize, float64
from typing import Final, Optional, Tuple, Union


# ----------===== Common Values / Typing =====---------- #

AF      = npt.NDArray[np.floating]
AF64    = npt.NDArray[np.float64]

DEG2RAD : Final[np.float64] = np.float64(np.pi / 180.0)
RAD2DEG : Final[np.float64] = np.float64(180.0 / np.pi)
EPS     : Final[np.float64] = np.float64(1e-12)

# Precompute common values
ONE     : Final[np.float64] = np.float64(1.0)
NEG_ONE : Final[np.float64] = np.float64(-1.0)
ZERO    : Final[np.float64] = np.float64(0.0)


# ============================================================
#       Utilities Functions
# ============================================================

@njit(fastmath=True, cache=True, inline='always')
def _as_1d_array_numba(x: np.ndarray) -> np.ndarray:
    """
    """
    return np.array([x],dtype=np.float64) if x.ndim==0 else x.ravel()


def _as_1d_array(x: Union[AF,float]) -> AF64:
    """
    """
    if isinstance(x,(int,float)): return np.array([x],np.float64)

    array = np.asarray(x, dtype=np.float64)
    return array.reshape(1) if array.ndim==0 else array.ravel()


@vectorize([float64(float64)], nopython=True, cache=True)
def _clip_to_unit(x: float64) -> float64:
    """
    """
    if x > ONE: return ONE
    if x < NEG_ONE: return NEG_ONE
    return x


# ============================================================
#       CARTESIAN TO SPHERICAL
# ============================================================

@njit(fastmath=True, parallel=True, cache=True)
def _cart2sph_kernel(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    r: np.ndarray, p: np.ndarray, t: np.ndarray
) -> None:
    """
    """
    n = len(x)
    for i in prange(n):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        
        r2 = xi * xi + yi * yi
        z2 = zi * zi        
        if r2 < EPS and z2 < EPS:
            r[i] = ZERO
            p[i] = ZERO
            t[i] = ZERO
            continue
        
        ri = np.sqrt(r2 + z2)
        r[i] = ri
        p[i] = np.arctan2(yi, xi)
        
        ct = zi / ri
        ct = max(NEG_ONE, min(ONE, ct))
        t[i] = np.arccos(ct)


def cartesian_to_spherical(dvec: AF) -> Tuple[AF64, AF64, AF64]:
    """
    """
    if hasattr(dvec, 'shape'):
        
        if dvec.ndim == 1:
            if dvec.size != 3:
                raise ValueError("dvec must have 3 elements when 1D")
            array = dvec.reshape(1, 3).astype(np.float64, copy=False)
        
        elif dvec.ndim == 2:
            if dvec.shape[1] != 3:
                raise ValueError("dvec must have shape (N, 3)")
            array = dvec.astype(np.float64, copy=False)
        
        else:
            raise ValueError("dvec must be 1D or 2D")
    
    else:
        array = np.asarray(dvec, dtype=np.float64)
        
        if array.ndim == 1:
            if array.size != 3:
                raise ValueError("dvec must have 3 elements when 1D")
            array = array.reshape(1, 3)
        
        elif array.ndim == 2 and array.shape[1] != 3:
            raise ValueError("dvec must have shape (N, 3)")
    
    n = array.shape[0]
    x, y, z = array[:,0], array[:,1], array[:,2]

    # Preallocation of output arrays to avoid copying
    r = np.empty(n, dtype=np.float64)
    p = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)

    _cart2sph_kernel(x, y, z, r, p, t)

    p *= RAD2DEG
    t *= RAD2DEG
    
    return r, p, t


# ============================================================
#       SPHERICAL TO CARTESIAN
# ============================================================

@njit(fastmath=True, parallel=True, cache=True)
def _sph2cart_kernel(
    r: np.ndarray, p: np.ndarray, t: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> None:
    """
    """
    n = len(r)
    for i in prange(n):
        
        ri = r[i]
        pi = p[i]
        ti = t[i]
        
        # Compute trig functions
        cp = np.cos(pi)
        sp = np.sin(pi)
        ct = np.cos(ti)
        st = np.sin(ti)
        
        ri_st = ri * st
        
        x[i] = ri_st * cp
        y[i] = ri_st * sp
        z[i] = ri * ct

def spherical_to_cartesian(radius: AF, phi: AF, theta: AF) -> AF64:
    """
    """
    r = _as_1d_array(radius)
    p = _as_1d_array(phi)
    t = _as_1d_array(theta)
    
    n = max(r.size, p.size, t.size)
    
    # Broadcasting
    if r.size == 1 and r.size < n:
        r_val = r[0]
        r = np.full(n, r_val, dtype=np.float64)
    
    elif r.size != n:
        r = np.broadcast_to(r, (n,))
    
    if p.size == 1 and p.size < n:
        p_val = p[0]
        p = np.full(n, p_val, dtype=np.float64)
    
    elif p.size != n:
        p = np.broadcast_to(p, (n,))
    
    if t.size == 1 and t.size < n:
        t_val = t[0]
        t = np.full(n, t_val, dtype=np.float64)
    
    elif t.size != n:
        t = np.broadcast_to(t, (n,))
    
    p = p * DEG2RAD
    t = t * DEG2RAD
    
    # Pre-allocate outputs
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    
    _sph2cart_kernel(r, p, t, x, y, z)
    
    # Stacking
    if n == 1:
        return np.array([[x[0], y[0], z[0]]], dtype=np.float64)
    
    return np.column_stack((x, y, z))


# ============================================================
#       Angle Combinations (Rotation)
# ============================================================

@njit(fastmath=True, cache=True, inline='always')
def _rotation_matrix_elements(
    cp1: float64, sp1: float64, ct1: float64, st1: float64, inv: bool
) -> Tuple[float64, ...]:
    """
    """
    if not inv: 
        return (
        cp1 * ct1, -sp1, cp1 * st1,
        sp1 * ct1, cp1, sp1 * st1,
        -st1, ZERO, ct1
    )
    else: 
        return (
        cp1 * ct1, sp1 * ct1, -st1,
        -sp1, cp1, ZERO,
        cp1 * st1, sp1 * st1, ct1
    )


@njit(fastmath=True, parallel=True, cache=True)
def _angle_rotation_kernel(
    p0: np.ndarray, t0: np.ndarray, p1: np.ndarray, t1: np.ndarray,
    inv: bool=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    n = len(p0)
    p_out = np.empty(n, dtype=np.float64)
    t_out = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
    
        cp0 = np.cos(p0[i])
        sp0 = np.sin(p0[i])
        ct0 = np.cos(t0[i])
        st0 = np.sin(t0[i])
        
        cp1 = np.cos(p1[i])
        sp1 = np.sin(p1[i])
        ct1 = np.cos(t1[i])
        st1 = np.sin(t1[i])
        
        x0 = st0 * cp0
        y0 = st0 * sp0
        z0 = ct0
        
        # Rotation matrix elements
        if not inv:
            m00, m01, m02 = cp1 * ct1, -sp1, cp1 * st1
            m10, m11, m12 = sp1 * ct1, cp1, sp1 * st1
            m20, m21, m22 = -st1, ZERO, ct1
        else:
            m00, m01, m02 = cp1 * ct1, sp1 * ct1, -st1
            m10, m11, m12 = -sp1, cp1, ZERO
            m20, m21, m22 = cp1 * st1, sp1 * st1, ct1
        
        # Apply rotation
        x = m00 * x0 + m01 * y0 + m02 * z0
        y = m10 * x0 + m11 * y0 + m12 * z0
        z = m20 * x0 + m21 * y0 + m22 * z0
        
        if z > ONE:
            z = ONE

        elif z < NEG_ONE:
            z = NEG_ONE
        
        # Convert back to spherical
        p_out[i] = np.arctan2(y, x)
        t_out[i] = np.arccos(z)
    
    return p_out, t_out


def _combine_angles(
    p0: AF64, t0: AF64, p1: AF64, t1: AF64,
    inv: bool=False
) -> Tuple[AF64, Af64]:
    """
    """
    p0 = np.asarray(p0, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    
    shape = p0.shape

    p0_rad = p0 * DEG2RAD
    t0_rad = t0 * DEG2RAD
    p1_rad = p1 * DEG2RAD
    t1_rad = t1 * DEG2RAD
    
    p0_flat = p0_rad.ravel()
    t0_flat = t0_rad.ravel()
    p1_flat = p1_rad.ravel()
    t1_flat = t1_rad.ravel()
    
    phi_rad, theta_rad = _angle_rotation_kernel(
        p0_flat, t0_flat, p1_flat, t1_flat, inv
    )
    
    # Convert back to degrees and reshape (very irritable bug)
    phi = phi_rad.reshape(shape) * RAD2DEG
    theta = theta_rad.reshape(shape) * RAD2DEG
    
    return phi, theta


def add_angles(phi0:AF, theta0:AF, phi1:AF, theta1:AF) -> Tuple[AF64,AF64]:
    """
    """
    return _combine_angles(phi0, theta0, phi1, theta1, inv=False)


def sub_angles(phi0:AF, theta0:AF, phi1:AF, theta1:AF) -> Tuple[AF64,AF64]:
    """
    """
    return _combine_angles(phi0, theta0, phi1, theta1, inv=True)


@njit(fastmath=True, parallel=True, cache=True)
def batch_cartesian_to_spherical(dvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    n = dvec.shape[0]
    x = dvec[:, 0]
    y = dvec[:, 1]
    z = dvec[:, 2]
    
    r = np.empty(n, dtype=np.float64)
    p = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    
    _cart2sph_kernel(x, y, z, r, p, t)
    p *= RAD2DEG
    t *= RAD2DEG
    
    return r, p, t


@njit(fastmath=True, parallel=True, cache=True)
def batch_spherical_to_cartesian(r: np.ndarray, p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    """
    n = len(r)
    
    # Convert to radians
    p_rad = p * DEG2RAD
    t_rad = t * DEG2RAD
    
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    
    _sph2cart_kernel(r, p_rad, t_rad, x, y, z)
    result = np.empty((n, 3), dtype=np.float64)
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = z
    
    return result
