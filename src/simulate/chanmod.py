"""
    Multi-path channel utilities for directional antenna simulations.

    This module defines a geometric multi-path channel representation and
    functions for evaluating directional path loss using antenna arrays
    and beamforming.
"""
import numpy as np
from src.cfg.data import LinkState
from typing import Any, List, Optional, Tuple, Union


class MultiPathChannel:
    """
    Container representing a geometric multi-path channel.

    The channel consists of multiple propagation rays, each defined by
    its path loss, delay, and angular parameters for departure and arrival.
    The class also provides utilities for computing aggregate channel
    metrics such as omnidirectional path loss and RMS delay spread.
    """
    n_angles = 4
    aoa_phi_ind     = 0
    aoa_theta_ind   = 1
    aod_phi_ind     = 2
    aod_theta_ind   = 3

    angle_name = ['AoA_Phi', 'AoA_Theta', 'AoD_Phi', 'AoD_Theta']
    large_path_loss = 250.0


    def __init__(self) -> None:
        """
            Initialize Multi - Path Channel Instance
        """
        # Parameters for each ray
        self.pl: np.ndarray=np.zeros(0, dtype=np.float32)
        self.dly: np.ndarray=np.zeros(0, dtype=np.float32)
        self.ang: np.ndarray=np.zeros((0, MultiPathChannel.n_angles), dtype=np.float32)

        self.link_state = LinkState.NO_LINK
    

    def compute_omni_path_loss(self) -> float:
        """
        Compute the effective omnidirectional path loss. The omnidirectional 
        path loss aggregates the contribution of all multi-path components by 
        summing their received powers in the linear domain.

        Returns:
        --------
        Effective omnidirectional path loss in dB. If no link exists,returns \
            infinity.
        """
        if self.link_state == LinkState.NO_LINK:
            return np.inf
        
        pl_min = np.min(self.pl)
        pl_lin = 10 ** (-0.1 * (self.pl - pl_min))
        pl_omni = pl_min - 10 * np.log10(np.sum(pl_lin))

        return float(pl_omni)


    def rms_delay(self) -> float:
        """
        Compute the RMS delay spread of the channel. The delay spread is 
        calculated as the weighted root-mean-square variation of path delays,
        where weights correspond to the received power of each path.

        Returns:
        --------
        RMS delay spread in the same time units as the stored delays. Returns 
        0 if the channel has no valid link.
        """
        if self.link_state == LinkState.NO_LINK:
            return 0.0
        
        # Compute the Weights
        pl_min = np.min(self.pl)
        w = 10 ** (-0.1 * (self.pl - pl_min))
        w = w / np.sum(w)

        # Compute Weighted RMS
        dly_mean = np.dot(w, self.dly)
        dly_rms = np.sqrt(np.dot(w, (self.dly - dly_mean) ** 2))

        return float(dly_rms)



def direction_path_loss(
    tx_array: 'ArrayBase', rx_array: 'ArrayBase',
    channel: MultiPathChannel,
    return_element_gain: bool=True, return_beamforming_gain: bool=False
) -> Union[float, Tuple[float, ...]]:
    """
    Compute effective directional path loss using beamforming. The function 
    evaluates transmit and receive steering vectors for all channel paths, 
    selects the strongest path based on element gains, and applies conjugate 
    beamforming toward that direction. The final effective path loss combines 
    contributions from all paths after beamforming.

    Args:
    -----
    tx_array:   Transmit antenna array.
    rx_array:   Receive antenna array.
    channel : Multi-path channel model containing path parameters.
    return_element_gain:    If True, return element radiation gains for each 
                            path.
    return_beamforming_gain:    If True, return beamforming gains for each
                                path.

    Returns:
    --------
    Effective path loss in dB. Additional outputs may include element   \
    gains and beamforming gains depending on the requested options.
    """
    if channel.link_state == LinkState.NO_LINK:
        pl_eff = MultiPathChannel.large_path_loss
        
        if not (return_beamforming_gain or return_element_gain):
            return pl_eff
    
        out: List[Any] = [pl_eff]
        if return_element_gain:
            out.extend([np.array(0), np.array(0)])
        
        if return_beamforming_gain:
            out.extend([np.array(0), np.array(0)])
        
        return tuple(out)
    
    # Get the angles of the path
    # """ Conversion from inclination to elevation angles """
    aod_theta = 90.0 - channel.ang[:, MultiPathChannel.aod_theta_ind]
    aod_phi = channel.ang[:, MultiPathChannel.aod_phi_ind]
    aoa_theta = 90.0 - channel.ang[:, MultiPathChannel.aoa_theta_ind]
    aoa_phi = channel.ang[:, MultiPathChannel.aoa_phi_ind]

    tx_sv, tx_elem_gain = tx_array.steering_vectors(
        aod_phi, aod_theta, include_element=True, return_element_gain=True
    )

    rx_sv, rx_elem_gain = rx_array.steering_vectors(
        aoa_phi, aoa_theta, include_element=True, return_element_gain=True
    )

    # Compute path loss with element gains
    pl_elem = channel.pl - tx_elem_gain - rx_elem_gain

    # Select the path with the lower path loss
    im = np.argmin(pl_elem)

    # Beamforming in the direction of rays
    wtx = np.conj(tx_sv[im, :])
    wtx /= np.sqrt(np.sum(np.abs(wtx) ** 2))
    wrx = np.conj(rx_sv[im, :])
    wrx /= np.sqrt(np.sum(np.abs(wrx) ** 2))

    # Compute the gain with both the element and BF gain
    tx_bf = 20 * np.log10(np.abs(tx_sv @ wtx))
    rx_bf = 20 * np.log10(np.abs(rx_sv @ wrx))
    pl_bf = channel.pl - tx_bf - rx_bf

    # Subtract the TX and RX element gains
    tx_bf -= tx_elem_gain
    rx_bf -= rx_elem_gain
    
    # Compute effective path loss
    pl_min = np.min(pl_bf)
    pl_lin = 10 ** (-0.1 * (pl_bf - pl_min))
    pl_eff = pl_min - 10 * np.log10(np.sum(pl_lin))
    
    # Build output
    out = [pl_eff]
    if return_elem_gain:
        out.extend([tx_elem_gain, rx_elem_gain])
    
    if return_bf_gain:
        out.extend([tx_bf, rx_bf])
    
    return tuple(out) if len(out) > 1 else out[0]



def direction_path_loss_multi_sector(
    tx_array_list: List['ArrayBase'], rx_array_list: List['ArrayBase'],
    channel: MultiPathChannel,
    return_element_gain: bool=True, return_beamforming_gain: bool=True,
    return_array_indices: bool=True
) -> Union[float,Tuple[float,...]]:
    """
    Compute directional path loss using sectorized transmit and receive arrays.
    The function evaluates all combinations of transmit and receive sector 
    arrays, selects the pair that minimizes the effective path loss, and then 
    performs beamforming toward the strongest propagation direction.

    Args:
    -----
    tx_array_list:  List of transmit sector arrays.
    rx_array_list:  List of receive sector arrays.
    channel:    Multi-path channel model.
    return_element_gain:    If True, return element radiation gains for the 
                            selected sector pair.
    return_beamforming_gain:    If True, return beamforming gains for the 
                                selected sector pair.
    return_array_indices:   If True, return the indices of the selected transmit 
                            and receive sector arrays.

    Returns:
    --------
    Effective path loss in dB. Additional outputs may include sector indices,\
        element gains, and beamforming gains depending on the selected options.
    """
    if channel.link_state == LinkState.NO_LINK:
        pl_eff = MultiPathChannel.large_path_loss
        
        if not (return_beamforming_gain or return_element_gain or return_array_indices):
            return pl_eff
        
        out: List[Any] = [pl_eff]
        if return_arr_ind:
            out.extend([0, 0])

        if return_elem_gain:
            out.extend([np.array(0), np.array(0)])
        
        if return_bf_gain:
            out.extend([np.array(0), np.array(0)])
        
        return tuple(out)
    
    # Get the angles of the path
    aod_theta = 90.0 - channel.ang[:, MultiPathChannel.aod_theta_ind]
    aod_phi = channel.ang[:, MultiPathChannel.aod_phi_ind]
    aoa_theta = 90.0 - channel.ang[:, MultiPathChannel.aoa_theta_ind]
    aoa_phi = channel.ang[:, MultiPathChannel.aoa_phi_ind]

    # Initialize tracking variables
    pl_min = MultiPathChannel.large_path_loss
    im = -1
    ind_rx, ind_tx = 0, 0
    rx_sv_opt, tx_sv_opt = None, None
    rx_elem_gain_opt, tx_elem_gain_opt = None, None

    # Loop over the array combinations to find the best array
    for irx, rxa in enumerate(rx_array_list):
        for itx, txa in enumerate(rx_array_list):
            tx_sv, tx_elem_gain = txa.steering_vectors(
                aod_phi, aod_theta, include_element=True, return_element_gain=True
            )
            rx_sv, rx_elem_gain = rxa.steering_vectors(
                aoa_phi, aoa_theta, include_element=True, return_element_gain=True
            )
            
            # Compute path loss with element gains
            pl_elem = chan.pl - tx_elem_gain - rx_elem_gain
            pl_mini = np.min(pl_elem)
            
            if pl_mini < pl_min:
                pl_min = pl_mini
                im = np.argmin(pl_elem)
                tx_sv_opt = tx_sv
                rx_sv_opt = rx_sv
                tx_elem_gain_opt = tx_elem_gain
                rx_elem_gain_opt = rx_elem_gain
                ind_rx = irx
                ind_tx = itx
    
    # Beamforming in the optimal direction
    wtx = np.conj(tx_sv_opt[im, :])
    wtx /= np.sqrt(np.sum(np.abs(wtx) ** 2))
    wrx = np.conj(rx_sv_opt[im, :])
    wrx /= np.sqrt(np.sum(np.abs(wrx) ** 2))

    # Compute the gain with both the element and BF gain
    tx_bf = 20 * np.log10(np.abs(tx_sv_opt @ wtx))
    rx_bf = 20 * np.log10(np.abs(rx_sv_opt @ wrx))
    pl_bf = channel.pl - tx_bf - rx_bf
    
    # Subtract the TX and RX element gains
    tx_bf -= tx_elem_gain_opt
    rx_bf -= rx_elem_gain_opt
    
    # Compute effective path loss
    pl_min = np.min(pl_bf)
    pl_lin = 10 ** (-0.1 * (pl_bf - pl_min))
    pl_eff = pl_min - 10 * np.log10(np.sum(pl_lin))
    
    # Build output
    out = [pl_eff]
    if return_arr_ind:
        out.extend([ind_tx, ind_rx])
    
    if return_elem_gain:
        out.extend([tx_elem_gain_opt, rx_elem_gain_opt])
    
    if return_bf_gain:
        out.extend([tx_bf, rx_bf])
    
    return tuple(out) if len(out) > 1 else out[0]
