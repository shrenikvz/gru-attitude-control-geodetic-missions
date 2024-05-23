'''
Required functions

Author: Shrenik Zinage, Vrushabh Zinage
'''

# Required libraries
import jax.numpy as jnp                             
import numpy as np                                  
from scipy.signal import hamming
from scipy.fft import fft, ifft

def create_sequences(data, sequence_length):

        xs, ys = [], []

        for i in range(len(data) - sequence_length):

            x = data[i:(i + sequence_length)]
            y = data[i + sequence_length]
            xs.append(x)
            ys.append(y)

        return jnp.array(xs), jnp.array(ys)

def dataloader(arrays, batch_size):

        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = np.arange(dataset_size)

        while True:

            perm = np.random.permutation(indices)
            start = 0
            end = batch_size

            while end <= dataset_size:

                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end
                end = start + batch_size

def attitude_dynamics(attitude_rates, moments_of_inertia, control_input, disturbance,disturbance_estimate):

    p, q, r = attitude_rates
    Ix, Iy, Iz = moments_of_inertia
    tau_p, tau_q, tau_r = control_input

    # Convert the single-element disturbance into a 3-element array
    disturbance_3d = np.array([disturbance, disturbance, disturbance])
    disturbance_estimate_3d = np.array([disturbance_estimate, disturbance_estimate, disturbance_estimate])

    # Incorporate disturbances into external torques
    tau_p += disturbance_3d[0]-disturbance_estimate_3d[0]
    tau_q += disturbance_3d[1]-disturbance_estimate_3d[1]
    tau_r += disturbance_3d[2]-disturbance_estimate_3d[2]

    # Euler's rotational equations
    p_dot = (tau_p + (Iz - Iy) * q * r) / Ix
    q_dot = (tau_q + (Ix - Iz) * p * r) / Iy
    r_dot = (tau_r + (Iy - Ix) * p * q) / Iz

    return np.array([p_dot, q_dot, r_dot]), np.array([tau_p, tau_q, tau_r])

def euler_rates(attitude_rates, euler_angles):

    p, q, r = attitude_rates
    phi, theta, _ = euler_angles

    phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    return np.array([phi_dot, theta_dot, psi_dot])

def psd_from_fft(z, n, m, time_step):

    '''
    # z : discrete time signal
    # n : length of z
    # m : hamming variable
    # time_step : time step
    '''

    # Compute the FFT of the input signal
    z_frequency = fft(z)
    
    # Calculate the power spectral density
    R = z_frequency * np.conj(z_frequency) / n
    fr = np.arange(n) / n * (1 / time_step)
    P = 2 * R * time_step
    
    # Generate and normalize the Hamming window
    w = hamming(m)
    w = w / np.sum(w)
    
    # Prepare the window for convolution
    w = np.concatenate((w[int(np.ceil((m+1)/2))-1:], np.zeros(n-m), w[:int(np.ceil((m+1)/2))-1]))
    w = fft(w)
    
    # Convolve the window with the PSD estimate
    pavg = fft(P)
    pavg = ifft(w * pavg)
    
    # Extract the positive frequency components and normalize
    S = np.abs(pavg[:int(np.ceil(n/2))])
    F = fr[:int(np.ceil(n/2))]
    S = S / (2 * np.pi)
    W = 2 * np.pi * F

    return S, W
