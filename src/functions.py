'''
Required functions

Author: Shrenik Zinage, Vrushabh Zinage
'''

# Required libraries
import jax.numpy as jnp                             # NumPy-like API for JAX
import numpy as np                                  # NumPy library for numerical computing

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

def plot_power_spectral_density(time_series, sample_spacing=1.0):

    # Compute the Power Spectral Density (PSD)
    fft_result = np.fft.fft(time_series)
    psd = np.abs(fft_result) ** 2

    # Compute the frequency bins
    sample_count = len(time_series)
    frequencies = np.fft.fftfreq(sample_count, sample_spacing)

    # Only plot the positive frequencies
    positive_frequencies = frequencies[:sample_count // 2]
    positive_psd = psd[:sample_count // 2]

    return positive_frequencies, positive_psd

def plot_frequency_spectrum(time_series, sample_spacing=1.0):

    # Compute the FFT
    fft_result = np.fft.fft(time_series)

    # Get the magnitudes
    magnitudes = np.abs(fft_result)

    # Compute the frequency bins
    sample_count = len(time_series)
    frequencies = np.fft.fftfreq(sample_count, sample_spacing)

    # Only plot the positive frequencies
    positive_frequencies = frequencies[:sample_count // 2]
    positive_magnitudes = magnitudes[:sample_count // 2]

    # Filtering the frequencies to be within the range 10^-3 to 10^-1
    filter_mask = (positive_frequencies >= 1e-3) & (positive_frequencies <= 1e-1)
    filtered_frequencies = positive_frequencies[filter_mask]
    filtered_magnitudes = positive_magnitudes[filter_mask]

    return filtered_frequencies, filtered_magnitudes
