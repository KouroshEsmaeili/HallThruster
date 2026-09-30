import numpy as np


def current_spectrum(sol):
    # Number of samples
    N = len(sol.t)

    # Compute sample rate (assume sol.t is in seconds)
    # Use the difference between the first two samples (0-indexed).
    dt = sol.t[1] - sol.t[0]
    sample_rate = 1.0 / dt

    # Extract the discharge current from sol["Id"].
    # Here we assume that sol["Id"] is a list of values or one-element containers.
    # If each element is a list, extract the first element.
    current = []
    for i in range(N):
        value = sol["Id"][i]
        # If value is a list or array, extract its first element.
        if isinstance(value, (list, np.ndarray)):
            current.append(value[0])
        else:
            current.append(value)
    current = np.array(current,dtype=np.float64)

    # Compute Fourier transform of current, shift, and take absolute value.
    amplitude = np.abs(np.fft.fftshift(np.fft.fft(current)))

    # Compute frequency bins and apply fftshift.
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dt))

    return freqs, amplitude
