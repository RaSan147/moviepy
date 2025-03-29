"""Cutting utilities working with audio."""

from moviepy.np_handler import np, np_get

def find_audio_period(clip, min_time=0.1, max_time=2, time_resolution=0.01):
    """Finds the period, in seconds of an audioclip.

    Parameters
    ----------

    min_time : float, optional
      Minimum bound for the returned value.

    max_time : float, optional
      Maximum bound for the returned value.

    time_resolution : float, optional
      Numerical precision.
    """
    chunksize = int(time_resolution * clip.fps)
    chunk_duration = 1.0 * chunksize / clip.fps
    # v denotes the list of volumes
    v = np.array([(chunk**2).sum() for chunk in clip.iter_chunks(chunksize)])
    v = v - v.mean()
    corrs = np.correlate(v, v, mode="full")[-len(v) :]
    corrs[: int(min_time / chunk_duration)] = 0
    corrs[int(max_time / chunk_duration) :] = 0
    return np_get(chunk_duration * np.argmax(corrs))


def find_audio_period(clip, min_time=0.1, max_time=2, time_resolution=0.01):
    """Finds the period, in seconds of an audioclip.

    Parameters
    ----------

    min_time : float, optional
		Minimum bound for the returned value.

    max_time : float, optional
		Maximum bound for the returned value.

    time_resolution : float, optional
		Numerical precision.
    """
    # Calculate chunk parameters
    chunksize = int(time_resolution * clip.fps)
    chunk_duration = chunksize / clip.fps  # Same as time_resolution

    # Process audio chunks on GPU
    volumes = np.array([
        (np.array(chunk, dtype=np.float32) ** 2).sum()
        for chunk in clip.iter_chunks(chunksize)
    ], dtype=np.float32)

    # Handle edge case with insufficient data
    if volumes.size < 2:
        return 0.0

    # Normalize volumes
    volumes -= volumes.mean()

    # Compute FFT-based autocorrelation
    n = volumes.size
    fft_result = np.fft.rfft(volumes, n=2 * n)  # Use real FFT
    power_spectrum = fft_result * fft_result.conj()
    autocorr = np.fft.irfft(power_spectrum)[:n]  # Compute autocorrelation
    
    # Apply time constraints
    min_idx = int(min_time / chunk_duration)
    max_idx = min(int(max_time / chunk_duration), n - 1)

    autocorr[:min_idx] = 0  # Ignore too-short periods
    autocorr[max_idx:] = 0  # Ignore too-long periods

    # Find dominant period
    best_idx = np.argmax(autocorr)
    return float(np_get(best_idx * chunk_duration))