import numpy as np
from scipy import signal as sig


# --- Standard EEG frequency bands ---
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


class filters:
    """Signal filtering operations. All modify data in-place."""

    @staticmethod
    def bandpass(data, sr, lo, hi, order=4, zero_phase=True):
        """Apply bandpass IIR filter. Returns filtered data (also modifies in-place)."""
        sos = sig.iirfilter(order, [lo, hi], btype='bandpass', fs=sr, output='sos')
        if zero_phase:
            result = sig.sosfiltfilt(sos, data, axis=-1)
        else:
            result = sig.sosfilt(sos, data, axis=-1)
        if isinstance(data, np.ndarray):
            data[:] = result
        return result

    @staticmethod
    def highpass(data, sr, cutoff, order=4, zero_phase=True):
        """Apply highpass IIR filter."""
        sos = sig.iirfilter(order, cutoff, btype='highpass', fs=sr, output='sos')
        if zero_phase:
            result = sig.sosfiltfilt(sos, data, axis=-1)
        else:
            result = sig.sosfilt(sos, data, axis=-1)
        if isinstance(data, np.ndarray):
            data[:] = result
        return result

    @staticmethod
    def lowpass(data, sr, cutoff, order=4, zero_phase=True):
        """Apply lowpass IIR filter."""
        sos = sig.iirfilter(order, cutoff, btype='lowpass', fs=sr, output='sos')
        if zero_phase:
            result = sig.sosfiltfilt(sos, data, axis=-1)
        else:
            result = sig.sosfilt(sos, data, axis=-1)
        if isinstance(data, np.ndarray):
            data[:] = result
        return result

    @staticmethod
    def bandstop(data, sr, lo, hi, order=4, zero_phase=True):
        """Apply bandstop IIR filter."""
        sos = sig.iirfilter(order, [lo, hi], btype='bandstop', fs=sr, output='sos')
        if zero_phase:
            result = sig.sosfiltfilt(sos, data, axis=-1)
        else:
            result = sig.sosfilt(sos, data, axis=-1)
        if isinstance(data, np.ndarray):
            data[:] = result
        return result

    @staticmethod
    def notch(data, sr, freq=50.0, Q=30.0):
        """Remove power line noise at given frequency (50 or 60 Hz)."""
        b, a = sig.iirnotch(freq, Q, fs=sr)
        result = sig.filtfilt(b, a, data, axis=-1)
        if isinstance(data, np.ndarray):
            data[:] = result
        return result

    @staticmethod
    def notch_50_60(data, sr, Q=30.0):
        """Remove both 50Hz and 60Hz power line noise."""
        result = filters.notch(data, sr, 50.0, Q)
        result = filters.notch(result, sr, 60.0, Q)
        return result

    @staticmethod
    def detrend(data, kind='linear'):
        """Remove DC offset (kind='constant') or linear trend (kind='linear')."""
        result = sig.detrend(data, axis=-1, type=kind)
        if isinstance(data, np.ndarray):
            data[:] = result
        return result


class spectral:
    """Spectral analysis and band power computation."""

    @staticmethod
    def psd(data, sr, nperseg=None, window='hann'):
        """Compute power spectral density using Welch's method.

        Returns (freqs, powers) arrays.
        """
        if nperseg is None:
            nperseg = min(len(data) if data.ndim == 1 else data.shape[-1], sr * 2)
        freqs, powers = sig.welch(data, fs=sr, nperseg=nperseg, window=window, axis=-1)
        return freqs, powers

    @staticmethod
    def band_power(freqs, powers, lo, hi):
        """Compute average power in a frequency band via trapezoidal integration."""
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        if powers.ndim == 1:
            return float(np.trapz(powers[mask], freqs[mask]))
        return np.trapz(powers[:, mask], freqs[mask], axis=-1)

    @staticmethod
    def band_powers(data, sr, bands=None, nperseg=None):
        """Compute power in standard EEG bands.

        Args:
            data: 1D or 2D array (channels x samples)
            sr: sampling rate
            bands: dict of {name: (lo, hi)}, defaults to standard EEG bands

        Returns:
            dict of {band_name: power} (or array of powers if multi-channel)
        """
        if bands is None:
            bands = EEG_BANDS
        freqs, powers = spectral.psd(data, sr, nperseg)
        result = {}
        for name, (lo, hi) in bands.items():
            result[name] = spectral.band_power(freqs, powers, lo, hi)
        return result

    @staticmethod
    def fft(data, window='hann'):
        """Compute FFT of data, optionally windowed."""
        n = data.shape[-1] if data.ndim > 1 else len(data)
        if window:
            w = sig.get_window(window, n)
            data = data * w
        return np.fft.rfft(data, axis=-1)

    @staticmethod
    def ifft(spectrum, n=None):
        """Inverse FFT."""
        return np.fft.irfft(spectrum, n=n, axis=-1)
