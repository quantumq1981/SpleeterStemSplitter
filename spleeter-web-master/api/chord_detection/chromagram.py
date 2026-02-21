"""
Phase 2: Chromagram Extraction via Constant-Q Transform (CQT)

Converts audio waveforms into chromagrams — 12-bin representations mapping
energy to each semitone of the Western chromatic scale. Unlike standard FFT,
CQT aligns frequency bins logarithmically to musical pitches, making it ideal
for harmonic analysis.
"""

import numpy as np

try:
    import librosa
except ImportError:
    raise ImportError(
        "librosa is required for chord detection. "
        "Install it with: pip install librosa"
    )


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class ChromagramExtractor:
    """Extracts chromagram features from audio using Constant-Q Transform.

    The CQT maps frequencies to the 12-semitone Western scale, producing a
    chromagram: a time-vs-chroma matrix showing which of the 12 pitch classes
    are active at each time frame.

    Parameters
    ----------
    sr : int
        Sample rate. Default 22050 (librosa standard).
    hop_length : int
        Number of samples between successive frames. Controls time resolution.
        Default 2048 (~93ms at 22050 Hz) balances resolution vs. stability.
    n_chroma : int
        Number of chroma bins (12 for standard Western music).
    norm : float or None
        Column-wise normalization. np.inf for max-normalization (default).
    """

    def __init__(self, sr=22050, hop_length=2048, n_chroma=12, norm=np.inf):
        self.sr = sr
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.norm = norm

    def load_audio(self, audio_path, sr=None):
        """Load an audio file and return the waveform.

        Parameters
        ----------
        audio_path : str
            Path to audio file (WAV, MP3, FLAC, etc.)
        sr : int or None
            Target sample rate. None uses self.sr.

        Returns
        -------
        y : np.ndarray
            Audio time series (mono).
        sr : int
            Sampling rate.
        """
        target_sr = sr or self.sr
        y, sr_out = librosa.load(audio_path, sr=target_sr, mono=True)
        return y, sr_out

    def extract_chromagram(self, y, sr=None):
        """Extract CQT-based chromagram from audio waveform.

        Parameters
        ----------
        y : np.ndarray
            Audio time series.
        sr : int or None
            Sample rate of y. Uses self.sr if None.

        Returns
        -------
        chroma : np.ndarray, shape (12, T)
            Chromagram matrix. Each column is a 12-bin chroma vector
            for one time frame.
        times : np.ndarray, shape (T,)
            Timestamps in seconds for each frame.
        """
        sr = sr or self.sr
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
            norm=self.norm,
        )
        times = librosa.times_like(chroma, sr=sr, hop_length=self.hop_length)
        return chroma, times

    def extract_chromagram_enhanced(self, y, sr=None):
        """Extract enhanced chromagram using CENS (Chroma Energy Normalized Statistics).

        CENS applies additional smoothing and normalization, making it more
        robust for chord recognition in noisy or reverb-heavy audio.

        Parameters
        ----------
        y : np.ndarray
            Audio time series.
        sr : int or None
            Sample rate.

        Returns
        -------
        chroma : np.ndarray, shape (12, T)
            CENS chromagram.
        times : np.ndarray, shape (T,)
            Frame timestamps in seconds.
        """
        sr = sr or self.sr
        chroma = librosa.feature.chroma_cens(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )
        times = librosa.times_like(chroma, sr=sr, hop_length=self.hop_length)
        return chroma, times

    def extract_from_file(self, audio_path, enhanced=False):
        """Convenience: load file and extract chromagram in one call.

        Parameters
        ----------
        audio_path : str
            Path to audio file.
        enhanced : bool
            If True, use CENS instead of raw CQT chromagram.

        Returns
        -------
        chroma : np.ndarray, shape (12, T)
        times : np.ndarray, shape (T,)
        """
        y, sr = self.load_audio(audio_path)
        if enhanced:
            return self.extract_chromagram_enhanced(y, sr)
        return self.extract_chromagram(y, sr)

    def get_dominant_notes(self, chroma, times, threshold=0.4):
        """Identify the dominant notes (pitch classes) at each time frame.

        Parameters
        ----------
        chroma : np.ndarray, shape (12, T)
            Chromagram.
        times : np.ndarray, shape (T,)
            Frame timestamps.
        threshold : float
            Minimum energy (0-1) for a note to be considered "active".

        Returns
        -------
        list of dict
            Each entry: {'time': float, 'notes': list[str], 'energies': list[float]}
        """
        results = []
        for i, t in enumerate(times):
            frame = chroma[:, i]
            active_indices = np.where(frame >= threshold)[0]
            notes = [NOTE_NAMES[idx] for idx in active_indices]
            energies = [float(frame[idx]) for idx in active_indices]
            results.append({
                'time': float(t),
                'notes': notes,
                'energies': energies,
            })
        return results

    def aggregate_chroma_segments(self, chroma, times, segment_duration=0.5):
        """Aggregate chromagram into fixed-length segments by averaging.

        Useful for reducing noise and producing one chord label per beat/bar
        rather than per frame.

        Parameters
        ----------
        chroma : np.ndarray, shape (12, T)
        times : np.ndarray, shape (T,)
        segment_duration : float
            Duration of each segment in seconds.

        Returns
        -------
        seg_chroma : np.ndarray, shape (12, N_segments)
        seg_times : np.ndarray, shape (N_segments,)
        """
        if len(times) == 0:
            return chroma, times

        total_duration = times[-1]
        n_segments = max(1, int(np.ceil(total_duration / segment_duration)))
        seg_chroma = np.zeros((self.n_chroma, n_segments))
        seg_times = np.zeros(n_segments)

        for seg_idx in range(n_segments):
            t_start = seg_idx * segment_duration
            t_end = (seg_idx + 1) * segment_duration
            mask = (times >= t_start) & (times < t_end)
            if np.any(mask):
                seg_chroma[:, seg_idx] = chroma[:, mask].mean(axis=1)
            seg_times[seg_idx] = t_start

        return seg_chroma, seg_times
