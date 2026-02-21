"""
Key Detection Module

Detects the global key of a track using the Krumhansl-Schmuckler key-finding
algorithm. This is the "Theory Filter" that enables context-aware chord
correction:

  - If the detected key is E major and the recognizer suggests Bb minor,
    the filter can flag it for manual review or suggest A# diminished instead.

Uses key profiles (correlation templates for major and minor keys) derived
from cognitive music perception research.
"""

import numpy as np
from typing import Tuple, Optional, List

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

ENHARMONIC_MAP = {
    'C#': 'Db',
    'D#': 'Eb',
    'F#': 'Gb',
    'G#': 'Ab',
    'A#': 'Bb',
}

# Krumhansl-Kessler key profiles (major and minor)
# These represent the "expected" distribution of pitch classes in a given key.
# Higher values = more likely to appear in that key.
MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])

# Diatonic scale degrees for each key mode
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
HARMONIC_MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 11]

# Common chord types for each scale degree in major keys
# I=maj, ii=min, iii=min, IV=maj, V=maj, vi=min, vii=dim
MAJOR_KEY_CHORD_QUALITIES = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']

# Common chord types for natural minor: i=min, ii=dim, III=maj, iv=min, v=min, VI=maj, VII=maj
MINOR_KEY_CHORD_QUALITIES = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']


class KeyDetector:
    """Detects the musical key of a piece from its chromagram.

    Uses the Krumhansl-Schmuckler algorithm: correlate the average chroma
    distribution with major/minor key profiles for all 12 possible roots.

    Parameters
    ----------
    use_enharmonic : bool
        Display flats instead of sharps where conventional.
    """

    def __init__(self, use_enharmonic=True):
        self.use_enharmonic = use_enharmonic
        self._build_all_profiles()

    def _build_all_profiles(self):
        """Pre-compute key profiles for all 24 keys (12 major + 12 minor)."""
        self.key_profiles = {}
        for root_idx in range(12):
            root_name = NOTE_NAMES[root_idx]

            # Major key
            major_shifted = np.roll(MAJOR_PROFILE, root_idx)
            self.key_profiles[f'{root_name} major'] = major_shifted

            # Minor key
            minor_shifted = np.roll(MINOR_PROFILE, root_idx)
            self.key_profiles[f'{root_name} minor'] = minor_shifted

    def detect_key(self, chroma):
        """Detect the most likely key from a chromagram.

        Parameters
        ----------
        chroma : np.ndarray, shape (12, T)
            Chromagram matrix.

        Returns
        -------
        key : str
            Detected key (e.g., "G major", "E minor").
        confidence : float
            Pearson correlation coefficient (higher = more confident).
        all_scores : list of tuple
            All (key_name, score) pairs sorted by score descending.
        """
        # Average chroma across all frames
        avg_chroma = chroma.mean(axis=1)

        scores = []
        for key_name, profile in self.key_profiles.items():
            corr = np.corrcoef(avg_chroma, profile)[0, 1]
            scores.append((key_name, float(corr)))

        scores.sort(key=lambda x: x[1], reverse=True)

        best_key = scores[0][0]
        best_score = scores[0][1]

        if self.use_enharmonic:
            best_key = self._apply_enharmonic_key(best_key)
            scores = [(self._apply_enharmonic_key(k), s) for k, s in scores]

        return best_key, best_score, scores

    def get_diatonic_chords(self, key_name):
        """Return the set of chords that naturally belong in a given key.

        Parameters
        ----------
        key_name : str
            Key name (e.g., "C major", "A minor").

        Returns
        -------
        chords : list of str
            Chord names that belong in this key (e.g., ["C", "Dm", "Em", ...]).
        """
        parts = key_name.split()
        root_name = parts[0]
        mode = parts[1] if len(parts) > 1 else 'major'

        # Reverse enharmonic for lookup
        root_idx = self._note_to_index(root_name)
        if root_idx is None:
            return []

        if mode == 'major':
            intervals = MAJOR_SCALE_INTERVALS
            qualities = MAJOR_KEY_CHORD_QUALITIES
        else:
            intervals = NATURAL_MINOR_SCALE_INTERVALS
            qualities = MINOR_KEY_CHORD_QUALITIES

        chords = []
        for interval, quality in zip(intervals, qualities):
            chord_root_idx = (root_idx + interval) % 12
            chord_root = NOTE_NAMES[chord_root_idx]
            if self.use_enharmonic:
                chord_root = ENHARMONIC_MAP.get(chord_root, chord_root)

            if quality == 'maj':
                chords.append(chord_root)
            elif quality == 'min':
                chords.append(f'{chord_root}m')
            elif quality == 'dim':
                chords.append(f'{chord_root}dim')

        return chords

    def get_extended_diatonic_chords(self, key_name):
        """Return an extended set of chords likely in a key.

        Includes diatonic triads plus common extensions:
        - Dominant 7ths (V7)
        - Secondary dominants
        - Minor 7ths on ii, iii, vi
        - Major 7ths on I, IV

        Parameters
        ----------
        key_name : str

        Returns
        -------
        chords : set of str
        """
        basic = set(self.get_diatonic_chords(key_name))

        parts = key_name.split()
        root_name = parts[0]
        mode = parts[1] if len(parts) > 1 else 'major'
        root_idx = self._note_to_index(root_name)

        if root_idx is None:
            return basic

        extended = set(basic)

        if mode == 'major':
            # Add dominant 7th on V
            v_root = NOTE_NAMES[(root_idx + 7) % 12]
            if self.use_enharmonic:
                v_root = ENHARMONIC_MAP.get(v_root, v_root)
            extended.add(f'{v_root}7')

            # Add major 7th on I and IV
            i_root = NOTE_NAMES[root_idx]
            iv_root = NOTE_NAMES[(root_idx + 5) % 12]
            if self.use_enharmonic:
                i_root = ENHARMONIC_MAP.get(i_root, i_root)
                iv_root = ENHARMONIC_MAP.get(iv_root, iv_root)
            extended.add(f'{i_root}maj7')
            extended.add(f'{iv_root}maj7')

            # Add minor 7ths on ii, iii, vi
            for interval in [2, 4, 9]:
                n = NOTE_NAMES[(root_idx + interval) % 12]
                if self.use_enharmonic:
                    n = ENHARMONIC_MAP.get(n, n)
                extended.add(f'{n}min7')

        elif mode == 'minor':
            # Add dominant 7th on V (harmonic minor)
            v_root = NOTE_NAMES[(root_idx + 7) % 12]
            if self.use_enharmonic:
                v_root = ENHARMONIC_MAP.get(v_root, v_root)
            extended.add(f'{v_root}7')

            # Add minor 7th on i
            i_root = NOTE_NAMES[root_idx]
            if self.use_enharmonic:
                i_root = ENHARMONIC_MAP.get(i_root, i_root)
            extended.add(f'{i_root}min7')

        return extended

    def flag_out_of_key_chords(self, chords, key_name, strict=False):
        """Flag chords that don't belong in the detected key.

        Parameters
        ----------
        chords : list of dict
            Chord sequence from ChordRecognizer.
        key_name : str
            Detected key.
        strict : bool
            If True, only allow basic diatonic chords.
            If False, allow extended set (7ths, secondary dominants).

        Returns
        -------
        flagged : list of dict
            Same as input, with added 'in_key' (bool) and 'suggestion' (str or None).
        """
        if strict:
            allowed = set(self.get_diatonic_chords(key_name))
        else:
            allowed = self.get_extended_diatonic_chords(key_name)

        flagged = []
        for entry in chords:
            chord = entry['chord']
            in_key = chord in allowed or chord == 'N'

            suggestion = None
            if not in_key and chord != 'N':
                suggestion = self._suggest_correction(chord, allowed)

            flagged.append({
                **entry,
                'in_key': in_key,
                'suggestion': suggestion,
            })

        return flagged

    def _suggest_correction(self, chord, allowed_chords):
        """Suggest the most likely in-key chord as a correction.

        Uses a simple heuristic: find the allowed chord whose root is closest
        (in semitones) to the detected chord's root.
        """
        chord_root_idx = self._chord_root_index(chord)
        if chord_root_idx is None:
            return None

        best_match = None
        best_distance = 12

        for candidate in allowed_chords:
            cand_root_idx = self._chord_root_index(candidate)
            if cand_root_idx is None:
                continue
            # Circular distance
            dist = min(
                abs(chord_root_idx - cand_root_idx),
                12 - abs(chord_root_idx - cand_root_idx)
            )
            if dist < best_distance:
                best_distance = dist
                best_match = candidate

        return best_match

    def _chord_root_index(self, chord_name):
        """Extract the root note index from a chord name."""
        if not chord_name or chord_name == 'N':
            return None

        # Try two-character root first (e.g., "C#", "Bb")
        if len(chord_name) >= 2:
            two_char = chord_name[:2]
            idx = self._note_to_index(two_char)
            if idx is not None:
                return idx

        # Single character root
        return self._note_to_index(chord_name[0])

    def _note_to_index(self, note_name):
        """Convert a note name to its chromatic index (0-11)."""
        if note_name in NOTE_NAMES:
            return NOTE_NAMES.index(note_name)
        # Check enharmonic
        for sharp, flat in ENHARMONIC_MAP.items():
            if note_name == flat:
                return NOTE_NAMES.index(sharp)
        return None

    def _apply_enharmonic_key(self, key_name):
        """Apply enharmonic substitution to a key name."""
        parts = key_name.split()
        root = parts[0]
        mode = parts[1] if len(parts) > 1 else ''
        root = ENHARMONIC_MAP.get(root, root)
        return f'{root} {mode}'.strip()
