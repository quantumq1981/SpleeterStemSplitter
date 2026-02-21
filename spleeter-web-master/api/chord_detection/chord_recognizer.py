"""
Phase 3: Chord Recognition — The "Musician Brain"

Takes 12-bin chroma vectors and matches them against a dictionary of chord
templates. When the chromagram shows high energy at A, C, E, and G, this
module identifies it as Am7.

The chord dictionary covers:
  - Major, minor, diminished, augmented triads
  - 7th chords (major 7, minor 7, dominant 7, half-dim, dim7)
  - Suspended chords (sus2, sus4)
  - Add9, 6th chords
  - Power chords (5)

Each template is a 12-element binary vector representing which pitch classes
are present in the chord.
"""

import numpy as np
from typing import List, Tuple, Optional

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enharmonic display preferences (sharps -> flats for readability in common keys)
ENHARMONIC_MAP = {
    'C#': 'Db',
    'D#': 'Eb',
    'F#': 'Gb',
    'G#': 'Ab',
    'A#': 'Bb',
}


def _intervals_to_template(intervals):
    """Convert a list of semitone intervals from root to a 12-bin template.

    Example: major triad = [0, 4, 7] -> [1,0,0,0,1,0,0,1,0,0,0,0]
    """
    template = np.zeros(12)
    for i in intervals:
        template[i % 12] = 1.0
    return template


# Chord quality definitions as semitone intervals from root
CHORD_QUALITIES = {
    'maj':    [0, 4, 7],
    'min':    [0, 3, 7],
    'dim':    [0, 3, 6],
    'aug':    [0, 4, 8],
    '7':      [0, 4, 7, 10],       # dominant 7
    'maj7':   [0, 4, 7, 11],
    'min7':   [0, 3, 7, 10],
    'dim7':   [0, 3, 6, 9],
    'm7b5':   [0, 3, 6, 10],       # half-diminished
    'sus2':   [0, 2, 7],
    'sus4':   [0, 5, 7],
    'add9':   [0, 4, 7, 14],       # same as [0,2,4,7] after mod 12
    '6':      [0, 4, 7, 9],
    'min6':   [0, 3, 7, 9],
    '9':      [0, 4, 7, 10, 14],   # dominant 9
    '5':      [0, 7],              # power chord
    'minmaj7': [0, 3, 7, 11],
    'aug7':   [0, 4, 8, 10],
    '7sus4':  [0, 5, 7, 10],
}


def _build_chord_templates():
    """Build the full chord template dictionary for all roots and qualities.

    Returns
    -------
    templates : dict
        {chord_name: np.ndarray shape (12,)} for every root/quality combination.
    """
    templates = {}
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for quality, intervals in CHORD_QUALITIES.items():
            # Shift intervals by root
            shifted = [(i + root_idx) % 12 for i in intervals]
            template = np.zeros(12)
            for s in shifted:
                template[s] = 1.0

            # Format chord name
            if quality == 'maj':
                chord_name = root_name
            elif quality == 'min':
                chord_name = f'{root_name}m'
            else:
                chord_name = f'{root_name}{quality}'

            templates[chord_name] = template

    # Add "N" (no chord / silence)
    templates['N'] = np.zeros(12)

    return templates


CHORD_TEMPLATES = _build_chord_templates()


class ChordRecognizer:
    """Matches chroma vectors against chord templates to identify chords.

    Uses cosine similarity (or optionally weighted correlation) to find the
    best-matching chord quality for each time frame.

    Parameters
    ----------
    min_energy : float
        Minimum total chroma energy to attempt chord recognition.
        Below this threshold, the frame is labeled "N" (no chord / silence).
    use_enharmonic : bool
        If True, display flats instead of sharps where conventional
        (e.g., Bb instead of A#).
    """

    def __init__(self, min_energy=0.1, use_enharmonic=True):
        self.templates = CHORD_TEMPLATES
        self.template_names = list(CHORD_TEMPLATES.keys())
        self.template_matrix = np.array(
            [CHORD_TEMPLATES[name] for name in self.template_names]
        )  # shape: (N_chords, 12)
        self.min_energy = min_energy
        self.use_enharmonic = use_enharmonic

    def _cosine_similarity(self, chroma_frame, templates):
        """Compute cosine similarity between a chroma frame and all templates."""
        # Normalize the chroma frame
        frame_norm = np.linalg.norm(chroma_frame)
        if frame_norm < 1e-10:
            return np.zeros(len(templates))

        normed_frame = chroma_frame / frame_norm

        # Normalize templates
        template_norms = np.linalg.norm(templates, axis=1, keepdims=True)
        template_norms = np.maximum(template_norms, 1e-10)
        normed_templates = templates / template_norms

        return normed_templates @ normed_frame

    def recognize_frame(self, chroma_frame):
        """Identify the chord for a single 12-bin chroma vector.

        Parameters
        ----------
        chroma_frame : np.ndarray, shape (12,)
            Chroma energy for one time frame.

        Returns
        -------
        chord_name : str
            Identified chord name (e.g., "Am7", "G", "N").
        confidence : float
            Cosine similarity score (0-1).
        """
        total_energy = np.sum(chroma_frame)
        if total_energy < self.min_energy:
            return 'N', 0.0

        similarities = self._cosine_similarity(chroma_frame, self.template_matrix)

        # Exclude "N" template from best match (it would match silence)
        n_idx = self.template_names.index('N')
        similarities[n_idx] = -1.0

        best_idx = np.argmax(similarities)
        chord_name = self.template_names[best_idx]
        confidence = float(similarities[best_idx])

        if self.use_enharmonic:
            chord_name = self._apply_enharmonic(chord_name)

        return chord_name, confidence

    def recognize_sequence(self, chroma, times):
        """Identify chords for an entire chromagram.

        Parameters
        ----------
        chroma : np.ndarray, shape (12, T)
            Chromagram matrix.
        times : np.ndarray, shape (T,)
            Frame timestamps.

        Returns
        -------
        chords : list of dict
            Each entry: {
                'time': float (seconds),
                'chord': str,
                'confidence': float
            }
        """
        chords = []
        for i in range(chroma.shape[1]):
            chord_name, confidence = self.recognize_frame(chroma[:, i])
            chords.append({
                'time': float(times[i]),
                'chord': chord_name,
                'confidence': confidence,
            })
        return chords

    def simplify_sequence(self, chords, min_duration=0.3):
        """Collapse consecutive identical chords and filter very short ones.

        Parameters
        ----------
        chords : list of dict
            Output of recognize_sequence().
        min_duration : float
            Minimum chord duration in seconds. Chords shorter than this
            are merged with neighbors.

        Returns
        -------
        simplified : list of dict
            Each entry: {
                'time': float,
                'end_time': float,
                'duration': float,
                'chord': str,
                'confidence': float (average over merged frames)
            }
        """
        if not chords:
            return []

        simplified = []
        current = {
            'time': chords[0]['time'],
            'chord': chords[0]['chord'],
            'confidences': [chords[0]['confidence']],
        }

        for entry in chords[1:]:
            if entry['chord'] == current['chord']:
                current['confidences'].append(entry['confidence'])
            else:
                # Close current segment
                avg_conf = np.mean(current['confidences'])
                end_time = entry['time']
                duration = end_time - current['time']

                simplified.append({
                    'time': current['time'],
                    'end_time': end_time,
                    'duration': duration,
                    'chord': current['chord'],
                    'confidence': float(avg_conf),
                })

                current = {
                    'time': entry['time'],
                    'chord': entry['chord'],
                    'confidences': [entry['confidence']],
                }

        # Close final segment
        if current['confidences']:
            end_time = chords[-1]['time'] + 0.1  # approximate
            simplified.append({
                'time': current['time'],
                'end_time': end_time,
                'duration': end_time - current['time'],
                'chord': current['chord'],
                'confidence': float(np.mean(current['confidences'])),
            })

        # Filter by minimum duration
        if min_duration > 0:
            simplified = self._filter_short_chords(simplified, min_duration)

        return simplified

    def _filter_short_chords(self, segments, min_duration):
        """Merge segments shorter than min_duration into their neighbors."""
        if len(segments) <= 1:
            return segments

        filtered = []
        for seg in segments:
            if seg['duration'] >= min_duration or not filtered:
                filtered.append(seg)
            else:
                # Merge into previous segment
                filtered[-1]['end_time'] = seg['end_time']
                filtered[-1]['duration'] = (
                    filtered[-1]['end_time'] - filtered[-1]['time']
                )

        return filtered

    def _apply_enharmonic(self, chord_name):
        """Convert sharp note names to flats where conventional."""
        for sharp, flat in ENHARMONIC_MAP.items():
            if chord_name.startswith(sharp):
                return flat + chord_name[len(sharp):]
        return chord_name

    def get_chord_notes(self, chord_name):
        """Return the individual note names that make up a chord.

        Parameters
        ----------
        chord_name : str
            Chord name (e.g., "Am7", "G").

        Returns
        -------
        notes : list[str] or None
            List of note names, or None if chord not in dictionary.
        """
        # Reverse enharmonic for lookup
        lookup_name = chord_name
        for sharp, flat in ENHARMONIC_MAP.items():
            if chord_name.startswith(flat):
                lookup_name = sharp + chord_name[len(flat):]
                break

        template = self.templates.get(lookup_name)
        if template is None:
            return None

        return [NOTE_NAMES[i] for i in range(12) if template[i] > 0]
