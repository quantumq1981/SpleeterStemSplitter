"""
Transition Matrix — Context-Aware Chord Correction

Encodes the statistical likelihood of chord-to-chord transitions based on
music theory and common harmonic practice. This is the "prompt engineering"
for our chord AI:

  "If you just heard a G7, you are 80% more likely to hear a C than a C#."

The matrix uses two layers:
  1. A theory-based prior (circle of fifths, common progressions)
  2. Viterbi decoding to find the globally optimal chord sequence given
     both the chromagram evidence and transition probabilities.
"""

import numpy as np
from typing import List, Dict, Optional

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _semitone_distance(a, b):
    """Circular semitone distance between two pitch classes."""
    return min(abs(a - b), 12 - abs(a - b))


class TransitionMatrix:
    """Chord transition probability model for sequence smoothing.

    Combines music-theory priors (circle of fifths, common progressions)
    with observation likelihoods (from template matching) to produce
    smoothed chord sequences via Viterbi decoding.

    Parameters
    ----------
    key : str or None
        If provided, biases transitions toward in-key chords.
        Format: "C major" or "A minor".
    smoothing : float
        Weight for transition probabilities vs. raw observations.
        Higher = smoother output, more theory influence. Range 0-1.
    """

    def __init__(self, key=None, smoothing=0.6):
        self.key = key
        self.smoothing = smoothing

    def build_transition_probs(self, chord_vocab):
        """Build a transition probability matrix for a chord vocabulary.

        Assigns higher probability to transitions that are musically common:
        - V -> I (dominant resolution): highest weight
        - IV -> V (pre-dominant to dominant): high weight
        - I -> IV, I -> V (tonic to subdominant/dominant): high weight
        - Circle-of-fifths motion: moderate weight
        - Same chord repeated: high weight (chords tend to sustain)
        - Chromatic/distant motion: low weight

        Parameters
        ----------
        chord_vocab : list of str
            List of chord names in the vocabulary.

        Returns
        -------
        trans_matrix : np.ndarray, shape (N, N)
            Row-normalized transition probability matrix.
            trans_matrix[i, j] = P(chord_j | chord_i).
        """
        n = len(chord_vocab)
        trans = np.ones((n, n)) * 0.01  # small uniform baseline

        for i, from_chord in enumerate(chord_vocab):
            from_root = self._chord_root_index(from_chord)

            for j, to_chord in enumerate(chord_vocab):
                to_root = self._chord_root_index(to_chord)

                # Same chord stays (chords sustain)
                if i == j:
                    trans[i, j] = 0.40
                    continue

                # N (silence) transitions
                if from_chord == 'N' or to_chord == 'N':
                    trans[i, j] = 0.05
                    continue

                if from_root is None or to_root is None:
                    continue

                interval = (to_root - from_root) % 12

                # Perfect 5th down (V -> I resolution) - strongest
                if interval == 7:  # up a 5th = down a 4th
                    trans[i, j] = 0.25
                # Perfect 4th down (IV -> I, also common)
                elif interval == 5:
                    trans[i, j] = 0.20
                # Step up (I -> ii, etc.)
                elif interval == 2:
                    trans[i, j] = 0.15
                # Step down
                elif interval == 10:
                    trans[i, j] = 0.12
                # Minor 3rd up (I -> iii, relative minor)
                elif interval == 3:
                    trans[i, j] = 0.10
                # Minor 3rd down (vi -> IV)
                elif interval == 9:
                    trans[i, j] = 0.10
                # Tritone (distant but used in jazz/blues)
                elif interval == 6:
                    trans[i, j] = 0.04
                # Semitone motion (chromatic)
                elif interval in (1, 11):
                    trans[i, j] = 0.06

                # Bonus: dominant 7th resolving down a 5th
                if '7' in from_chord and interval == 7:
                    trans[i, j] = 0.30

        # Row-normalize
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        trans = trans / row_sums

        return trans

    def viterbi_decode(self, observations, chord_vocab, trans_matrix=None):
        """Run Viterbi decoding to find the optimal chord sequence.

        Combines observation likelihoods (template matching scores) with
        transition probabilities to find the most likely sequence.

        Parameters
        ----------
        observations : list of dict
            Each entry: {'chord': str, 'confidence': float, 'time': float}
            from ChordRecognizer.recognize_sequence().
        chord_vocab : list of str
            Ordered chord vocabulary.
        trans_matrix : np.ndarray or None
            Transition probability matrix. Built automatically if None.

        Returns
        -------
        smoothed : list of dict
            Same format as input but with potentially corrected chord labels.
        """
        if not observations:
            return []

        n_states = len(chord_vocab)
        n_obs = len(observations)
        chord_to_idx = {c: i for i, c in enumerate(chord_vocab)}

        if trans_matrix is None:
            trans_matrix = self.build_transition_probs(chord_vocab)

        # Build observation probability matrix
        obs_probs = np.zeros((n_obs, n_states))
        for t, obs in enumerate(observations):
            detected = obs['chord']
            conf = obs['confidence']

            if detected in chord_to_idx:
                detected_idx = chord_to_idx[detected]
                # Detected chord gets the confidence score
                obs_probs[t, detected_idx] = conf
            else:
                # Unknown chord - use uniform
                obs_probs[t, :] = 1.0 / n_states
                continue

            # Spread remaining probability mass
            remaining = max(0, 1.0 - conf)
            for s in range(n_states):
                if s != chord_to_idx.get(detected):
                    obs_probs[t, s] = remaining / max(1, n_states - 1)

        # Ensure no zeros (log-safe)
        obs_probs = np.maximum(obs_probs, 1e-10)

        # Log-domain Viterbi
        log_trans = np.log(np.maximum(trans_matrix, 1e-10))
        log_obs = np.log(obs_probs)

        # Initialize
        V = np.zeros((n_obs, n_states))
        backpointer = np.zeros((n_obs, n_states), dtype=int)

        # Blend: (1-smoothing)*observation + smoothing*transition
        alpha = self.smoothing

        # Initial state: just observation probabilities (uniform prior)
        V[0, :] = log_obs[0, :]

        for t in range(1, n_obs):
            for s in range(n_states):
                # For each possible current state s, find best previous state
                scores = V[t - 1, :] + alpha * log_trans[:, s]
                best_prev = np.argmax(scores)
                V[t, s] = scores[best_prev] + (1 - alpha) * log_obs[t, s]
                backpointer[t, s] = best_prev

        # Backtrace
        best_path = np.zeros(n_obs, dtype=int)
        best_path[-1] = np.argmax(V[-1, :])

        for t in range(n_obs - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        # Build output
        smoothed = []
        for t in range(n_obs):
            smoothed.append({
                'time': observations[t]['time'],
                'chord': chord_vocab[best_path[t]],
                'confidence': observations[t]['confidence'],
                'original_chord': observations[t]['chord'],
                'was_corrected': observations[t]['chord'] != chord_vocab[best_path[t]],
            })

        return smoothed

    def smooth_sequence(self, chord_sequence):
        """Convenience method: apply Viterbi smoothing to a chord sequence.

        Parameters
        ----------
        chord_sequence : list of dict
            Output from ChordRecognizer.recognize_sequence().

        Returns
        -------
        smoothed : list of dict
        """
        if not chord_sequence:
            return []

        # Build vocabulary from observed chords + common additions
        vocab = list(set(entry['chord'] for entry in chord_sequence))
        if 'N' not in vocab:
            vocab.append('N')
        vocab.sort()

        trans_matrix = self.build_transition_probs(vocab)
        return self.viterbi_decode(chord_sequence, vocab, trans_matrix)

    def _chord_root_index(self, chord_name):
        """Extract root note index from chord name."""
        if not chord_name or chord_name == 'N':
            return None

        # Reverse enharmonic
        enharmonic = {
            'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
        }

        # Try two-char root
        if len(chord_name) >= 2:
            two = chord_name[:2]
            if two in NOTE_NAMES:
                return NOTE_NAMES.index(two)
            if two in enharmonic:
                return NOTE_NAMES.index(enharmonic[two])

        # Single char root
        one = chord_name[0]
        if one in NOTE_NAMES:
            return NOTE_NAMES.index(one)

        return None
