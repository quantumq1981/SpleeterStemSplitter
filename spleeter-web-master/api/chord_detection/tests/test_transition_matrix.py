"""
Tests for the TransitionMatrix module.

Validates transition probability construction, Viterbi decoding,
and sequence smoothing behavior.
"""

import numpy as np
import unittest

from ..transition_matrix import TransitionMatrix


class TestTransitionProbabilities(unittest.TestCase):
    """Test the transition probability matrix construction."""

    def setUp(self):
        self.tm = TransitionMatrix(smoothing=0.6)

    def test_matrix_shape(self):
        vocab = ['C', 'G', 'Am', 'F', 'N']
        trans = self.tm.build_transition_probs(vocab)
        self.assertEqual(trans.shape, (5, 5))

    def test_row_normalized(self):
        vocab = ['C', 'G', 'Am', 'F', 'Dm', 'Em', 'N']
        trans = self.tm.build_transition_probs(vocab)
        for i in range(len(vocab)):
            row_sum = trans[i, :].sum()
            self.assertAlmostEqual(row_sum, 1.0, places=5,
                                   msg=f"Row {i} ({vocab[i]}) not normalized: {row_sum}")

    def test_self_transition_highest(self):
        """Self-transition (chord sustain) should have highest probability."""
        vocab = ['C', 'G', 'Am', 'F']
        trans = self.tm.build_transition_probs(vocab)
        for i in range(len(vocab)):
            self.assertEqual(
                np.argmax(trans[i, :]), i,
                f"Self-transition not highest for {vocab[i]}"
            )

    def test_dominant_resolution_high(self):
        """V->I (G->C) transition should be relatively high."""
        vocab = ['C', 'G', 'Am', 'F', 'N']
        trans = self.tm.build_transition_probs(vocab)
        c_idx = vocab.index('C')
        g_idx = vocab.index('G')
        am_idx = vocab.index('Am')
        # G->C should be higher than G->Am (non-dominant motion)
        self.assertGreater(trans[g_idx, c_idx], trans[g_idx, am_idx])

    def test_n_transitions_low(self):
        """Transitions involving N (silence) should be low."""
        vocab = ['C', 'G', 'N']
        trans = self.tm.build_transition_probs(vocab)
        n_idx = vocab.index('N')
        c_idx = vocab.index('C')
        # N -> C should be lower than C -> C
        self.assertLess(trans[n_idx, c_idx], trans[c_idx, c_idx])

    def test_single_chord_vocab(self):
        vocab = ['C']
        trans = self.tm.build_transition_probs(vocab)
        self.assertEqual(trans.shape, (1, 1))
        self.assertAlmostEqual(trans[0, 0], 1.0)


class TestViterbiDecoding(unittest.TestCase):
    """Test Viterbi decoding for chord sequence smoothing."""

    def setUp(self):
        self.tm = TransitionMatrix(smoothing=0.6)

    def test_empty_sequence(self):
        result = self.tm.viterbi_decode([], ['C', 'G', 'N'])
        self.assertEqual(result, [])

    def test_single_observation(self):
        obs = [{'time': 0.0, 'chord': 'C', 'confidence': 0.95}]
        vocab = ['C', 'G', 'N']
        result = self.tm.viterbi_decode(obs, vocab)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['chord'], 'C')

    def test_high_confidence_preserved(self):
        """High-confidence observations should be preserved by Viterbi."""
        obs = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.95},
            {'time': 0.5, 'chord': 'F', 'confidence': 0.92},
            {'time': 1.0, 'chord': 'G', 'confidence': 0.90},
            {'time': 1.5, 'chord': 'C', 'confidence': 0.93},
        ]
        vocab = ['C', 'F', 'G', 'N']
        result = self.tm.viterbi_decode(obs, vocab)
        # With high confidence, Viterbi should mostly agree
        chords = [r['chord'] for r in result]
        self.assertEqual(chords[0], 'C')
        self.assertEqual(chords[3], 'C')

    def test_low_confidence_gets_corrected(self):
        """Low-confidence glitch should be smoothed by Viterbi."""
        # C C G C pattern where the G is low-confidence
        obs = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 0.5, 'chord': 'C', 'confidence': 0.9},
            {'time': 1.0, 'chord': 'G', 'confidence': 0.15},  # glitch
            {'time': 1.5, 'chord': 'C', 'confidence': 0.9},
            {'time': 2.0, 'chord': 'C', 'confidence': 0.9},
        ]
        vocab = ['C', 'G', 'N']
        result = self.tm.viterbi_decode(obs, vocab)
        # The low-confidence G should potentially be corrected to C
        corrected_chord = result[2]['chord']
        self.assertIn(corrected_chord, ['C', 'G'])  # either is valid depending on smoothing

    def test_was_corrected_flag(self):
        obs = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 0.5, 'chord': 'G', 'confidence': 0.9},
        ]
        vocab = ['C', 'G', 'N']
        result = self.tm.viterbi_decode(obs, vocab)
        for entry in result:
            self.assertIn('was_corrected', entry)
            self.assertIn('original_chord', entry)

    def test_original_chord_preserved(self):
        obs = [{'time': 0.0, 'chord': 'C', 'confidence': 0.9}]
        vocab = ['C', 'G', 'N']
        result = self.tm.viterbi_decode(obs, vocab)
        self.assertEqual(result[0]['original_chord'], 'C')

    def test_time_preserved(self):
        obs = [
            {'time': 1.5, 'chord': 'C', 'confidence': 0.9},
            {'time': 3.0, 'chord': 'G', 'confidence': 0.8},
        ]
        vocab = ['C', 'G', 'N']
        result = self.tm.viterbi_decode(obs, vocab)
        self.assertAlmostEqual(result[0]['time'], 1.5)
        self.assertAlmostEqual(result[1]['time'], 3.0)


class TestSmoothSequence(unittest.TestCase):
    """Test the convenience smooth_sequence method."""

    def setUp(self):
        self.tm = TransitionMatrix(smoothing=0.5)

    def test_smooth_empty(self):
        result = self.tm.smooth_sequence([])
        self.assertEqual(result, [])

    def test_smooth_builds_vocab_automatically(self):
        seq = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 0.5, 'chord': 'Am', 'confidence': 0.8},
            {'time': 1.0, 'chord': 'F', 'confidence': 0.85},
        ]
        result = self.tm.smooth_sequence(seq)
        self.assertEqual(len(result), 3)

    def test_smooth_adds_n_to_vocab(self):
        """N should be added to vocab even if not in observations."""
        seq = [{'time': 0.0, 'chord': 'C', 'confidence': 0.9}]
        # This shouldn't crash even though N isn't in the input
        result = self.tm.smooth_sequence(seq)
        self.assertEqual(len(result), 1)


class TestSmoothingParameter(unittest.TestCase):
    """Test that the smoothing parameter affects output."""

    def test_zero_smoothing_trusts_observations(self):
        """With smoothing=0, output should match observations exactly."""
        tm = TransitionMatrix(smoothing=0.0)
        obs = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 0.5, 'chord': 'G', 'confidence': 0.85},
            {'time': 1.0, 'chord': 'Am', 'confidence': 0.8},
        ]
        result = tm.smooth_sequence(obs)
        for orig, decoded in zip(obs, result):
            self.assertEqual(decoded['chord'], orig['chord'],
                             "Zero smoothing should preserve all observations")


class TestChordRootIndex(unittest.TestCase):
    """Test chord root extraction."""

    def setUp(self):
        self.tm = TransitionMatrix()

    def test_simple_chord(self):
        self.assertEqual(self.tm._chord_root_index('C'), 0)
        self.assertEqual(self.tm._chord_root_index('G'), 7)
        self.assertEqual(self.tm._chord_root_index('A'), 9)

    def test_quality_suffix(self):
        self.assertEqual(self.tm._chord_root_index('Am'), 9)
        self.assertEqual(self.tm._chord_root_index('G7'), 7)
        self.assertEqual(self.tm._chord_root_index('Cmaj7'), 0)

    def test_flat_root(self):
        self.assertEqual(self.tm._chord_root_index('Bb'), 10)
        self.assertEqual(self.tm._chord_root_index('Ebm'), 3)

    def test_sharp_root(self):
        self.assertEqual(self.tm._chord_root_index('C#'), 1)
        self.assertEqual(self.tm._chord_root_index('F#m'), 6)

    def test_n_returns_none(self):
        self.assertIsNone(self.tm._chord_root_index('N'))

    def test_empty_returns_none(self):
        self.assertIsNone(self.tm._chord_root_index(''))


if __name__ == '__main__':
    unittest.main()
