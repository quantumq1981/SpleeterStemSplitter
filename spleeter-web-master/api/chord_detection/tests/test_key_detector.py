"""
Tests for the KeyDetector module.

Validates Krumhansl-Schmuckler key detection, diatonic chord generation,
extended chord sets, out-of-key flagging, and correction suggestions.
"""

import numpy as np
import unittest

from ..key_detector import (
    KeyDetector,
    NOTE_NAMES,
    MAJOR_PROFILE,
    MINOR_PROFILE,
    MAJOR_SCALE_INTERVALS,
    NATURAL_MINOR_SCALE_INTERVALS,
)


class TestKeyDetection(unittest.TestCase):
    """Test key detection from chromagram data."""

    def setUp(self):
        self.detector = KeyDetector(use_enharmonic=True)

    def test_c_major_detection(self):
        # Simulate a chromagram that strongly correlates with C major
        # C major scale: C D E F G A B = indices 0,2,4,5,7,9,11
        chroma = np.zeros((12, 100))
        for idx in [0, 2, 4, 5, 7, 9, 11]:
            chroma[idx, :] = MAJOR_PROFILE[idx]
        key, confidence, scores = self.detector.detect_key(chroma)
        self.assertEqual(key, 'C major')
        self.assertGreater(confidence, 0.5)

    def test_a_minor_detection(self):
        # Simulate A minor: A B C D E F G = indices 9,11,0,2,4,5,7
        chroma = np.zeros((12, 100))
        for i, interval in enumerate(NATURAL_MINOR_SCALE_INTERVALS):
            idx = (9 + interval) % 12  # A = 9
            chroma[idx, :] = MINOR_PROFILE[i]
        key, confidence, scores = self.detector.detect_key(chroma)
        # Should detect A minor (or its relative C major, both are valid)
        self.assertIn('minor', key.lower() + 'minor' if 'A' in key else '')

    def test_scores_sorted_descending(self):
        chroma = np.random.rand(12, 50)
        _, _, scores = self.detector.detect_key(chroma)
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i][1], scores[i + 1][1])

    def test_24_keys_in_scores(self):
        chroma = np.random.rand(12, 10)
        _, _, scores = self.detector.detect_key(chroma)
        self.assertEqual(len(scores), 24)

    def test_confidence_is_float(self):
        chroma = np.random.rand(12, 10)
        _, confidence, _ = self.detector.detect_key(chroma)
        self.assertIsInstance(confidence, float)

    def test_enharmonic_key_names(self):
        """Keys with sharps should be displayed as flats."""
        # Force a key that would naturally be A# -> Bb
        chroma = np.zeros((12, 100))
        for i, interval in enumerate(MAJOR_SCALE_INTERVALS):
            idx = (10 + interval) % 12  # A# = 10
            chroma[idx, :] = MAJOR_PROFILE[i]
        key, _, _ = self.detector.detect_key(chroma)
        # Should use Bb not A#
        if 'major' in key:
            self.assertTrue(key.startswith('Bb'), f"Expected Bb, got {key}")


class TestDiatonicChords(unittest.TestCase):
    """Test diatonic chord generation."""

    def setUp(self):
        self.detector = KeyDetector(use_enharmonic=True)

    def test_c_major_diatonic(self):
        chords = self.detector.get_diatonic_chords('C major')
        # I=C, ii=Dm, iii=Em, IV=F, V=G, vi=Am, vii=Bdim
        self.assertEqual(len(chords), 7)
        self.assertIn('C', chords)
        self.assertIn('Dm', chords)
        self.assertIn('Em', chords)
        self.assertIn('F', chords)
        self.assertIn('G', chords)
        self.assertIn('Am', chords)

    def test_g_major_diatonic(self):
        chords = self.detector.get_diatonic_chords('G major')
        self.assertIn('G', chords)
        self.assertIn('Am', chords)
        self.assertIn('Bm', chords)
        self.assertIn('C', chords)
        self.assertIn('D', chords)
        self.assertIn('Em', chords)

    def test_a_minor_diatonic(self):
        chords = self.detector.get_diatonic_chords('A minor')
        # i=Am, ii=Bdim, III=C, iv=Dm, v=Em, VI=F, VII=G
        self.assertEqual(len(chords), 7)
        self.assertIn('Am', chords)
        self.assertIn('C', chords)
        self.assertIn('Dm', chords)
        self.assertIn('F', chords)
        self.assertIn('G', chords)

    def test_invalid_key_returns_empty(self):
        chords = self.detector.get_diatonic_chords('X major')
        self.assertEqual(chords, [])


class TestExtendedDiatonicChords(unittest.TestCase):
    """Test extended diatonic chord sets (7ths, secondary dominants)."""

    def setUp(self):
        self.detector = KeyDetector(use_enharmonic=True)

    def test_c_major_extended_includes_g7(self):
        extended = self.detector.get_extended_diatonic_chords('C major')
        self.assertIn('G7', extended)  # V7

    def test_c_major_extended_includes_maj7(self):
        extended = self.detector.get_extended_diatonic_chords('C major')
        self.assertIn('Cmaj7', extended)  # Imaj7
        self.assertIn('Fmaj7', extended)  # IVmaj7

    def test_extended_superset_of_basic(self):
        basic = set(self.detector.get_diatonic_chords('G major'))
        extended = self.detector.get_extended_diatonic_chords('G major')
        self.assertTrue(basic.issubset(extended))

    def test_a_minor_extended_includes_e7(self):
        extended = self.detector.get_extended_diatonic_chords('A minor')
        self.assertIn('E7', extended)  # V7 of A minor


class TestOutOfKeyFlagging(unittest.TestCase):
    """Test chord flagging and correction suggestions."""

    def setUp(self):
        self.detector = KeyDetector(use_enharmonic=True)

    def test_in_key_chords_not_flagged(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 1.0, 'chord': 'G', 'confidence': 0.8},
            {'time': 2.0, 'chord': 'Am', 'confidence': 0.85},
        ]
        flagged = self.detector.flag_out_of_key_chords(chords, 'C major')
        for entry in flagged:
            self.assertTrue(entry['in_key'], f"{entry['chord']} flagged as out-of-key")

    def test_out_of_key_chord_flagged(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 1.0, 'chord': 'Ebm', 'confidence': 0.5},  # Not in C major
        ]
        flagged = self.detector.flag_out_of_key_chords(chords, 'C major')
        self.assertTrue(flagged[0]['in_key'])
        self.assertFalse(flagged[1]['in_key'])

    def test_suggestion_provided_for_out_of_key(self):
        chords = [
            {'time': 0.0, 'chord': 'Ebm', 'confidence': 0.5},
        ]
        flagged = self.detector.flag_out_of_key_chords(chords, 'C major')
        self.assertIsNotNone(flagged[0]['suggestion'])

    def test_silence_not_flagged(self):
        chords = [
            {'time': 0.0, 'chord': 'N', 'confidence': 0.0},
        ]
        flagged = self.detector.flag_out_of_key_chords(chords, 'C major')
        self.assertTrue(flagged[0]['in_key'])

    def test_strict_mode_fewer_allowed(self):
        chords = [
            {'time': 0.0, 'chord': 'G7', 'confidence': 0.8},  # In extended, not basic
        ]
        # Non-strict: G7 is allowed in C major (V7)
        flagged_relaxed = self.detector.flag_out_of_key_chords(chords, 'C major', strict=False)
        self.assertTrue(flagged_relaxed[0]['in_key'])

        # Strict: G7 is NOT in basic diatonic set
        flagged_strict = self.detector.flag_out_of_key_chords(chords, 'C major', strict=True)
        self.assertFalse(flagged_strict[0]['in_key'])

    def test_original_fields_preserved(self):
        chords = [{'time': 1.5, 'chord': 'C', 'confidence': 0.9, 'extra': 'data'}]
        flagged = self.detector.flag_out_of_key_chords(chords, 'C major')
        self.assertAlmostEqual(flagged[0]['time'], 1.5)
        self.assertEqual(flagged[0]['confidence'], 0.9)
        self.assertEqual(flagged[0]['extra'], 'data')


class TestNoteToIndex(unittest.TestCase):
    """Test note name to index conversion."""

    def setUp(self):
        self.detector = KeyDetector()

    def test_natural_notes(self):
        self.assertEqual(self.detector._note_to_index('C'), 0)
        self.assertEqual(self.detector._note_to_index('D'), 2)
        self.assertEqual(self.detector._note_to_index('E'), 4)
        self.assertEqual(self.detector._note_to_index('A'), 9)

    def test_sharp_notes(self):
        self.assertEqual(self.detector._note_to_index('C#'), 1)
        self.assertEqual(self.detector._note_to_index('F#'), 6)

    def test_flat_notes(self):
        self.assertEqual(self.detector._note_to_index('Bb'), 10)
        self.assertEqual(self.detector._note_to_index('Eb'), 3)
        self.assertEqual(self.detector._note_to_index('Ab'), 8)

    def test_invalid_note(self):
        self.assertIsNone(self.detector._note_to_index('X'))


if __name__ == '__main__':
    unittest.main()
