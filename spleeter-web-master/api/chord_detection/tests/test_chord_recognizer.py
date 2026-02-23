"""
Tests for the ChordRecognizer module.

Validates template matching, cosine similarity, enharmonic conversion,
sequence recognition, and simplification.
"""

import numpy as np
import unittest

from ..chord_recognizer import (
    ChordRecognizer,
    CHORD_TEMPLATES,
    CHORD_QUALITIES,
    NOTE_NAMES,
    ENHARMONIC_MAP,
    _intervals_to_template,
    _build_chord_templates,
)


class TestIntervalsToTemplate(unittest.TestCase):
    """Test the helper that converts interval lists to 12-bin templates."""

    def test_major_triad(self):
        template = _intervals_to_template([0, 4, 7])
        expected = np.zeros(12)
        expected[[0, 4, 7]] = 1.0
        np.testing.assert_array_equal(template, expected)

    def test_minor_triad(self):
        template = _intervals_to_template([0, 3, 7])
        expected = np.zeros(12)
        expected[[0, 3, 7]] = 1.0
        np.testing.assert_array_equal(template, expected)

    def test_mod_12_wrapping(self):
        # add9 has interval 14, which should wrap to 2
        template = _intervals_to_template([0, 4, 7, 14])
        self.assertEqual(template[2], 1.0)
        self.assertEqual(template[0], 1.0)
        self.assertEqual(template[4], 1.0)
        self.assertEqual(template[7], 1.0)

    def test_empty_intervals(self):
        template = _intervals_to_template([])
        np.testing.assert_array_equal(template, np.zeros(12))


class TestChordTemplates(unittest.TestCase):
    """Test the global template dictionary."""

    def test_template_count(self):
        # 12 roots * 19 qualities + 1 "N" = 229
        expected_count = 12 * len(CHORD_QUALITIES) + 1
        self.assertEqual(len(CHORD_TEMPLATES), expected_count)

    def test_n_template_is_silent(self):
        np.testing.assert_array_equal(CHORD_TEMPLATES['N'], np.zeros(12))

    def test_c_major_template(self):
        # C major = C(0), E(4), G(7)
        t = CHORD_TEMPLATES['C']
        self.assertEqual(t[0], 1.0)  # C
        self.assertEqual(t[4], 1.0)  # E
        self.assertEqual(t[7], 1.0)  # G
        self.assertEqual(sum(t), 3.0)

    def test_a_minor_template(self):
        # Am = A(9), C(0), E(4)
        t = CHORD_TEMPLATES['Am']
        self.assertEqual(t[9], 1.0)  # A
        self.assertEqual(t[0], 1.0)  # C
        self.assertEqual(t[4], 1.0)  # E
        self.assertEqual(sum(t), 3.0)

    def test_g7_template(self):
        # G7 = G(7), B(11), D(2), F(5)
        t = CHORD_TEMPLATES['G7']
        self.assertEqual(t[7], 1.0)
        self.assertEqual(t[11], 1.0)
        self.assertEqual(t[2], 1.0)
        self.assertEqual(t[5], 1.0)
        self.assertEqual(sum(t), 4.0)

    def test_all_templates_binary(self):
        for name, template in CHORD_TEMPLATES.items():
            for val in template:
                self.assertIn(val, [0.0, 1.0], f"Template {name} has non-binary value")


class TestChordRecognizer(unittest.TestCase):
    """Test the main ChordRecognizer class."""

    def setUp(self):
        self.recognizer = ChordRecognizer(min_energy=0.1, use_enharmonic=True)

    def test_recognize_c_major(self):
        # Pure C major chroma: energy at C, E, G
        chroma = np.zeros(12)
        chroma[[0, 4, 7]] = 1.0
        chord, confidence = self.recognizer.recognize_frame(chroma)
        self.assertEqual(chord, 'C')
        self.assertGreater(confidence, 0.9)

    def test_recognize_a_minor(self):
        chroma = np.zeros(12)
        chroma[[9, 0, 4]] = 1.0  # A, C, E
        chord, confidence = self.recognizer.recognize_frame(chroma)
        self.assertEqual(chord, 'Am')
        self.assertGreater(confidence, 0.9)

    def test_recognize_silence(self):
        chroma = np.zeros(12)
        chord, confidence = self.recognizer.recognize_frame(chroma)
        self.assertEqual(chord, 'N')
        self.assertEqual(confidence, 0.0)

    def test_recognize_below_min_energy(self):
        chroma = np.ones(12) * 0.005  # total = 0.06, below 0.1
        chord, confidence = self.recognizer.recognize_frame(chroma)
        self.assertEqual(chord, 'N')

    def test_enharmonic_bb(self):
        # A# should display as Bb
        chroma = np.zeros(12)
        chroma[[10, 2, 5]] = 1.0  # A#, D, F -> A# minor = Bbm
        chord, confidence = self.recognizer.recognize_frame(chroma)
        self.assertTrue(chord.startswith('Bb'), f"Expected Bb prefix, got {chord}")

    def test_enharmonic_disabled(self):
        rec = ChordRecognizer(use_enharmonic=False)
        chroma = np.zeros(12)
        chroma[[10, 2, 5]] = 1.0
        chord, _ = rec.recognize_frame(chroma)
        self.assertTrue(chord.startswith('A#'), f"Expected A# prefix, got {chord}")

    def test_recognize_sequence(self):
        # 3-frame chromagram: C, G, Am
        chroma = np.zeros((12, 3))
        chroma[[0, 4, 7], 0] = 1.0   # C
        chroma[[7, 11, 2], 1] = 1.0   # G
        chroma[[9, 0, 4], 2] = 1.0    # Am
        times = np.array([0.0, 0.5, 1.0])

        chords = self.recognizer.recognize_sequence(chroma, times)
        self.assertEqual(len(chords), 3)
        self.assertEqual(chords[0]['chord'], 'C')
        self.assertEqual(chords[1]['chord'], 'G')
        self.assertEqual(chords[2]['chord'], 'Am')

    def test_recognize_sequence_times(self):
        chroma = np.zeros((12, 2))
        chroma[[0, 4, 7], 0] = 1.0
        chroma[[0, 4, 7], 1] = 1.0
        times = np.array([1.5, 3.0])

        chords = self.recognizer.recognize_sequence(chroma, times)
        self.assertAlmostEqual(chords[0]['time'], 1.5)
        self.assertAlmostEqual(chords[1]['time'], 3.0)

    def test_confidence_range(self):
        chroma = np.zeros(12)
        chroma[[0, 4, 7]] = 1.0
        _, confidence = self.recognizer.recognize_frame(chroma)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertAlmostEqual(confidence, 1.0, places=5)  # floating-point tolerance


class TestSimplifySequence(unittest.TestCase):
    """Test chord sequence simplification."""

    def setUp(self):
        self.recognizer = ChordRecognizer()

    def test_collapse_consecutive(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 0.5, 'chord': 'C', 'confidence': 0.85},
            {'time': 1.0, 'chord': 'G', 'confidence': 0.8},
            {'time': 1.5, 'chord': 'G', 'confidence': 0.75},
        ]
        simplified = self.recognizer.simplify_sequence(chords, min_duration=0)
        self.assertEqual(len(simplified), 2)
        self.assertEqual(simplified[0]['chord'], 'C')
        self.assertEqual(simplified[1]['chord'], 'G')

    def test_timing_preserved(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 0.5, 'chord': 'C', 'confidence': 0.85},
            {'time': 1.0, 'chord': 'Am', 'confidence': 0.8},
        ]
        simplified = self.recognizer.simplify_sequence(chords, min_duration=0)
        self.assertAlmostEqual(simplified[0]['time'], 0.0)
        self.assertAlmostEqual(simplified[0]['end_time'], 1.0)
        self.assertAlmostEqual(simplified[1]['time'], 1.0)

    def test_confidence_averaged(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.8},
            {'time': 0.5, 'chord': 'C', 'confidence': 1.0},
        ]
        simplified = self.recognizer.simplify_sequence(chords, min_duration=0)
        self.assertAlmostEqual(simplified[0]['confidence'], 0.9)

    def test_empty_input(self):
        result = self.recognizer.simplify_sequence([], min_duration=0)
        self.assertEqual(result, [])

    def test_single_chord(self):
        chords = [{'time': 0.0, 'chord': 'D', 'confidence': 0.7}]
        simplified = self.recognizer.simplify_sequence(chords, min_duration=0)
        self.assertEqual(len(simplified), 1)
        self.assertEqual(simplified[0]['chord'], 'D')

    def test_min_duration_filtering(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 1.0, 'chord': 'G', 'confidence': 0.8},
            {'time': 1.1, 'chord': 'Am', 'confidence': 0.7},  # only 0.1s - should merge
        ]
        simplified = self.recognizer.simplify_sequence(chords, min_duration=0.3)
        # The short G chord (0.1s) should be merged
        self.assertLessEqual(len(simplified), 2)


class TestGetChordNotes(unittest.TestCase):
    """Test chord-to-notes lookup."""

    def setUp(self):
        self.recognizer = ChordRecognizer()

    def test_c_major_notes(self):
        notes = self.recognizer.get_chord_notes('C')
        self.assertIsNotNone(notes)
        self.assertIn('C', notes)
        self.assertIn('E', notes)
        self.assertIn('G', notes)

    def test_enharmonic_lookup(self):
        # Bb -> looks up A# internally
        notes = self.recognizer.get_chord_notes('Bb')
        self.assertIsNotNone(notes)

    def test_unknown_chord(self):
        notes = self.recognizer.get_chord_notes('Xaug42')
        self.assertIsNone(notes)


if __name__ == '__main__':
    unittest.main()
