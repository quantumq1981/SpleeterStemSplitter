"""
Tests for the ChartExporter module.

Validates CSV, Markdown, JSON, simple chart, and timed lyrics export.
"""

import json
import unittest

from ..chart_export import ChartExporter, _format_time, _format_time_short


class TestFormatTime(unittest.TestCase):
    """Test time formatting helpers."""

    def test_format_time_zero(self):
        self.assertEqual(_format_time(0), '0:00.00')

    def test_format_time_seconds(self):
        self.assertEqual(_format_time(5.25), '0:05.25')

    def test_format_time_minutes(self):
        self.assertEqual(_format_time(125.5), '2:05.50')

    def test_format_time_short_zero(self):
        self.assertEqual(_format_time_short(0), '0:00')

    def test_format_time_short_seconds(self):
        self.assertEqual(_format_time_short(65), '1:05')


class TestCSVExport(unittest.TestCase):
    """Test CSV export format."""

    def setUp(self):
        self.exporter = ChartExporter(title='Test Song', artist='Test Artist', key='C major')

    def test_basic_csv(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 1.0, 'chord': 'G', 'confidence': 0.8},
        ]
        csv_str = self.exporter.to_csv(chords)
        self.assertIn('Timestamp', csv_str)
        self.assertIn('Chord', csv_str)
        self.assertIn('C', csv_str)
        self.assertIn('G', csv_str)

    def test_csv_skips_silence(self):
        chords = [
            {'time': 0.0, 'chord': 'N', 'confidence': 0.0},
            {'time': 1.0, 'chord': 'C', 'confidence': 0.9},
        ]
        csv_str = self.exporter.to_csv(chords)
        lines = csv_str.strip().split('\n')
        # Header + 1 data row (N skipped)
        self.assertEqual(len(lines), 2)

    def test_csv_with_flags(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9, 'in_key': True, 'suggestion': None},
            {'time': 1.0, 'chord': 'Ebm', 'confidence': 0.5, 'in_key': False, 'suggestion': 'Em'},
        ]
        csv_str = self.exporter.to_csv(chords, include_flags=True)
        self.assertIn('In Key', csv_str)
        self.assertIn('NO', csv_str)
        self.assertIn('Em', csv_str)

    def test_csv_without_confidence(self):
        chords = [{'time': 0.0, 'chord': 'C', 'confidence': 0.9}]
        csv_str = self.exporter.to_csv(chords, include_confidence=False)
        self.assertNotIn('Confidence', csv_str)

    def test_csv_with_corrections(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9, 'was_corrected': False, 'original_chord': 'C'},
            {'time': 1.0, 'chord': 'G', 'confidence': 0.8, 'was_corrected': True, 'original_chord': 'F#'},
        ]
        csv_str = self.exporter.to_csv(chords)
        self.assertIn('Original', csv_str)
        self.assertIn('Corrected', csv_str)


class TestMarkdownExport(unittest.TestCase):
    """Test Markdown chart export."""

    def setUp(self):
        self.exporter = ChartExporter(title='My Song', artist='My Artist', key='G major', bpm=120)

    def test_markdown_header(self):
        md = self.exporter.to_markdown([])
        self.assertIn('# My Artist - My Song', md)
        self.assertIn('**Key:** G major', md)
        self.assertIn('**BPM:** 120', md)

    def test_markdown_table(self):
        chords = [
            {'time': 0.0, 'chord': 'G', 'confidence': 0.9},
            {'time': 1.0, 'chord': 'C', 'confidence': 0.85},
        ]
        md = self.exporter.to_markdown(chords)
        self.assertIn('| Time | Chord | Confidence |', md)
        self.assertIn('**G**', md)
        self.assertIn('**C**', md)

    def test_markdown_skips_silence(self):
        chords = [
            {'time': 0.0, 'chord': 'N', 'confidence': 0.0},
            {'time': 1.0, 'chord': 'Am', 'confidence': 0.8},
        ]
        md = self.exporter.to_markdown(chords)
        # N should not appear as a chord row
        lines = [l for l in md.split('\n') if '**N**' in l]
        self.assertEqual(len(lines), 0)

    def test_markdown_out_of_key_flagged(self):
        chords = [
            {'time': 0.0, 'chord': 'Ebm', 'confidence': 0.5, 'in_key': False, 'suggestion': 'Em'},
        ]
        md = self.exporter.to_markdown(chords)
        self.assertIn('*(Em?)*', md)
        self.assertIn('out-of-key', md.lower())


class TestJSONExport(unittest.TestCase):
    """Test JSON export format."""

    def setUp(self):
        self.exporter = ChartExporter(title='Test', artist='Artist', key='D minor', bpm=90)

    def test_json_structure(self):
        chords = [
            {'time': 0.0, 'chord': 'Dm', 'confidence': 0.9},
        ]
        json_str = self.exporter.to_json(chords)
        data = json.loads(json_str)
        self.assertIn('metadata', data)
        self.assertIn('chords', data)
        self.assertEqual(data['metadata']['title'], 'Test')
        self.assertEqual(data['metadata']['key'], 'D minor')
        self.assertEqual(data['metadata']['bpm'], 90)

    def test_json_skips_silence(self):
        chords = [
            {'time': 0.0, 'chord': 'N', 'confidence': 0.0},
            {'time': 1.0, 'chord': 'C', 'confidence': 0.9},
        ]
        json_str = self.exporter.to_json(chords)
        data = json.loads(json_str)
        self.assertEqual(len(data['chords']), 1)

    def test_json_compact(self):
        chords = [{'time': 0.0, 'chord': 'C', 'confidence': 0.9}]
        compact = self.exporter.to_json(chords, pretty=False)
        self.assertNotIn('\n', compact)

    def test_json_pretty(self):
        chords = [{'time': 0.0, 'chord': 'C', 'confidence': 0.9}]
        pretty = self.exporter.to_json(chords, pretty=True)
        self.assertIn('\n', pretty)


class TestSimpleChart(unittest.TestCase):
    """Test Nashville-style simple chord chart."""

    def setUp(self):
        self.exporter = ChartExporter(title='Hit Song', artist='Band', key='E major')

    def test_chart_header(self):
        chart = self.exporter.to_simple_chart([])
        self.assertIn('Band - Hit Song', chart)
        self.assertIn('Key: E major', chart)

    def test_chart_with_chords(self):
        chords = [
            {'time': 0.0, 'chord': 'E', 'confidence': 0.9},
            {'time': 1.0, 'chord': 'A', 'confidence': 0.85},
            {'time': 2.0, 'chord': 'B', 'confidence': 0.8},
            {'time': 3.0, 'chord': 'E', 'confidence': 0.9},
        ]
        chart = self.exporter.to_simple_chart(chords, bars_per_line=4)
        self.assertIn('|', chart)
        self.assertIn('E', chart)
        self.assertIn('A', chart)

    def test_chart_skips_silence(self):
        chords = [
            {'time': 0.0, 'chord': 'N', 'confidence': 0.0},
            {'time': 1.0, 'chord': 'C', 'confidence': 0.9},
        ]
        chart = self.exporter.to_simple_chart(chords)
        # N should not appear in the chord bars
        bar_lines = [l for l in chart.split('\n') if l.startswith('|')]
        for line in bar_lines:
            self.assertNotIn(' N ', line)

    def test_no_chords_message(self):
        chords = [{'time': 0.0, 'chord': 'N', 'confidence': 0.0}]
        chart = self.exporter.to_simple_chart(chords)
        self.assertIn('No chords detected', chart)


class TestTimedLyricsFormat(unittest.TestCase):
    """Test timed lyrics export."""

    def setUp(self):
        self.exporter = ChartExporter()

    def test_basic_export(self):
        chords = [
            {'time': 0.0, 'chord': 'C', 'confidence': 0.9},
            {'time': 2.0, 'chord': 'G', 'confidence': 0.8},
        ]
        result = self.exporter.to_timed_lyrics_format(chords)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (0.0, 'C'))
        self.assertEqual(result[1], (2.0, 'G'))

    def test_silence_excluded(self):
        chords = [
            {'time': 0.0, 'chord': 'N', 'confidence': 0.0},
            {'time': 1.0, 'chord': 'Am', 'confidence': 0.8},
        ]
        result = self.exporter.to_timed_lyrics_format(chords)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 'Am')


class TestConfidenceBar(unittest.TestCase):
    """Test the text confidence bar visualization."""

    def setUp(self):
        self.exporter = ChartExporter()

    def test_full_confidence(self):
        bar = self.exporter._confidence_bar(1.0, width=5)
        self.assertEqual(bar, '#####')

    def test_zero_confidence(self):
        bar = self.exporter._confidence_bar(0.0, width=5)
        self.assertEqual(bar, '-----')

    def test_half_confidence(self):
        bar = self.exporter._confidence_bar(0.5, width=4)
        self.assertEqual(bar, '##--')


if __name__ == '__main__':
    unittest.main()
