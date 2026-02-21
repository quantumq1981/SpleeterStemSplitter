"""
Chart Export Module

Formats chord analysis results into scannable charts for gigging musicians.
Supports:
  - CSV (for spreadsheets, databases)
  - Markdown (for quick reference, printing)
  - JSON (for programmatic use)
  - Plain text "Nashville Number" style charts
"""

import csv
import json
import io
from typing import List, Dict, Optional


def _format_time(seconds):
    """Format seconds as MM:SS.ms"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f'{minutes}:{secs:05.2f}'


def _format_time_short(seconds):
    """Format seconds as M:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f'{minutes}:{secs:02d}'


class ChartExporter:
    """Exports chord analysis results in various formats.

    Parameters
    ----------
    title : str
        Song title.
    artist : str
        Artist name.
    key : str
        Detected key (e.g., "G major").
    bpm : float or None
        Tempo in BPM (if detected).
    """

    def __init__(self, title='Unknown', artist='Unknown', key='', bpm=None):
        self.title = title
        self.artist = artist
        self.key = key
        self.bpm = bpm

    def to_csv(self, chords, include_confidence=True, include_flags=True):
        """Export chord sequence as CSV.

        Parameters
        ----------
        chords : list of dict
            Chord sequence with 'time', 'chord', 'confidence', etc.
        include_confidence : bool
            Include confidence column.
        include_flags : bool
            Include in_key and suggestion columns if available.

        Returns
        -------
        csv_string : str
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        headers = ['Timestamp', 'Time (sec)', 'Chord']
        if include_confidence:
            headers.append('Confidence')
        if include_flags and chords and 'in_key' in chords[0]:
            headers.extend(['In Key', 'Suggestion'])
        if chords and 'was_corrected' in chords[0]:
            headers.extend(['Original', 'Corrected'])

        writer.writerow(headers)

        # Data
        for entry in chords:
            if entry['chord'] == 'N':
                continue  # Skip silence in charts

            row = [
                _format_time(entry['time']),
                f"{entry['time']:.2f}",
                entry['chord'],
            ]
            if include_confidence:
                row.append(f"{entry.get('confidence', 0):.3f}")
            if include_flags and 'in_key' in entry:
                row.append('Yes' if entry['in_key'] else 'NO')
                row.append(entry.get('suggestion', '') or '')
            if 'was_corrected' in entry:
                row.append(entry.get('original_chord', ''))
                row.append('Yes' if entry['was_corrected'] else '')

            writer.writerow(row)

        return output.getvalue()

    def to_markdown(self, chords, sections=None):
        """Export as a Markdown chord chart.

        Parameters
        ----------
        chords : list of dict
            Chord sequence.
        sections : list of dict or None
            Optional section markers: [{'time': float, 'label': str}, ...]

        Returns
        -------
        md : str
        """
        lines = []

        # Header
        lines.append(f'# {self.artist} - {self.title}')
        lines.append('')
        if self.key:
            lines.append(f'**Key:** {self.key}')
        if self.bpm:
            lines.append(f'**BPM:** {self.bpm:.0f}')
        lines.append('')

        # Chord table
        lines.append('| Time | Chord | Confidence |')
        lines.append('|------|-------|------------|')

        for entry in chords:
            if entry['chord'] == 'N':
                continue

            time_str = _format_time_short(entry['time'])
            chord = entry['chord']
            conf = entry.get('confidence', 0)

            # Flag out-of-key chords
            flag = ''
            if entry.get('in_key') is False:
                flag = ' *'
                suggestion = entry.get('suggestion')
                if suggestion:
                    flag = f' *({suggestion}?)*'

            conf_bar = self._confidence_bar(conf)
            lines.append(f'| {time_str} | **{chord}**{flag} | {conf_bar} {conf:.0%} |')

        lines.append('')

        # Legend
        if any(e.get('in_key') is False for e in chords if e['chord'] != 'N'):
            lines.append('*Italicized chords are flagged as potentially out-of-key*')
            lines.append('')

        return '\n'.join(lines)

    def to_simple_chart(self, chords, bars_per_line=4):
        """Export as a simple text chord chart (Nashville-style).

        Groups chords into measures/bars for quick reading on stage.

        Parameters
        ----------
        chords : list of dict
            Simplified chord sequence (collapsed duplicates).
        bars_per_line : int
            Number of chord changes per line.

        Returns
        -------
        chart : str
        """
        lines = []
        lines.append(f'{self.artist} - {self.title}')
        if self.key:
            lines.append(f'Key: {self.key}')
        lines.append('=' * 40)
        lines.append('')

        # Filter out silence
        chord_changes = [e for e in chords if e['chord'] != 'N']

        if not chord_changes:
            lines.append('(No chords detected)')
            return '\n'.join(lines)

        # Group into lines
        current_line = []
        for entry in chord_changes:
            time_str = _format_time_short(entry['time'])
            chord = entry['chord']
            current_line.append(f'{chord:8s}')

            if len(current_line) >= bars_per_line:
                # Prefix with time of first chord in line
                lines.append('| ' + ' | '.join(current_line) + ' |')
                current_line = []

        if current_line:
            # Pad remaining
            while len(current_line) < bars_per_line:
                current_line.append('        ')
            lines.append('| ' + ' | '.join(current_line) + ' |')

        lines.append('')
        return '\n'.join(lines)

    def to_json(self, chords, pretty=True):
        """Export full analysis as JSON.

        Parameters
        ----------
        chords : list of dict
            Chord sequence.
        pretty : bool
            Format with indentation.

        Returns
        -------
        json_string : str
        """
        data = {
            'metadata': {
                'title': self.title,
                'artist': self.artist,
                'key': self.key,
                'bpm': self.bpm,
            },
            'chords': [
                {k: v for k, v in entry.items() if v is not None}
                for entry in chords
                if entry['chord'] != 'N'
            ],
        }

        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)

    def to_timed_lyrics_format(self, chords):
        """Export in a format suitable for overlay on lyrics/lead sheets.

        Returns
        -------
        list of tuple
            [(time_seconds, chord_name), ...]
        """
        return [
            (entry['time'], entry['chord'])
            for entry in chords
            if entry['chord'] != 'N'
        ]

    def _confidence_bar(self, confidence, width=5):
        """Generate a simple text confidence bar."""
        filled = int(round(confidence * width))
        return '#' * filled + '-' * (width - filled)
