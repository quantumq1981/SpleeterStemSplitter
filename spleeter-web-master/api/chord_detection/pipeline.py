"""
Chord Detection Pipeline — Full Orchestrator

Chains together all three phases:
  Phase 1: Harmonic Isolation (source separation via Demucs/Spleeter/BS-RoFormer)
  Phase 2: Chromagram extraction via CQT (librosa)
  Phase 3: Chord inference with key-aware filtering and transition smoothing

This module provides a single entry point that takes an audio file (or
pre-separated stems) and produces a complete chord chart.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chromagram import ChromagramExtractor
from .chord_recognizer import ChordRecognizer
from .key_detector import KeyDetector
from .transition_matrix import TransitionMatrix
from .chart_export import ChartExporter


class ChordDetectionPipeline:
    """End-to-end chord detection from audio.

    Parameters
    ----------
    sr : int
        Sample rate for audio processing.
    hop_length : int
        Hop length for chromagram extraction.
    segment_duration : float
        Duration in seconds to aggregate chroma frames.
        Larger = fewer chord changes, more stable.
        Smaller = more responsive to quick changes.
    min_chord_duration : float
        Minimum chord duration (seconds) after simplification.
    smoothing : float
        Transition matrix smoothing factor (0=trust raw detection, 1=trust theory).
    enhanced_chroma : bool
        Use CENS (enhanced) chromagram instead of raw CQT.
    """

    def __init__(
        self,
        sr=22050,
        hop_length=2048,
        segment_duration=0.5,
        min_chord_duration=0.3,
        smoothing=0.6,
        enhanced_chroma=False,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.segment_duration = segment_duration
        self.min_chord_duration = min_chord_duration
        self.smoothing = smoothing
        self.enhanced_chroma = enhanced_chroma

        self.chromagram_extractor = ChromagramExtractor(
            sr=sr, hop_length=hop_length
        )
        self.chord_recognizer = ChordRecognizer()
        self.key_detector = KeyDetector()

    def analyze_file(self, audio_path, title='Unknown', artist='Unknown'):
        """Run full chord detection on a single audio file.

        Parameters
        ----------
        audio_path : str
            Path to audio file (WAV, MP3, FLAC, etc.)
        title : str
            Song title for chart headers.
        artist : str
            Artist name for chart headers.

        Returns
        -------
        result : dict
            {
                'key': str,
                'key_confidence': float,
                'chords_raw': list of dict (every frame),
                'chords_smoothed': list of dict (after Viterbi),
                'chords_simplified': list of dict (collapsed duplicates),
                'chords_flagged': list of dict (with in_key markers),
                'metadata': dict,
            }
        """
        # Phase 2: Extract chromagram
        chroma, times = self.chromagram_extractor.extract_from_file(
            audio_path, enhanced=self.enhanced_chroma
        )

        # Aggregate into segments for stability
        seg_chroma, seg_times = self.chromagram_extractor.aggregate_chroma_segments(
            chroma, times, segment_duration=self.segment_duration
        )

        # Detect key from the full chromagram
        key, key_confidence, all_key_scores = self.key_detector.detect_key(chroma)

        # Phase 3: Raw chord recognition
        chords_raw = self.chord_recognizer.recognize_sequence(seg_chroma, seg_times)

        # Apply transition matrix smoothing (Viterbi)
        transition = TransitionMatrix(key=key, smoothing=self.smoothing)
        chords_smoothed = transition.smooth_sequence(chords_raw)

        # Simplify (collapse consecutive same chords)
        chords_simplified = self.chord_recognizer.simplify_sequence(
            chords_smoothed, min_duration=self.min_chord_duration
        )

        # Flag out-of-key chords
        chords_flagged = self.key_detector.flag_out_of_key_chords(
            chords_simplified, key
        )

        return {
            'key': key,
            'key_confidence': key_confidence,
            'key_scores_top5': all_key_scores[:5],
            'chords_raw': chords_raw,
            'chords_smoothed': chords_smoothed,
            'chords_simplified': chords_simplified,
            'chords_flagged': chords_flagged,
            'metadata': {
                'title': title,
                'artist': artist,
                'sr': self.sr,
                'hop_length': self.hop_length,
                'segment_duration': self.segment_duration,
                'smoothing': self.smoothing,
            },
        }

    def analyze_stems(
        self,
        stem_paths,
        title='Unknown',
        artist='Unknown',
        weights=None,
    ):
        """Analyze multiple separated stems and combine their chroma information.

        This is the recommended path: separate audio first (Phase 1), then
        analyze the harmonic stems (bass + other/keys) for cleaner detection.

        Parameters
        ----------
        stem_paths : dict
            {'bass': '/path/to/bass.wav', 'other': '/path/to/other.wav', ...}
            At minimum, 'other' (or 'keys'/'guitar'/'piano') should be provided.
        title : str
        artist : str
        weights : dict or None
            Weight for each stem's chroma contribution.
            Default: {'other': 0.7, 'bass': 0.3}

        Returns
        -------
        result : dict (same format as analyze_file)
        """
        if weights is None:
            weights = {
                'other': 0.7,
                'bass': 0.3,
                'vocals': 0.0,  # usually omit vocals
                'drums': 0.0,   # drums are atonal
                'piano': 0.7,
                'guitar': 0.7,
            }

        combined_chroma = None
        combined_times = None
        total_weight = 0.0

        for stem_name, path in stem_paths.items():
            if not os.path.exists(path):
                print(f'Warning: stem file not found: {path}')
                continue

            weight = weights.get(stem_name, 0.1)
            if weight <= 0:
                continue

            chroma, times = self.chromagram_extractor.extract_from_file(
                path, enhanced=self.enhanced_chroma
            )

            if combined_chroma is None:
                combined_chroma = chroma * weight
                combined_times = times
            else:
                # Align to shorter length
                min_len = min(combined_chroma.shape[1], chroma.shape[1])
                combined_chroma = combined_chroma[:, :min_len] + chroma[:, :min_len] * weight
                combined_times = combined_times[:min_len]

            total_weight += weight

        if combined_chroma is None:
            raise ValueError('No valid stem files found.')

        # Normalize by total weight
        if total_weight > 0:
            combined_chroma /= total_weight

        # Aggregate into segments
        seg_chroma, seg_times = self.chromagram_extractor.aggregate_chroma_segments(
            combined_chroma, combined_times,
            segment_duration=self.segment_duration,
        )

        # Key detection from combined chroma
        key, key_confidence, all_key_scores = self.key_detector.detect_key(
            combined_chroma
        )

        # Chord recognition
        chords_raw = self.chord_recognizer.recognize_sequence(seg_chroma, seg_times)

        # Viterbi smoothing
        transition = TransitionMatrix(key=key, smoothing=self.smoothing)
        chords_smoothed = transition.smooth_sequence(chords_raw)

        # Simplify
        chords_simplified = self.chord_recognizer.simplify_sequence(
            chords_smoothed, min_duration=self.min_chord_duration
        )

        # Flag out-of-key
        chords_flagged = self.key_detector.flag_out_of_key_chords(
            chords_simplified, key
        )

        return {
            'key': key,
            'key_confidence': key_confidence,
            'key_scores_top5': all_key_scores[:5],
            'chords_raw': chords_raw,
            'chords_smoothed': chords_smoothed,
            'chords_simplified': chords_simplified,
            'chords_flagged': chords_flagged,
            'stems_analyzed': list(stem_paths.keys()),
            'metadata': {
                'title': title,
                'artist': artist,
                'sr': self.sr,
                'hop_length': self.hop_length,
                'segment_duration': self.segment_duration,
                'smoothing': self.smoothing,
                'stem_weights': {k: weights.get(k, 0.1) for k in stem_paths},
            },
        }

    def generate_chart(self, result, format='markdown'):
        """Generate an exportable chart from analysis results.

        Parameters
        ----------
        result : dict
            Output of analyze_file() or analyze_stems().
        format : str
            'csv', 'markdown', 'json', 'simple' (Nashville-style text).

        Returns
        -------
        chart : str
        """
        exporter = ChartExporter(
            title=result['metadata']['title'],
            artist=result['metadata']['artist'],
            key=result['key'],
        )

        chords = result['chords_flagged']

        if format == 'csv':
            return exporter.to_csv(chords)
        elif format == 'markdown':
            return exporter.to_markdown(chords)
        elif format == 'json':
            return exporter.to_json(chords)
        elif format == 'simple':
            return exporter.to_simple_chart(chords)
        else:
            raise ValueError(f'Unknown format: {format}. Use csv/markdown/json/simple.')
