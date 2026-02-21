"""
Chord Detection AI Module

A three-phase pipeline for extracting chord progressions from audio:
  Phase 1: Harmonic Isolation (handled by existing separators - Demucs/Spleeter/BS-RoFormer)
  Phase 2: Chromagram extraction via Constant-Q Transform (CQT)
  Phase 3: Chord inference with key-aware theory filtering and transition modeling
"""

from .chromagram import ChromagramExtractor
from .chord_recognizer import ChordRecognizer
from .key_detector import KeyDetector
from .transition_matrix import TransitionMatrix
from .chart_export import ChartExporter
from .pipeline import ChordDetectionPipeline

__all__ = [
    'ChromagramExtractor',
    'ChordRecognizer',
    'KeyDetector',
    'TransitionMatrix',
    'ChartExporter',
    'ChordDetectionPipeline',
]
