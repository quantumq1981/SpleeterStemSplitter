#!/usr/bin/env python3
"""
Chord Analyzer CLI — Standalone chord detection tool

Usage:
  # Analyze a single audio file (full mix)
  python chord_analyzer_cli.py song.mp3

  # Analyze with specific output format
  python chord_analyzer_cli.py song.mp3 --format csv --output chart.csv

  # Analyze pre-separated stems (best accuracy)
  python chord_analyzer_cli.py --stems bass=bass.wav other=other.wav

  # Adjust sensitivity
  python chord_analyzer_cli.py song.mp3 --segment-duration 0.25 --smoothing 0.8

  # Full pipeline: separate first, then analyze
  python chord_analyzer_cli.py song.mp3 --separate --separator demucs
"""

import argparse
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_stems_arg(stems_list):
    """Parse stem arguments like 'bass=path/to/bass.wav other=path/to/other.wav'"""
    stems = {}
    for item in stems_list:
        if '=' not in item:
            print(f"Error: stem argument must be in format 'name=path', got '{item}'")
            sys.exit(1)
        name, path = item.split('=', 1)
        stems[name] = path
    return stems


def run_separation(audio_path, separator_name='htdemucs', output_dir=None):
    """Run source separation using Demucs (Phase 1).

    Returns dict of stem paths.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(audio_path), 'separated')

    print(f'Phase 1: Separating stems with {separator_name}...')
    print(f'  Input: {audio_path}')
    print(f'  Output dir: {output_dir}')

    try:
        import demucs.separate
        demucs.separate.main([
            '-n', separator_name,
            '-o', output_dir,
            audio_path,
        ])
    except ImportError:
        print('Error: demucs is not installed. Install with: pip install demucs')
        print('Falling back to full-mix analysis (less accurate)...')
        return None
    except Exception as e:
        print(f'Separation failed: {e}')
        print('Falling back to full-mix analysis...')
        return None

    # Find separated stems
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, separator_name, basename)

    stems = {}
    for stem_name in ['vocals', 'drums', 'bass', 'other']:
        for ext in ['.wav', '.mp3', '.flac']:
            stem_path = os.path.join(stem_dir, f'{stem_name}{ext}')
            if os.path.exists(stem_path):
                stems[stem_name] = stem_path
                break

    if stems:
        print(f'  Found stems: {", ".join(stems.keys())}')
    else:
        print(f'  Warning: no stems found in {stem_dir}')
        return None

    return stems


def main():
    parser = argparse.ArgumentParser(
        description='Chord Analyzer CLI — Detect chords from audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.mp3
  %(prog)s song.mp3 --format csv --output chart.csv
  %(prog)s --stems bass=bass.wav other=other.wav
  %(prog)s song.mp3 --separate --separator htdemucs
  %(prog)s song.mp3 --smoothing 0.8 --segment-duration 0.25
        """,
    )

    parser.add_argument(
        'audio_file', nargs='?', default=None,
        help='Path to audio file (MP3, WAV, FLAC, etc.)',
    )
    parser.add_argument(
        '--stems', nargs='+', metavar='NAME=PATH',
        help='Pre-separated stem files (e.g., bass=bass.wav other=other.wav)',
    )
    parser.add_argument(
        '--format', '-f', choices=['csv', 'markdown', 'json', 'simple', 'all'],
        default='simple',
        help='Output format (default: simple)',
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Output file path (default: stdout)',
    )
    parser.add_argument(
        '--separate', action='store_true',
        help='Run source separation before chord detection (requires demucs)',
    )
    parser.add_argument(
        '--separator', default='htdemucs',
        help='Separator model name for --separate (default: htdemucs)',
    )
    parser.add_argument(
        '--title', default=None,
        help='Song title (auto-detected from filename if not set)',
    )
    parser.add_argument(
        '--artist', default=None,
        help='Artist name',
    )

    # Tuning parameters
    parser.add_argument(
        '--segment-duration', type=float, default=0.5,
        help='Segment duration in seconds for chord aggregation (default: 0.5)',
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.6,
        help='Transition smoothing factor 0-1 (default: 0.6, higher=more theory)',
    )
    parser.add_argument(
        '--min-duration', type=float, default=0.3,
        help='Minimum chord duration in seconds (default: 0.3)',
    )
    parser.add_argument(
        '--enhanced', action='store_true',
        help='Use enhanced CENS chromagram (more robust, less detail)',
    )
    parser.add_argument(
        '--sr', type=int, default=22050,
        help='Sample rate (default: 22050)',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print detailed analysis info',
    )

    args = parser.parse_args()

    if not args.audio_file and not args.stems:
        parser.error('Either audio_file or --stems must be provided')

    # Auto-detect title from filename
    title = args.title
    artist = args.artist or 'Unknown'
    if title is None and args.audio_file:
        title = os.path.splitext(os.path.basename(args.audio_file))[0]
    elif title is None:
        title = 'Unknown'

    # Import pipeline
    from api.chord_detection.pipeline import ChordDetectionPipeline

    pipeline = ChordDetectionPipeline(
        sr=args.sr,
        hop_length=2048,
        segment_duration=args.segment_duration,
        min_chord_duration=args.min_duration,
        smoothing=args.smoothing,
        enhanced_chroma=args.enhanced,
    )

    start_time = time.time()

    if args.stems:
        # Analyze pre-separated stems
        stem_paths = parse_stems_arg(args.stems)
        print(f'Analyzing stems: {", ".join(f"{k}={v}" for k, v in stem_paths.items())}')
        result = pipeline.analyze_stems(stem_paths, title=title, artist=artist)

    elif args.separate and args.audio_file:
        # Phase 1: Separate, then analyze stems
        stems = run_separation(args.audio_file, args.separator)
        if stems:
            result = pipeline.analyze_stems(stems, title=title, artist=artist)
        else:
            # Fallback to full mix
            print('Analyzing full mix (no separation)...')
            result = pipeline.analyze_file(args.audio_file, title=title, artist=artist)

    else:
        # Analyze full mix directly
        print(f'Analyzing: {args.audio_file}')
        result = pipeline.analyze_file(args.audio_file, title=title, artist=artist)

    elapsed = time.time() - start_time

    # Print analysis summary
    print(f'\n{"=" * 50}')
    print(f'Key: {result["key"]} (confidence: {result["key_confidence"]:.2f})')
    print(f'Chords detected: {len(result["chords_simplified"])}')
    print(f'Analysis time: {elapsed:.1f}s')
    print(f'{"=" * 50}\n')

    if args.verbose:
        print('Top 5 key candidates:')
        for key_name, score in result['key_scores_top5']:
            print(f'  {key_name:15s} {score:.3f}')
        print()

        # Show correction stats
        smoothed = result['chords_smoothed']
        corrections = sum(1 for c in smoothed if c.get('was_corrected', False))
        if corrections > 0:
            print(f'Transition matrix corrected {corrections}/{len(smoothed)} frames')
        print()

    # Generate output
    if args.format == 'all':
        formats = ['simple', 'markdown', 'csv', 'json']
    else:
        formats = [args.format]

    for fmt in formats:
        chart = pipeline.generate_chart(result, format=fmt)

        if args.output and len(formats) == 1:
            with open(args.output, 'w') as f:
                f.write(chart)
            print(f'Chart written to: {args.output}')
        elif args.output and len(formats) > 1:
            base, ext = os.path.splitext(args.output)
            fmt_ext = {'csv': '.csv', 'markdown': '.md', 'json': '.json', 'simple': '.txt'}
            out_path = base + fmt_ext.get(fmt, f'.{fmt}')
            with open(out_path, 'w') as f:
                f.write(chart)
            print(f'Chart written to: {out_path}')
        else:
            if len(formats) > 1:
                print(f'\n--- {fmt.upper()} ---')
            print(chart)


if __name__ == '__main__':
    main()
