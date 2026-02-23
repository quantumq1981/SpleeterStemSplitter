import os
import os.path
import pathlib
import shutil
from typing import Dict, List
import traceback

from billiard.context import Process
from billiard.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

from .celery import app
from .models import (DEMUCS_FAMILY, D3NET, SPLEETER, SPLEETER_PIANO, XUMX, BS_ROFORMER,
                     BS_ROFORMER_5S_GUITAR, BS_ROFORMER_5S_PIANO, BS_ROFORMER_6S,
                     BS_ROFORMER_FAMILY,
                     ChordAnalysis, DynamicMix, SourceFile, StaticMix, TaskStatus,
                     YTAudioDownloadTask)
# Lazy-import separators: they require torch/tensorflow which may not be
# installed in lightweight deployments (e.g. Fly.io).
DemucsSeparator = None
SpleeterSeparator = None
BSRoformerSeparator = None

def _load_separators():
    global DemucsSeparator, SpleeterSeparator, BSRoformerSeparator
    if DemucsSeparator is None:
        try:
            from .separators.demucs_separator import DemucsSeparator as _D
            DemucsSeparator = _D
        except ImportError:
            pass
    if SpleeterSeparator is None:
        try:
            from .separators.spleeter_separator import SpleeterSeparator as _S
            SpleeterSeparator = _S
        except ImportError:
            pass
    if BSRoformerSeparator is None:
        try:
            from .separators.bs_roformer_separator import BSRoformerSeparator as _B
            BSRoformerSeparator = _B
        except ImportError:
            pass
from .util import ALL_PARTS, ALL_PARTS_5_PIANO, ALL_PARTS_5_GUITAR, ALL_PARTS_6, output_format_to_ext, get_valid_filename
from .youtubedl import download_audio, get_file_ext

"""
This module defines various Celery tasks used for Spleeter Web.
"""

LEGACY_SEPARATORS = {D3NET, XUMX}


def get_separator(separator: str, separator_args: Dict, bitrate: int,
                  cpu_separation: bool):
    """Returns separator object for corresponding source separation model."""
    _load_separators()
    if separator in LEGACY_SEPARATORS:
        raise ValueError(
            f'{separator} is no longer supported for new separations.')
    if separator == SPLEETER:
        return SpleeterSeparator(cpu_separation, bitrate, False)
    if separator == SPLEETER_PIANO:
        return SpleeterSeparator(cpu_separation, bitrate, True)
    if separator in BS_ROFORMER_FAMILY:
        # Map separator to stem_mode
        stem_mode_map = {
            BS_ROFORMER: '4stem',
            BS_ROFORMER_5S_GUITAR: '5stem_guitar',
            BS_ROFORMER_5S_PIANO: '5stem_piano',
            BS_ROFORMER_6S: '6stem',
        }
        stem_mode = stem_mode_map.get(separator, '4stem')
        return BSRoformerSeparator(cpu_separation=cpu_separation, output_format=bitrate, stem_mode=stem_mode)
    if separator in DEMUCS_FAMILY:
        random_shifts = separator_args.get('random_shifts', 0)
        return DemucsSeparator(separator, cpu_separation, bitrate,
                               random_shifts)
    raise ValueError(f'Unknown separator "{separator}".')

@app.task()
def create_static_mix(static_mix_id):
    """
    Task to create static mix and write to appropriate storage backend.
    :param static_mix_id: The id of the StaticMix to be processed
    """
    # Mark as in progress
    try:
        static_mix = StaticMix.objects.get(id=static_mix_id)
    except StaticMix.DoesNotExist:
        # Does not exist, perhaps due to stale task
        print('StaticMix does not exist')
        return
    static_mix.status = TaskStatus.IN_PROGRESS
    static_mix.save()

    ext = output_format_to_ext(static_mix.bitrate)

    try:
        # Get paths
        directory = os.path.join(settings.MEDIA_ROOT, settings.SEPARATE_DIR,
                                 static_mix_id)
        filename = get_valid_filename(static_mix.formatted_name()) + f'.{ext}'
        rel_media_path = os.path.join(settings.SEPARATE_DIR, static_mix_id,
                                      filename)
        rel_path = os.path.join(settings.MEDIA_ROOT, rel_media_path)
        rel_path_dir = os.path.join(settings.MEDIA_ROOT, settings.SEPARATE_DIR,
                                    static_mix_id)

        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        try:
            separator = get_separator(static_mix.separator,
                                      static_mix.separator_args,
                                      static_mix.bitrate,
                                      settings.CPU_SEPARATION)
        except ValueError as exc:
            static_mix.status = TaskStatus.ERROR
            static_mix.error = str(exc)
            static_mix.date_finished = timezone.now()
            static_mix.save()
            return

        parts = {
            'vocals': static_mix.vocals,
            'drums': static_mix.drums,
            'bass': static_mix.bass,
            'other': static_mix.other
        }
        if static_mix.separator == SPLEETER_PIANO:
            parts['piano'] = static_mix.piano
        elif static_mix.separator == BS_ROFORMER_5S_GUITAR:
            parts['guitar'] = static_mix.guitar
        elif static_mix.separator == BS_ROFORMER_5S_PIANO:
            parts['piano'] = static_mix.piano
        elif static_mix.separator == BS_ROFORMER_6S:
            parts['guitar'] = static_mix.guitar
            parts['piano'] = static_mix.piano

        # Non-local filesystems like S3/Azure Blob do not support source_path()
        is_local = settings.DEFAULT_FILE_STORAGE == 'api.storage.FileSystemStorage'
        path = static_mix.source_path() if is_local else static_mix.source_url(
        )

        if not settings.CPU_SEPARATION:
            # For GPU separation, do separation in separate process.
            # Otherwise, GPU memory is not automatically freed afterwards
            process_eval = Process(target=separator.create_static_mix,
                                   args=(parts, path, rel_path))
            process_eval.start()
            try:
                process_eval.join()
            except SoftTimeLimitExceeded as e:
                # Kill process if user aborts task
                process_eval.terminate()
                raise e
        else:
            separator.create_static_mix(parts, path, rel_path)

        # Check file exists
        if os.path.exists(rel_path):
            static_mix.status = TaskStatus.DONE
            static_mix.date_finished = timezone.now()
            if is_local:
                # File is already on local filesystem
                static_mix.file.name = rel_media_path
            else:
                # Need to copy local file to S3/Azure Blob/etc.
                raw_file = open(rel_path, 'rb')
                content_file = ContentFile(raw_file.read())
                content_file.name = filename
                static_mix.file = content_file
                # Remove local file
                os.remove(rel_path)
                # Remove empty directory
                os.rmdir(rel_path_dir)
            static_mix.save()
        else:
            raise Exception('Error writing to file')
    except FileNotFoundError as error:
        print(error)
        print('Please make sure you have FFmpeg and FFprobe installed.')
        static_mix.status = TaskStatus.ERROR
        static_mix.date_finished = timezone.now()
        static_mix.error = str(error)
        static_mix.save()
    except SoftTimeLimitExceeded:
        print('Aborted!')
    except Exception as error:
        print(traceback.format_exc())
        static_mix.status = TaskStatus.ERROR
        static_mix.date_finished = timezone.now()
        static_mix.error = str(error)
        static_mix.save()

@app.task()
def create_dynamic_mix(dynamic_mix_id):
    """
    Task to create dynamic mix and write to appropriate storage backend.
    :param dynamic_mix_id: The id of the audio track model (StaticMix) to be processed
    """
    # Mark as in progress
    try:
        dynamic_mix = DynamicMix.objects.get(id=dynamic_mix_id)
    except DynamicMix.DoesNotExist:
        # Does not exist, perhaps due to stale task
        print('DynamicMix does not exist')
        return
    dynamic_mix.status = TaskStatus.IN_PROGRESS
    dynamic_mix.save()

    try:
        # Get paths
        directory = os.path.join(settings.MEDIA_ROOT, settings.SEPARATE_DIR,
                                 dynamic_mix_id)
        rel_media_path = os.path.join(settings.SEPARATE_DIR, dynamic_mix_id)
        file_prefix = get_valid_filename(dynamic_mix.formatted_prefix())
        file_suffix = dynamic_mix.formatted_suffix()
        rel_path = os.path.join(settings.MEDIA_ROOT, rel_media_path)

        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        try:
            separator = get_separator(dynamic_mix.separator,
                                      dynamic_mix.separator_args,
                                      dynamic_mix.bitrate,
                                      settings.CPU_SEPARATION)
        except ValueError as exc:
            dynamic_mix.status = TaskStatus.ERROR
            dynamic_mix.error = str(exc)
            dynamic_mix.date_finished = timezone.now()
            dynamic_mix.save()
            return

        all_parts = ALL_PARTS_5_PIANO if dynamic_mix.separator == SPLEETER_PIANO else ALL_PARTS
        if dynamic_mix.separator == BS_ROFORMER_5S_GUITAR:
            all_parts = ALL_PARTS_5_GUITAR
        elif dynamic_mix.separator == BS_ROFORMER_5S_PIANO:
            all_parts = ALL_PARTS_5_PIANO
        elif dynamic_mix.separator == BS_ROFORMER_6S:
            all_parts = ALL_PARTS_6

        # Non-local filesystems like S3/Azure Blob do not support source_path()
        is_local = settings.DEFAULT_FILE_STORAGE == 'api.storage.FileSystemStorage'
        path = dynamic_mix.source_path(
        ) if is_local else dynamic_mix.source_url()

        # Do separation
        if not settings.CPU_SEPARATION:
            # For GPU separation, do separation in separate process.
            # Otherwise, GPU memory is not automatically freed afterwards
            process_eval = Process(target=separator.separate_into_parts,
                                   args=(path, rel_path))
            process_eval.start()
            try:
                process_eval.join()
            except SoftTimeLimitExceeded as e:
                # Kill process if user aborts task
                process_eval.terminate()
                raise e
        else:
            separator.separate_into_parts(path, rel_path)

        ext = output_format_to_ext(dynamic_mix.bitrate)
        # Check all parts exist
        if exists_all_parts(rel_path, ext, all_parts):
            rename_all_parts(rel_path, file_prefix, file_suffix, ext,
                             all_parts)
            dynamic_mix.status = TaskStatus.DONE
            dynamic_mix.date_finished = timezone.now()
            if is_local:
                save_to_local_storage(dynamic_mix, rel_media_path, file_prefix,
                                      file_suffix, ext, all_parts)
            else:
                save_to_ext_storage(dynamic_mix, rel_path, file_prefix,
                                    file_suffix, ext, all_parts)
        else:
            raise Exception('Error writing to file')
    except FileNotFoundError as error:
        print(traceback.format_exc())
        print('Please make sure you have FFmpeg and FFprobe installed.')
        dynamic_mix.status = TaskStatus.ERROR
        dynamic_mix.date_finished = timezone.now()
        dynamic_mix.error = str(error)
        dynamic_mix.save()
    except SoftTimeLimitExceeded:
        print('Aborted!')
    except Exception as error:
        print(traceback.format_exc())
        dynamic_mix.status = TaskStatus.ERROR
        dynamic_mix.date_finished = timezone.now()
        dynamic_mix.error = str(error)
        dynamic_mix.save()

@app.task(autoretry_for=(Exception, ),
          default_retry_delay=3,
          retry_kwargs={'max_retries': settings.YOUTUBE_MAX_RETRIES})
def fetch_youtube_audio(source_file_id, fetch_task_id, artist, title, link):
    """
    Task that uses youtubedl to extract the audio from a YouTube link.

    :param source_file_id: SourceFile id
    :param fetch_task_id: YouTube audio fetch task model id
    :param artist: Track artist
    :param title: Track title
    :param link: YouTube link
    """
    try:
        source_file = SourceFile.objects.get(id=source_file_id)
    except SourceFile.DoesNotExist:
        # Does not exist, perhaps due to stale task
        print('SourceFile does not exist')
        return
    fetch_task = YTAudioDownloadTask.objects.get(id=fetch_task_id)
    # Mark as in progress
    fetch_task.status = TaskStatus.IN_PROGRESS
    fetch_task.save()

    try:
        # Get paths
        directory = os.path.join(settings.MEDIA_ROOT, settings.UPLOAD_DIR,
                                 str(source_file_id))
        filename = get_valid_filename(artist + ' - ' +
                                      title) + get_file_ext(link)
        rel_media_path = os.path.join(settings.UPLOAD_DIR, str(source_file_id),
                                      filename)
        rel_path = os.path.join(settings.MEDIA_ROOT, rel_media_path)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        # Start download
        download_audio(link, rel_path)

        is_local = settings.DEFAULT_FILE_STORAGE == 'api.storage.FileSystemStorage'

        # Check file exists
        if os.path.exists(rel_path):
            fetch_task.status = TaskStatus.DONE
            fetch_task.date_finished = timezone.now()
            if is_local:
                # File is already on local filesystem
                source_file.file.name = rel_media_path
            else:
                # Need to copy local file to S3/Azure Blob/etc.
                raw_file = open(rel_path, 'rb')
                content_file = ContentFile(raw_file.read())
                content_file.name = filename
                source_file.file = content_file
                rel_dir_path = os.path.join(settings.MEDIA_ROOT,
                                            settings.UPLOAD_DIR,
                                            source_file_id)
                # Remove local file
                os.remove(rel_path)
                # Remove empty directory
                os.rmdir(rel_dir_path)
            fetch_task.save()
            source_file.save()
        else:
            raise Exception('Error writing to file')
    except SoftTimeLimitExceeded:
        print('Aborted!')
    except Exception as error:
        print(traceback.format_exc())
        fetch_task.status = TaskStatus.ERROR
        fetch_task.date_finished = timezone.now()
        fetch_task.error = str(error)
        fetch_task.save()
        raise error

def exists_all_parts(rel_path, ext, parts: List[str]):
    """Returns whether all of the individual component tracks exist on filesystem."""
    for part in parts:
        rel_part_path = os.path.join(rel_path, f'{part}.{ext}')
        if not os.path.exists(rel_part_path):
            print(f'{rel_part_path} does not exist')
            return False
    return True

def rename_all_parts(rel_path, file_prefix: str, file_suffix: str, ext: str, parts: List[str]):
    """Renames individual part files to names with track artist and title."""
    for part in parts:
        old_rel_path = os.path.join(rel_path, f'{part}.{ext}')
        new_rel_path = os.path.join(
            rel_path, f'{file_prefix} ({part}) {file_suffix}.{ext}')
        print(f'Renaming {old_rel_path} to {new_rel_path}')
        os.rename(old_rel_path, new_rel_path)

def save_to_local_storage(dynamic_mix,
                          rel_media_path,
                          file_prefix: str,
                          file_suffix: str,
                          ext: str,
                          parts=List[str]):
    """Saves individual parts to the local file system

    :param dynamic_mix: DynamicMix model
    :param rel_media_path: Relative path from media/ to DynamicMix ID directory
    :param file_prefix: Filename prefix
    """
    rel_media_path_vocals = os.path.join(
        rel_media_path, f'{file_prefix} (vocals) {file_suffix}.{ext}')
    rel_media_path_other = os.path.join(
        rel_media_path, f'{file_prefix} (other) {file_suffix}.{ext}')
    rel_media_path_bass = os.path.join(
        rel_media_path, f'{file_prefix} (bass) {file_suffix}.{ext}')
    rel_media_path_drums = os.path.join(
        rel_media_path, f'{file_prefix} (drums) {file_suffix}.{ext}')
    if 'piano' in parts:
        rel_media_path_piano = os.path.join(
            rel_media_path, f'{file_prefix} (piano) {file_suffix}.{ext}')
    if 'guitar' in parts:
        rel_media_path_guitar = os.path.join(
            rel_media_path, f'{file_prefix} (guitar) {file_suffix}.{ext}')

    # File is already on local filesystem
    dynamic_mix.vocals_file.name = rel_media_path_vocals
    dynamic_mix.other_file.name = rel_media_path_other
    dynamic_mix.bass_file.name = rel_media_path_bass
    dynamic_mix.drums_file.name = rel_media_path_drums
    if 'piano' in parts:
        dynamic_mix.piano_file.name = rel_media_path_piano
    if 'guitar' in parts:
        dynamic_mix.guitar_file.name = rel_media_path_guitar

    dynamic_mix.save()

def save_to_ext_storage(dynamic_mix, rel_path_dir, file_prefix: str,
                        file_suffix: str, ext: str, parts: List[str]):
    """Saves individual parts to external file storage (S3, Azure, etc.)

    :param dynamic_mix: DynamicMix model
    :param rel_path_dir: Relative path to DynamicMix ID directory
    :param file_prefix: Filename prefix
    """
    filenames = {
        part: f'{file_prefix} ({part}) {file_suffix}.{ext}'
        for part in parts
    }
    content_files = {}

    for part in parts:
        filename = filenames[part]
        rel_path = os.path.join(rel_path_dir, filename)
        raw_file = open(rel_path, 'rb')
        content_files[part] = ContentFile(raw_file.read())
        content_files[part].name = filename

    dynamic_mix.vocals_file = content_files['vocals']
    dynamic_mix.other_file = content_files['other']
    dynamic_mix.bass_file = content_files['bass']
    dynamic_mix.drums_file = content_files['drums']
    if 'piano' in parts:
        dynamic_mix.piano_file = content_files['piano']
    if 'guitar' in parts:
        dynamic_mix.guitar_file = content_files['guitar']

    dynamic_mix.save()

    shutil.rmtree(rel_path_dir, ignore_errors=True)


@app.task()
def analyze_chords(chord_analysis_id):
    """
    Celery task to run chord detection on a source track or dynamic mix.

    If a DynamicMix is associated, analyzes the separated stems (bass + other)
    for higher accuracy. Otherwise falls back to analyzing the full mix.

    :param chord_analysis_id: The id of the ChordAnalysis to be processed
    """
    try:
        analysis = ChordAnalysis.objects.get(id=chord_analysis_id)
    except ChordAnalysis.DoesNotExist:
        print('ChordAnalysis does not exist')
        return

    analysis.status = TaskStatus.IN_PROGRESS
    analysis.save()

    try:
        from .chord_detection.pipeline import ChordDetectionPipeline

        pipeline = ChordDetectionPipeline(
            segment_duration=analysis.segment_duration,
            smoothing=analysis.smoothing,
        )

        title = analysis.source_track.title
        artist = analysis.source_track.artist

        is_local = settings.DEFAULT_FILE_STORAGE == 'api.storage.FileSystemStorage'

        # Prefer analyzing separated stems if a DynamicMix is available
        if analysis.dynamic_mix and analysis.dynamic_mix.status == TaskStatus.DONE:
            stem_paths = {}
            mix = analysis.dynamic_mix
            stem_file_map = {
                'bass': mix.bass_file,
                'other': mix.other_file,
                'vocals': mix.vocals_file,
                'drums': mix.drums_file,
            }
            if mix.piano_file:
                stem_file_map['piano'] = mix.piano_file
            if mix.guitar_file:
                stem_file_map['guitar'] = mix.guitar_file

            for name, file_field in stem_file_map.items():
                if file_field and file_field.name:
                    if is_local:
                        stem_paths[name] = file_field.path
                    else:
                        stem_paths[name] = file_field.url

            if stem_paths:
                result = pipeline.analyze_stems(
                    stem_paths, title=title, artist=artist
                )
            else:
                # Fallback to full mix
                path = analysis.source_track.source_file.file.path if is_local else analysis.source_track.source_file.file.url
                result = pipeline.analyze_file(path, title=title, artist=artist)
        else:
            # Analyze full mix
            if is_local:
                path = analysis.source_track.source_file.file.path
            else:
                path = analysis.source_track.source_file.file.url
            result = pipeline.analyze_file(path, title=title, artist=artist)

        # Generate charts
        chart_md = pipeline.generate_chart(result, format='markdown')
        chart_csv = pipeline.generate_chart(result, format='csv')

        # Build JSON-serializable result (exclude raw numpy data)
        result_json = {
            'key': result['key'],
            'key_confidence': result['key_confidence'],
            'key_scores_top5': result['key_scores_top5'],
            'chords': [
                {
                    'time': c['time'],
                    'end_time': c.get('end_time', c['time']),
                    'chord': c['chord'],
                    'confidence': c.get('confidence', 0),
                    'in_key': c.get('in_key', True),
                    'suggestion': c.get('suggestion'),
                }
                for c in result['chords_flagged']
            ],
            'metadata': result['metadata'],
        }

        analysis.key = result['key']
        analysis.key_confidence = result['key_confidence']
        analysis.result_json = result_json
        analysis.chart_markdown = chart_md
        analysis.chart_csv = chart_csv
        analysis.status = TaskStatus.DONE
        analysis.date_finished = timezone.now()
        analysis.save()

    except Exception as error:
        print(traceback.format_exc())
        analysis.status = TaskStatus.ERROR
        analysis.date_finished = timezone.now()
        analysis.error = str(error)
        analysis.save()
