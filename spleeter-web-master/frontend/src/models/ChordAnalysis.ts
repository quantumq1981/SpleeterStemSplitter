import { TaskStatus } from './TaskStatus';

/**
 * Represents a single detected chord in the analysis.
 */
export interface ChordEntry {
  time: number;
  end_time: number;
  chord: string;
  confidence: number;
  in_key: boolean;
  suggestion: string | null;
}

/**
 * Represents a ChordAnalysis result from the backend.
 */
export interface ChordAnalysis {
  id: string;
  celery_id: string | null;
  source_track: string;
  dynamic_mix: string | null;
  key: string;
  key_confidence: number;
  result_json: {
    key: string;
    key_confidence: number;
    key_scores_top5: [string, number][];
    chords: ChordEntry[];
    metadata: Record<string, unknown>;
  };
  chart_markdown: string;
  chart_csv: string;
  segment_duration: number;
  smoothing: number;
  status: TaskStatus;
  error: string;
  date_created: string;
  date_finished: string | null;
}
