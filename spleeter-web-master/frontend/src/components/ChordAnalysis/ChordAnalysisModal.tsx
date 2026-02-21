import axios from 'axios';
import * as React from 'react';
import { Alert, Badge, Button, Form, Modal, Spinner, Table } from 'react-bootstrap';
import { ChordAnalysis, ChordEntry } from '../../models/ChordAnalysis';
import { SongData } from '../../models/SongData';
import './ChordAnalysis.css';

interface Props {
  song: SongData | null;
  show: boolean;
  onHide: () => void;
}

interface State {
  analyses: ChordAnalysis[];
  currentAnalysis: ChordAnalysis | null;
  isLoading: boolean;
  isSubmitting: boolean;
  error: string;
  segmentDuration: number;
  smoothing: number;
  useDynamicMix: boolean;
  selectedDynamicMixId: string;
}

/**
 * Modal component for chord analysis.
 * Allows creating new analyses and viewing results.
 */
class ChordAnalysisModal extends React.Component<Props, State> {
  pollInterval: ReturnType<typeof setInterval> | null = null;

  constructor(props: Props) {
    super(props);
    this.state = {
      analyses: [],
      currentAnalysis: null,
      isLoading: false,
      isSubmitting: false,
      error: '',
      segmentDuration: 0.5,
      smoothing: 0.6,
      useDynamicMix: false,
      selectedDynamicMixId: '',
    };
  }

  componentDidUpdate(prevProps: Props): void {
    if (this.props.show && !prevProps.show && this.props.song) {
      this.fetchAnalyses();
    }
    if (!this.props.show && prevProps.show) {
      this.stopPolling();
    }
  }

  componentWillUnmount(): void {
    this.stopPolling();
  }

  fetchAnalyses = (): void => {
    const { song } = this.props;
    if (!song) return;

    this.setState({ isLoading: true });
    axios
      .get(`/api/chord-analysis/track/${song.id}/`)
      .then(({ data }) => {
        this.setState({
          analyses: data,
          isLoading: false,
          currentAnalysis: data.length > 0 ? data[0] : null,
        });
        // If any analysis is in progress, start polling
        if (data.some((a: ChordAnalysis) => a.status === 'In Progress' || a.status === 'Queued')) {
          this.startPolling();
        }
      })
      .catch(() => {
        this.setState({ isLoading: false, error: 'Failed to fetch analyses.' });
      });
  };

  startPolling = (): void => {
    if (this.pollInterval) return;
    this.pollInterval = setInterval(() => {
      this.fetchAnalyses();
    }, 3000);
  };

  stopPolling = (): void => {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  };

  handleSubmit = (): void => {
    const { song } = this.props;
    const { segmentDuration, smoothing, useDynamicMix, selectedDynamicMixId } = this.state;
    if (!song) return;

    this.setState({ isSubmitting: true, error: '' });

    const data: Record<string, unknown> = {
      source_track: song.id,
      segment_duration: segmentDuration,
      smoothing: smoothing,
    };

    if (useDynamicMix && selectedDynamicMixId) {
      data.dynamic_mix = selectedDynamicMixId;
    }

    axios
      .post('/api/chord-analysis/', data)
      .then(() => {
        this.setState({ isSubmitting: false });
        this.fetchAnalyses();
        this.startPolling();
      })
      .catch(({ response }) => {
        const errors = response?.data?.errors;
        this.setState({
          isSubmitting: false,
          error: errors ? JSON.stringify(errors) : 'Failed to start analysis.',
        });
      });
  };

  handleExportCSV = (): void => {
    const { currentAnalysis } = this.state;
    if (!currentAnalysis?.chart_csv) return;

    const blob = new Blob([currentAnalysis.chart_csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chord_chart_${currentAnalysis.key.replace(' ', '_')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  formatTime = (seconds: number): string => {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec.toString().padStart(2, '0')}`;
  };

  renderChordTable = (chords: ChordEntry[]): JSX.Element => {
    return (
      <Table striped bordered hover size="sm" className="chord-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Chord</th>
            <th>Confidence</th>
            <th>In Key</th>
          </tr>
        </thead>
        <tbody>
          {chords.map((chord, idx) => (
            <tr key={idx} className={chord.in_key ? '' : 'table-warning'}>
              <td>{this.formatTime(chord.time)}</td>
              <td>
                <strong>{chord.chord}</strong>
                {!chord.in_key && chord.suggestion && (
                  <span className="text-muted ml-2">({chord.suggestion}?)</span>
                )}
              </td>
              <td>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${chord.confidence * 100}%` }}
                  />
                  <span className="confidence-text">{(chord.confidence * 100).toFixed(0)}%</span>
                </div>
              </td>
              <td>
                {chord.in_key ? (
                  <Badge variant="success">Yes</Badge>
                ) : (
                  <Badge variant="warning">Flag</Badge>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    );
  };

  renderAnalysisResult = (): JSX.Element | null => {
    const { currentAnalysis } = this.state;
    if (!currentAnalysis) return null;

    if (currentAnalysis.status === 'Queued' || currentAnalysis.status === 'In Progress') {
      return (
        <div className="text-center p-4">
          <Spinner animation="border" variant="primary" />
          <p className="mt-2">Analyzing chords...</p>
        </div>
      );
    }

    if (currentAnalysis.status === 'Error') {
      return <Alert variant="danger">Analysis failed: {currentAnalysis.error}</Alert>;
    }

    const result = currentAnalysis.result_json;
    if (!result || !result.chords) {
      return <Alert variant="info">No chord data available.</Alert>;
    }

    return (
      <div>
        <div className="analysis-header mb-3">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h5 className="mb-1">
                Key: <Badge variant="info">{currentAnalysis.key}</Badge>
                <small className="text-muted ml-2">
                  (confidence: {(currentAnalysis.key_confidence * 100).toFixed(0)}%)
                </small>
              </h5>
              <small className="text-muted">
                {result.chords.length} chord changes detected
              </small>
            </div>
            <Button variant="outline-secondary" size="sm" onClick={this.handleExportCSV}>
              Export CSV
            </Button>
          </div>
        </div>
        {this.renderChordTable(result.chords)}
      </div>
    );
  };

  render(): JSX.Element {
    const { show, onHide, song } = this.props;
    const {
      isLoading,
      isSubmitting,
      error,
      segmentDuration,
      smoothing,
      useDynamicMix,
      selectedDynamicMixId,
      analyses,
    } = this.state;

    const completedDynamicMixes = song?.dynamic?.filter(d => d.status === 'Done') || [];

    return (
      <Modal show={show} onHide={onHide} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>
            Chord Analysis {song && `- ${song.artist} - ${song.title}`}
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {error && <Alert variant="danger">{error}</Alert>}

          {/* New Analysis Form */}
          <div className="mb-4">
            <h6>Run New Analysis</h6>
            <Form>
              <Form.Group className="mb-2">
                <Form.Label>Segment Duration (seconds)</Form.Label>
                <Form.Control
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="2.0"
                  value={segmentDuration}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    this.setState({ segmentDuration: parseFloat(e.target.value) || 0.5 })
                  }
                />
                <Form.Text className="text-muted">
                  Smaller = more chord changes, larger = more stable
                </Form.Text>
              </Form.Group>
              <Form.Group className="mb-2">
                <Form.Label>Theory Smoothing ({(smoothing * 100).toFixed(0)}%)</Form.Label>
                <Form.Control
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={smoothing}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    this.setState({ smoothing: parseFloat(e.target.value) })
                  }
                />
                <Form.Text className="text-muted">
                  Low = trust raw detection, High = trust music theory
                </Form.Text>
              </Form.Group>
              {completedDynamicMixes.length > 0 && (
                <Form.Group className="mb-2">
                  <Form.Check
                    type="checkbox"
                    label="Use separated stems (recommended for accuracy)"
                    checked={useDynamicMix}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      this.setState({
                        useDynamicMix: e.target.checked,
                        selectedDynamicMixId: e.target.checked
                          ? completedDynamicMixes[0]?.id || ''
                          : '',
                      })
                    }
                  />
                  {useDynamicMix && (
                    <Form.Control
                      as="select"
                      className="mt-1"
                      value={selectedDynamicMixId}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                        this.setState({ selectedDynamicMixId: e.target.value })
                      }>
                      {completedDynamicMixes.map(mix => (
                        <option key={mix.id} value={mix.id}>
                          {mix.separator} - {mix.extra_info?.join(', ')}
                        </option>
                      ))}
                    </Form.Control>
                  )}
                </Form.Group>
              )}
              <Button
                variant="primary"
                disabled={isSubmitting || !song?.url}
                onClick={this.handleSubmit}>
                {isSubmitting ? (
                  <Spinner animation="border" size="sm" className="mr-1" />
                ) : null}
                Analyze Chords
              </Button>
            </Form>
          </div>

          <hr />

          {/* Results */}
          {isLoading ? (
            <div className="text-center p-3">
              <Spinner animation="border" />
            </div>
          ) : analyses.length > 0 ? (
            <div>
              <h6>Results</h6>
              {this.renderAnalysisResult()}
            </div>
          ) : (
            <p className="text-muted">No chord analyses yet. Run an analysis above.</p>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={onHide}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    );
  }
}

export default ChordAnalysisModal;
