import * as React from 'react';
import { Button } from 'react-bootstrap';
import { MusicNoteBeamed } from 'react-bootstrap-icons';
import { SongData } from '../../../models/SongData';

interface Props {
  song: SongData;
  disabled: boolean;
  onClick: (song: SongData) => void;
}

/**
 * Button to trigger chord analysis for a song.
 */
class ChordAnalysisButton extends React.Component<Props> {
  handleClick = (): void => {
    this.props.onClick(this.props.song);
  };

  render(): JSX.Element {
    const { disabled } = this.props;
    return (
      <Button
        className="ml-1"
        variant="outline-info"
        size="sm"
        disabled={disabled}
        title="Chord Analysis"
        onClick={this.handleClick}>
        <MusicNoteBeamed className="align-middle" size={16} />
        <span className="align-middle ml-1">Chords</span>
      </Button>
    );
  }
}

export default ChordAnalysisButton;
