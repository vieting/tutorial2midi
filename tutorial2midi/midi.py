"""
MIDI helpers.
"""
import pandas as pd
import tqdm
from midiutil import MIDIFile


class MIDIWrapper:
    """
    Wrapper class for MIDI.
    """
    def __init__(self, tempo: int = 100):
        """
        :param tempo: in bpm
        """
        self._midi = MIDIFile(2)
        for track in [0, 1]:
            self._midi.addTempo(track, 0.0, tempo)

    def add_notes(self, notes: pd.DataFrame, volume: int = 100):
        """
        Add notes from dataframe to MIDI file.

        volume in 0-127, as per the MIDI standard
        """
        hand_mapping = {"right": 0, "left": 1}
        for index, note in tqdm.tqdm(notes.iterrows(), desc="Add notes", unit="notes"):
            self._midi.addNote(
                0,
                hand_mapping[note.hand],
                note.key,
                note.start,
                note.duration,
                volume,
            )

    def write_to_file(self, filename: str):
        """
        Write to MIDI file.
        """
        assert filename.endswith(".midi"), "Filename should end on .midi"
        with open(filename, "wb") as output_file:
            self._midi.writeFile(output_file)
        print(f"Wrote MIDI to {filename}")
