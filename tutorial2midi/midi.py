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

    def add_keys(self, keys: pd.DataFrame, channel: int = 0, volume: int = 100):
        """
        Add keys from dataframe to MIDI file.

        volume in 0-127, as per the MIDI standard
        """
        for index, k in tqdm.tqdm(keys.iterrows(), desc="Add notes", unit="notes"):
            self._midi.addNote(
                int(k.left_hand),
                channel,
                int(k.key_piano_pos),
                k.key_start,
                k.key_duration,
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
