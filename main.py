"""
Main entry point for tutorial2midi
"""
import argparse
import pandas as pd

from tutorial2midi import Video, MIDIWrapper


def main(video_filename: str, midi_filename: str = None):
    """
    Main function for conversion.
    """
    if midi_filename is None:
        midi_filename = video_filename.replace(".mp4", ".midi")

    video = Video(video_filename)

    midi = MIDIWrapper()
    tmp = pd.DataFrame(
        [
            {"left_hand": True, "key_piano_pos": 50, "key_start": 0.0, "key_duration": 1.0},
            {"left_hand": True, "key_piano_pos": 51, "key_start": 1.0, "key_duration": 1.0},
            {"left_hand": True, "key_piano_pos": 52, "key_start": 2.0, "key_duration": 1.0},
            {"left_hand": True, "key_piano_pos": 53, "key_start": 3.0, "key_duration": 1.0},
            {"left_hand": True, "key_piano_pos": 54, "key_start": 4.0, "key_duration": 4.0},
        ],
        list(range(5))
    )
    midi.add_keys(tmp)
    midi.write_to_file(midi_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video_filename", help="filename of mp4 video")
    parser.add_argument("--midi_filename", help="filename of mp4 video", default=None)

    args = parser.parse_args()
    main(**vars(args))
