"""
Main entry point for tutorial2midi
"""
import argparse
import pandas as pd

from tutorial2midi import MIDIWrapper


def main(filename: str):
    """
    Main function for conversion.
    """
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
    midi.write_to_file(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", help="filename of mp4 video")

    args = parser.parse_args()
    main(**vars(args))
