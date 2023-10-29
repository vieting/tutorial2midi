"""
Main entry point for tutorial2midi
"""
import argparse
import pandas as pd

from tutorial2midi import Video, MIDIWrapper, get_notes_from_video


def main(video_filename: str, midi_filename: str, tempo: 100):
    """
    Main function for conversion.
    """
    if midi_filename is None:
        midi_filename = video_filename.replace(".mp4", ".midi")

    video = Video(video_filename)
    notes = get_notes_from_video(video, tempo)

    midi = MIDIWrapper(tempo)
    midi.add_notes(notes)
    midi.write_to_file(midi_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video_filename", help="filename of mp4 video")
    parser.add_argument("--midi_filename", help="filename of mp4 video", default=None)
    parser.add_argument("--tempo", help="tempo in bpm", type=int, default=100)

    args = parser.parse_args()
    main(**vars(args))
