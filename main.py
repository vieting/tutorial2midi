"""
Main entry point for tutorial2midi
"""
import argparse

from tutorial2midi import Video, MIDIWrapper, get_notes_from_video


def main(video_filename: str, midi_filename: str, tempo: int, **kwargs):
    """
    Main function for conversion.
    """
    if midi_filename is None:
        midi_filename = video_filename.replace(".mp4", ".midi")

    video = Video(video_filename)
    notes = get_notes_from_video(video, tempo, **kwargs)

    midi = MIDIWrapper(tempo)
    midi.add_notes(notes)
    midi.write_to_file(midi_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video_filename", help="filename of mp4 video")
    parser.add_argument("--midi_filename", help="filename of mp4 video", default=None)
    parser.add_argument("--tempo", help="tempo in bpm", type=int, default=100)
    parser.add_argument("--key_offset", help="key offset, used to transpose (default 21)", type=int, default=21)
    parser.add_argument("--right_hand_boundary", help="lowest key for right hand", type=int, default=60)
    parser.add_argument("--quantization", help="quantization, e.g. 16 for 1/16 notes", type=int, default=16)
    parser.add_argument("--anacrusis", help="anacrusis in quarter notes", type=float, default=0.0)

    args = parser.parse_args()
    main(**vars(args))
