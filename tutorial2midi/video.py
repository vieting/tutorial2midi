"""
Video helpers.
"""
import imageio
import numpy as np
import tqdm


class Video:
    """
    Wrapper class for video.
    """
    def __init__(self, filename: str):
        self.frame_rate = 0.0
        self._data = self.read_frames(filename)

    def read_frames(self, filename: str) -> np.ndarray:
        """
        Read image frames from video.
        """
        reader = imageio.get_reader(filename, "ffmpeg")
        self.frame_rate = reader.get_meta_data()["fps"]
        frames = np.expand_dims(reader.get_data(0), -1)
        for idx in tqdm.tqdm(range(1, reader.count_frames()), desc="Read frames", unit="frames"):
            frames = np.concatenate([frames, np.expand_dims(reader.get_data(idx), -1)], axis=-1)
        return frames

    def get_keyboard_stripe(self) -> np.ndarray:
        """
        Cut vertical stripe from video that represents the keyboard and return frames.
        """
        return self._data[500, :, :]  # TODO: hard-coded

    def get_pianoroll_stripe(self) -> np.ndarray:
        """
        Cut vertical stripe from video that represents the pianoroll and return frames.
        """
        return self._data[480, :, :]  # TODO: hard-coded
