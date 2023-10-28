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
        self._data = self.read_frames(filename)

    @staticmethod
    def read_frames(filename: str) -> np.ndarray:
        """
        Read image frames from video.
        """
        reader = imageio.get_reader(filename, "ffmpeg")
        frames = np.expand_dims(reader.get_data(0), -1)
        for idx in tqdm.tqdm(range(1, reader.count_frames()), desc="Read frames", unit="frames"):
            frames = np.concatenate([frames, np.expand_dims(reader.get_data(idx), -1)], axis=-1)
        return frames
