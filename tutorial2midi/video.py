"""
Video helpers.
"""
import cv2
import imageio
import numpy as np
import tqdm
from typing import Optional


class Video:
    """
    Wrapper class for video.
    """
    def __init__(self, filename: str):
        self.frame_rate = 0.0
        self._data = self.read_frames(filename)  # shape (height, width, RGB channels, frames)
        self._keyboard_idx = 500  # TODO: hard-coded
        self._pianoroll_idx = 480  # TODO: hard-coded

    def read_frames(self, filename: str) -> np.ndarray:
        """
        Read image frames from video.
        """
        reader = imageio.get_reader(filename, "ffmpeg")
        self.frame_rate = reader.get_meta_data()["fps"]
        frames = np.expand_dims(reader.get_data(0), -1)
        for idx in tqdm.tqdm(range(1, reader.count_frames()), desc="Read frames", unit="frames"):
            frames = np.concatenate([frames, np.expand_dims(reader.get_data(idx)[:, :, ::-1], -1)], axis=-1)
        return frames

    def get_keyboard_stripe(self) -> np.ndarray:
        """
        Cut vertical stripe from video that represents the keyboard and return frames.
        """
        return self._data[self._keyboard_idx, :, :]

    def get_pianoroll_stripe(self) -> np.ndarray:
        """
        Cut vertical stripe from video that represents the pianoroll and return frames.
        """
        return self._data[self._pianoroll_idx, :, :]

    def visualize_video(self, filename: str, frame: Optional[int] = None):
        """
        Visualize video content to check keyboard and pianoroll stripe.
        """
        if frame is None:
            frame = self._data.shape[-1] // 2
        image = self._data[..., frame].copy()
        image[self._keyboard_idx, :, -1] = 255  # red
        image[self._pianoroll_idx, :, -1] = 255  # red
        cv2.imwrite(filename, image)
