"""
Functions to extract notes.
"""
import cv2
import numpy as np
import pandas as pd

from .video import Video


class Keyboard:
    """
    Wrapper class for actions regarding the keyboard in the video.
    """
    def __init__(self, ref_stripe: np.ndarray):
        self._ref_stripe = ref_stripe
        self._keys = {}
        self.init_keys()

    def init_keys(self):
        """
        Find the borders for each key of the keyboard.
        """
        stripe = self._ref_stripe.mean(axis=-1)
        deltas = np.abs(np.diff(stripe))
        borders = deltas > 2 * np.mean(deltas)
        key = 0
        border = 0
        for idx in range(borders.size - 1):
            if borders[idx] and any(borders[idx + 1:idx + 3]):  # remove double borders
                borders[idx] = False
            if borders[idx]:
                self._keys[key] = (border, idx)
                key += 1
                border = idx + 1
        self._keys[key] = (border, borders.size)

    def visualize_ref_keyboard_with_borders(self, filename: str, pianoroll: np.ndarray):
        """
        Helper to plot the keyboard along with the detected borders to help debugging.
        """
        keys_image = np.repeat(self._ref_stripe[None, ...], 100, axis=0)
        border_image = np.zeros(keys_image.shape)
        for key in self._keys.values():
            border_image[:, key[0], :] = 255
        image = np.concatenate([keys_image, border_image], axis=0)
        if pianoroll is not None:
            image = np.concatenate([image, pianoroll], axis=0)
        cv2.imwrite(filename, image)

    def get_active_keys(self, pianoroll: np.ndarray) -> np.ndarray:
        """
        Detect areas of activity per key.
        """
        active_keys = np.zeros((pianoroll.shape[-1], len(self._keys)), bool)
        for key, borders in self._keys.items():
            active_keys[:, key] = pianoroll[borders[0]:borders[1], ...].any(axis=0)
        # keys_image = np.repeat(active_keys[..., None], 3, axis=-1) * 255
        # cv2.imwrite("images/active_keys.png", keys_image)
        return active_keys


def get_range_borders(array: np.ndarray) -> list:
    """
    Find indices of borders of ranges in array. E.g. [0, 0, 1, 1, 1, 0, 1, 1] -> [(2, 4), (6, 7)]
    """
    assert len(array.shape) == 1
    idx = 0
    borders = []
    while idx < array.size:
        while idx < array.size and not array[idx]:
            idx += 1
        if idx == array.size:
            break
        left = idx
        while idx < array.size and array[idx]:
            idx += 1
        right = idx
        borders.append((left, right))
        idx += 1
    return borders


def process_pianoroll(pianoroll: np.ndarray) -> np.ndarray:
    """
    Process pianoroll video

    Return shape (width, #frames)
    """
    pianoroll = pianoroll.mean(axis=1)
    threshold = pianoroll.mean() * 2
    pianoroll = pianoroll > threshold

    # morphological opening
    pianoroll = cv2.morphologyEx(pianoroll.astype("uint8"), cv2.MORPH_OPEN, np.ones((5,5)))

    # reduce each note to center
    for frame in range(pianoroll.shape[1]):
        idx = 0
        while idx < pianoroll.shape[0]:
            while idx < pianoroll.shape[0] and not pianoroll[idx, frame]:
                idx += 1
            left = idx
            while idx < pianoroll.shape[0] and pianoroll[idx, frame]:
                idx += 1
            right = idx
            if right > left:
                pianoroll[left:right, frame] = False
                pianoroll[left + (right - left) // 2, frame] = True
            idx += 1
    return pianoroll


def get_notes_from_video(video: Video, tempo: int) -> pd.DataFrame:
    """
    Extract notes from the video.
    """
    notes = pd.DataFrame(columns=["key", "start", "duration", "hand"])
    keyboard_video = video.get_keyboard_stripe()
    pianoroll_video = video.get_pianoroll_stripe()
    keyboard = Keyboard(keyboard_video.mean(axis=-1))
    # keyboard.visualize_ref_keyboard_with_borders(
    #     "images/keyboard.png", pianoroll_video.transpose(2, 0, 1)[::-1, :, :])

    pianoroll_video = process_pianoroll(pianoroll_video)
    active_keys = keyboard.get_active_keys(pianoroll_video)
    notes = []
    for key in range(active_keys.shape[1]):
        for borders in get_range_borders(active_keys[:, key]):
            notes.append({"hand": "right", "key": key + 22, "start": borders[0], "duration": borders[1] - borders[0]})
    notes = pd.DataFrame(notes, list(range(len(notes))))

    # post process notes
    notes = notes.sort_values("start")
    notes.start -= notes.start.min()
    notes.start = notes.start / video.frame_rate * tempo / 60
    notes.duration = notes.duration / video.frame_rate * tempo / 60
    # import ipdb
    # ipdb.set_trace()
    return notes
