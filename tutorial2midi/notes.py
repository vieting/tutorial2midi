"""
Functions to extract notes.
"""
from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
from typing import Optional

from .midi import MIDIWrapper
from .video import Video


class Keyboard:
    """
    Wrapper class for actions regarding the keyboard in the video.
    """
    def __init__(self, ref_stripe: np.ndarray):
        self._ref_stripe = ref_stripe  # shape (width, rgb)
        self._key_borders = {}
        self._keys = np.zeros(ref_stripe.shape[0])  # shape (width,)
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
                self._key_borders[key] = (border, idx)
                key += 1
                border = idx + 1
        self._key_borders[key] = (border, borders.size)
        for key, borders in self._key_borders.items():
            self._keys[borders[0]:borders[1] + 1] = key

    def pixel_to_key(self, pixel: int) -> int:
        """
        Get key for given pixel.
        """
        return int(self._keys[pixel])

    def visualize_ref_keyboard_with_borders(
            self,
            filename: str,
            pianoroll: Optional[np.ndarray] = None,
            pianoroll_activity: Optional[np.ndarray] = None,
            active_keys: Optional[np.ndarray] = None,
    ):
        """
        Helper to plot the keyboard along with the detected borders to help debugging.
        """
        image = np.repeat(self._ref_stripe[None, ...], 100, axis=0)
        sep_stripe = np.ones((1, image.shape[1], 3)) * 255
        pianoroll_image = None
        if pianoroll is not None:
            if pianoroll.ndim == 2:
                pianoroll = np.repeat(pianoroll[:, None, :], 3, axis=1)
            pianoroll_image = pianoroll
            # cv2.imwrite(filename.replace(".png", "pianoroll.png"), pianoroll.transpose(2, 0, 1)[::-1, :, :])
        if pianoroll_activity is not None:
            if pianoroll_activity.ndim == 2:
                pianoroll_activity = np.repeat(pianoroll_activity[:, None, :], 3, axis=1)
            pianoroll_image[pianoroll_activity.astype(bool)] = 255
        if pianoroll_image is not None:
            image = np.concatenate([image, sep_stripe, pianoroll_image.transpose(2, 0, 1)[::-1, :, :]], axis=0)
        if active_keys is not None:
            assert pianoroll is not None, "Need pianoroll to print active keys on it"
            active_keys_image = pianoroll.transpose(2, 0, 1).copy()
            for frame in range(active_keys.shape[0]):
                for key in range(active_keys.shape[1]):
                    if active_keys[frame, key]:
                        active_keys_image[frame, self._key_borders[key][0]:self._key_borders[key][1], 1] = 255  # color green
            image = np.concatenate([image, sep_stripe, active_keys_image[::-1, :, :]], axis=0)
        for key in self._key_borders.values():  # red border lines
            image[:, key[0], -1] = 255
            image[:, key[1], -1] = 255
        cv2.imwrite(filename, image)


class Pianoroll:
    """
    Wrapper class for pianoroll in tutorial.
    """
    def __init__(self, pianoroll: np.ndarray):
        self.pianoroll = pianoroll.transpose(2, 0, 1)  # shape (frames, width, rgb)

    def get_notes_pixel(self) -> np.ndarray:
        """
        Get notes from the pianoroll. Notes are represented by their pixel position.
        """
        gray = cv2.cvtColor(self.pianoroll, cv2.COLOR_BGR2GRAY)  # (height, width, bgr)

        _, thresh = cv2.threshold(gray, 30, 255, 0)
        contours, _ = cv2.findContours(thresh, 1, 2)

        box_widths = []
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5:  # sort out very narrow boxes
                box_widths.append(w)

        white_key_width = int(np.argmax(np.bincount(box_widths)))  # most common value
        black_key_width = 0.55 * white_key_width  # approximation

        note_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if np.abs(w / white_key_width - 1) < 0.2:  # white key
                note_boxes.append((x, y, w, h))
            elif np.abs(w / black_key_width - 1) < 0.2:  # black key
                note_boxes.append((x, y, w, h))
            elif np.abs(w / (2 * white_key_width) - 1) % 1 < 0.2:  # probably two white keys next to each other
                area = gray.copy()[y:y + h, x:x + w]
                threshold = area.mean()
                offset = (w - 2 * white_key_width) // 2
                for key in range(2):
                    middle_idx = white_key_width // 2 + key * white_key_width + offset
                    key_activity = area[:, middle_idx] > threshold
                    key_activity[0] = False
                    key_activity[-1] = False
                    starts = np.flatnonzero(~key_activity[:-1] & key_activity[1:])
                    ends = np.flatnonzero(key_activity[:-1] & ~key_activity[1:])
                    for start, end in zip(starts, ends):
                        note_boxes.append((x + key * white_key_width + offset, y + start, white_key_width, end - start))

        img_marked = self.pianoroll.copy()
        for note_box in note_boxes:
            x, y, w, h = note_box
            img_marked = cv2.rectangle(img_marked, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imwrite("images/detected_notes.png", img_marked[::-1, :, :])

        notes = []
        for note_box in note_boxes:
            x, y, w, h = note_box
            notes.append({
                "hand": "right",
                "key": x + w // 2,
                "start": y,
                "duration": h,
            })
        notes = pd.DataFrame(notes, list(range(len(notes))))
        return notes


class Notes:
    """
    Wrapper class for notes.
    """
    def __init__(self, notes: pd.DataFrame, tempo: int):
        self._notes = notes.copy()
        self._notes_proc = notes.copy()
        self._tempo = tempo

    @classmethod
    def from_video(
            cls,
            video: Video,
            tempo: int,
            key_offset: int = 21,
            right_hand_boundary: int = 60,
    ) -> Notes:
        """
        Extract notes from the video.
        """
        # video.visualize_video("images/keyboard_regions.png")
        keyboard_video = video.get_keyboard_stripe()  # shape (width, rgb, frames)
        pianoroll_video = video.get_pianoroll_stripe()  # shape (width, rgb, frames)
        keyboard = Keyboard(keyboard_video.mean(axis=-1))

        pianoroll = Pianoroll(pianoroll_video)
        notes = pianoroll.get_notes_pixel()
        notes.start = notes.start / video.frame_rate * tempo / 60
        notes.duration = notes.duration / video.frame_rate * tempo / 60
        notes.key = notes.key.map(keyboard.pixel_to_key) + key_offset
        if right_hand_boundary:
            notes.hand = "right"
            notes.hand = notes.hand.where(notes.key >= right_hand_boundary, "left")
        return cls(notes, tempo)

    def post_process(self, quantization: Optional[int] = None, anacrusis: float = 0.0):
        """
        Post process notes.
        """
        notes = self._notes.sort_values("start")
        notes.start -= notes.start.min()
        if quantization:
            quantization /= 4
            notes.start = np.round(notes.start * quantization) / quantization
            notes.duration = np.round(notes.duration * quantization) / quantization
        notes = notes[notes.duration > 0]
        notes.start += 4 * (anacrusis // 4 + 1) - anacrusis
        self._notes_proc = notes

    def write_to_midi_file(self, filename: str):
        """
        Write processed notes to midi file.
        """
        midi = MIDIWrapper(self._tempo)
        midi.add_notes(self._notes_proc)
        midi.write_to_file(filename)
