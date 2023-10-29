"""
Functions to extract notes.
"""
from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from .midi import MIDIWrapper
from .video import Video


class Keyboard:
    """
    Wrapper class for actions regarding the keyboard in the video.
    """
    def __init__(self, ref_stripe: np.ndarray, ref_stripe_white: np.ndarray):
        self._ref_stripe = ref_stripe  # shape (width, rgb)
        self._keys, self._key_borders = self.get_keys_and_borders(ref_stripe)  # keys shape (width,)
        self._key_matrix = (self._keys[:, None] == np.array(range(self._keys.max()))[None, :]).astype(int)
        self._ref_stripe_white = ref_stripe_white  # shape (width, rgb)
        self._keys_white, self._key_borders_white = self.get_keys_and_borders(ref_stripe_white)  # keys shape (width,)
        self._key_matrix_white = (
                self._keys_white[:, None] == np.array(range(self._keys_white.max()))[None, :]).astype(int)

        self._white_to_real_key = {}
        for key, borders in self._key_borders.items():
            if self.is_white_key(key):
                white_key = int(self._keys_white[int(np.mean(borders))])
                self._white_to_real_key[white_key] = key

    @staticmethod
    def get_keys_and_borders(ref_stripe: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Find the borders for each key of the keyboard.
        """
        keys = np.zeros(ref_stripe.shape[0], dtype=int)
        key_borders = {}
        stripe = ref_stripe.mean(axis=-1)
        deltas = np.abs(np.diff(stripe))
        borders = deltas > 2 * np.mean(deltas)
        key = 0
        border = 0
        for idx in range(borders.size - 1):
            if borders[idx] and any(borders[idx + 1:idx + 3]):  # remove double borders
                borders[idx] = False
            if borders[idx]:
                key_borders[key] = (border, idx)
                key += 1
                border = idx + 1
        key_borders[key] = (border, borders.size)
        for key, borders in key_borders.items():
            keys[borders[0]:borders[1] + 1] = key
        return keys, key_borders

    def pixel_to_key(self, pixel: int) -> int:
        """
        Get key for given pixel.
        """
        return int(self._keys[pixel])

    def key_to_pixels(self, key: int) -> np.ndarray:
        """
        Get array with the active pixels for a given key.
        """
        return self._keys == key

    def note_box_to_key(self, start: int, width: int) -> int:
        """
        Get key for given box.
        """
        box = np.zeros(self._ref_stripe.shape[0])
        box[start:start+width] = 1
        diffs = self._key_matrix - box[:, None]
        least_diff = np.min(np.abs(diffs).sum(axis=0))
        least_diff_key = np.argmin(np.abs(diffs).sum(axis=0))
        diffs_white = self._key_matrix_white - box[:, None]
        least_diff_white = np.min(np.abs(diffs_white).sum(axis=0))
        least_diff_white_key = np.argmin(np.abs(diffs_white).sum(axis=0))
        least_diff_white_key = self._white_to_real_key[least_diff_white_key]
        if least_diff_white < least_diff:
            return least_diff_white_key
        else:
            return least_diff_key

    def is_white_key(self, key: int) -> bool:
        """
        Check whether the given key is white.
        """
        average_gray = self._ref_stripe[self._key_borders[key][0]:self._key_borders[key][1]].mean()
        return average_gray > 255 / 2

    def add_keyboard_to_image(self, img: np.ndarray, add_borders: bool = True) -> np.ndarray:
        """
        Add keyboard to pianoroll image.
        """
        keyboard_img = np.repeat(self._ref_stripe[None, ...], 100, axis=0)
        sep_stripe = np.ones((1, img.shape[1], 3)) * 150
        img = np.concatenate([keyboard_img, sep_stripe, img], axis=0)
        if add_borders:
            for key in self._key_borders.values():  # gray border lines
                img[:, key[0], :] = 150
                img[:, key[1], :] = 150
        return img


class Pianoroll:
    """
    Wrapper class for pianoroll in tutorial.
    """
    def __init__(self, pianoroll: np.ndarray, frame_rate: float):
        self.pianoroll = pianoroll.transpose(2, 0, 1)  # shape (frames, width, rgb)
        self.note_boxes = []  # list of detected note boxes
        self._detect_notes()
        self.frame_rate = frame_rate

    def _detect_notes(self) -> np.ndarray:
        """
        Detect notes in the pianoroll.
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

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if np.abs(w / white_key_width - 1) < 0.2:  # white key
                self.note_boxes.append((x, y, w, h))
            elif np.abs(w / black_key_width - 1) < 0.2:  # black key
                self.note_boxes.append((x, y, w, h))
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
                        self.note_boxes.append(
                            (x + key * white_key_width + offset, y + start, white_key_width, end - start)
                        )

    def get_notes(self, keyboard: Keyboard) -> np.ndarray:
        """
        Get notes from the pianoroll.
        """
        notes = []
        for note_box in self.note_boxes:
            x, y, w, h = note_box
            notes.append({
                "hand": "right",
                "key": keyboard.note_box_to_key(x, w),
                "start": y,
                "duration": h,
            })
        notes = pd.DataFrame(notes, list(range(len(notes))))
        return notes

    def get_image(self, note_boxes: bool = True) -> np.ndarray:
        """
        Return pianoroll image.
        """
        img = self.pianoroll.copy()
        if note_boxes:
            img = self.add_boxes_to_image(img)
        return img

    def add_boxes_to_image(self, img: np.ndarray) -> np.ndarray:
        """
        Add detected boxes to given pianoroll image.
        """
        for note_box in self.note_boxes:
            x, y, w, h = note_box
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return img


class Notes:
    """
    Wrapper class for notes.
    """
    def __init__(self, notes: pd.DataFrame, tempo: int, key_offset: int):
        self._notes = notes.copy()
        self._notes_proc = notes.copy()
        self._tempo = tempo
        self._key_offset = key_offset

    def add_notes_to_image(self, img: np.ndarray, keyboard: Keyboard, frame_rate: float) -> np.ndarray:
        """
        Add detected notes to pianoroll image.
        Note that notes need to have start and end in frames, not seconds or beats.
        """
        for _, note in self._notes.iterrows():
            # add red color shade
            start = int(note.start * 60 / self._tempo * frame_rate)
            duration = int(note.duration * 60 / self._tempo * frame_rate)
            color_shade = (keyboard.key_to_pixels(note.key - self._key_offset) * 100).astype("uint8")
            img[start:start + duration, :, 2] += color_shade
        return img

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
        video.visualize_video("images/keyboard_regions.png")
        keyboard_video = video.get_keyboard_stripe()  # shape (width, rgb, frames)
        keyboard_white_video = video.get_keyboard_white_stripe()  # shape (width, rgb, frames)
        pianoroll_video = video.get_pianoroll_stripe()  # shape (width, rgb, frames)
        keyboard = Keyboard(keyboard_video.mean(axis=-1), keyboard_white_video.mean(axis=-1))

        pianoroll = Pianoroll(pianoroll_video, video.frame_rate)
        notes = pianoroll.get_notes(keyboard)
        notes.key += key_offset
        notes.start = notes.start / video.frame_rate * tempo / 60
        notes.duration = notes.duration / video.frame_rate * tempo / 60
        if right_hand_boundary:
            notes.hand = "right"
            notes.hand = notes.hand.where(notes.key >= right_hand_boundary, "left")

        notes_obj = cls(notes, tempo, key_offset)
        img = TutorialImage(keyboard, pianoroll, notes_obj)
        img.write_to_file("images/detected_notes.png")
        return notes_obj

    def post_process(self, quantization: Optional[int] = None, anacrusis: float = 0.0):
        """
        Post process notes.
        """
        notes = self._notes.sort_values("start")
        notes.start -= notes.start.min()
        notes.start += 4 * (anacrusis // 4 + 1) - anacrusis

        if quantization:
            quantization /= 4
            notes.start = np.round(notes.start * quantization) / quantization
            notes.duration = np.round(notes.duration * quantization) / quantization
        notes = notes[notes.duration > 0]
        self._notes_proc = notes

    def write_to_midi_file(self, filename: str):
        """
        Write processed notes to midi file.
        """
        midi = MIDIWrapper(self._tempo)
        midi.add_notes(self._notes_proc)
        midi.write_to_file(filename)


class TutorialImage:
    """
    Helper class to create images of a tutorial.
    """
    def __init__(self, keyboard: Keyboard, pianoroll: Pianoroll, notes: Notes):
        self.keyboard = keyboard
        self.pianoroll = pianoroll
        self.notes = notes

        self._img = pianoroll.get_image()
        self._img = notes.add_notes_to_image(self._img, keyboard, pianoroll.frame_rate)
        self._img = keyboard.add_keyboard_to_image(self._img)

    def write_to_file(self, filename: str):
        """
        Write image to file.
        """
        cv2.imwrite(filename, self._img[::-1, :, :])
