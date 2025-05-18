from dataclasses import dataclass

import numpy as np

from moviepy.Clip import Clip
from moviepy.decorators import audio_video_effect
from moviepy.Effect import Effect
from moviepy.np_handler import cnp, np_convert
from moviepy.tools import convert_to_seconds


@dataclass
class MultiplyVolume(Effect):
    """Returns a clip with audio volume multiplied by the
    value `factor`. Can be applied to both audio and video clips.

    Parameters
    ----------

    factor : float
      Volume multiplication factor.

    start_time : float, optional
      Time from the beginning of the clip until the volume transformation
      begins to take effect, in seconds. By default at the beginning.

    end_time : float, optional
      Time from the beginning of the clip until the volume transformation
      ends to take effect, in seconds. By default at the end.

    Examples
    --------

    .. code:: python

        from moviepy import AudioFileClip

        music = AudioFileClip("music.ogg")
        # doubles audio volume
        doubled_audio_clip = music.with_effects([afx.MultiplyVolume(2)])
        # halves audio volume
        half_audio_clip = music.with_effects([afx.MultiplyVolume(0.5)])
        # silences clip during one second at third
        effect = afx.MultiplyVolume(0, start_time=2, end_time=3)
        silenced_clip = clip.with_effects([effect])
    """

    factor: float
    start_time: float = None
    end_time: float = None

    def __post_init__(self):
        self.start_time = (convert_to_seconds(self.start_time)
                           if self.start_time is not None else None)
        self.end_time = (convert_to_seconds(self.end_time)
                         if self.end_time is not None else None)
        self.xp = cnp if cnp else np  # Unified array module

    def _create_volume_mask(self, t):
        """Vectorized volume factor calculation"""
        mask = self.xp.ones_like(t)
        if self.start_time is not None or self.end_time is not None:
            start = self.start_time if self.start_time is not None else 0
            end = self.end_time if self.end_time is not None else t[-1] + 1
            mask[(t >= start) & (t <= end)] = self.factor
        return mask

    @audio_video_effect
    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        xp = self.xp
        nchannels = clip.nchannels

        if self.start_time is None and self.end_time is None:
            # Full clip optimization (remove explicit dtype)
            def full_volume(get_frame, t):
                frame = np_convert(get_frame(t, to_np=False))
                return xp.multiply(frame, self.factor)
            return clip.transform(full_volume, keep_duration=True)

        # Partial clip optimization
        def volume_transform(get_frame, t):
            frame = np_convert(get_frame(t, to_np=False))
            t_array = xp.asarray(t)
            factors = self._create_volume_mask(t_array)

            if nchannels > 1:
                factors = factors[:, xp.newaxis]  # Broadcast to stereo channels

            return xp.multiply(frame, factors)  # Remove dtype forcing

        return clip.transform(volume_transform, keep_duration=True)
