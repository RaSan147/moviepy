from dataclasses import dataclass

from PIL import Image, ImageFilter

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.np_handler import np, np_get


@dataclass
class GaussianBlur(Effect):
    """Returns a filter that applies Gaussian blur to the entire frame.

    Parameters
    ----------
    radius : float
        The radius of the Gaussian blur. Higher values create more blur.
    """

    radius: float

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        def filter(gf, t):
            im = gf(t)
            im = np_get(im)
            image = Image.fromarray(im)
            blurred = image.filter(ImageFilter.GaussianBlur(radius=self.radius))
            return np.array(blurred)

        return clip.transform(filter)
