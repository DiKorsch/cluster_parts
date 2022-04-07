import numpy as np

from cvargparse.utils.enumerations import BaseChoiceType

def _norm(arr):
	arr = arr - arr.min()
	arr_max = arr.max()
	if arr_max == 0:
		return arr
	else:
		return arr / arr_max

class FeatureType(BaseChoiceType):

	COORDS = "coords"
	SALIENCY = "saliency"
	RGB = "rgb"

	Default = COORDS

	def __call__(self, im, saliency, coords):

		assert len(coords) == 2, \
			f"Incorrect input: {len(coords)=} != 2!"

		assert saliency.ndim in (2, 3), \
			f"Incorrect input: {saliency.ndim=} is neither 2 nor 3!"

		ys, xs = coords
		if self == FeatureType.COORDS:
			return [ _norm(ys), _norm(xs) ]

		elif self == FeatureType.SALIENCY:
			_saliency = saliency[ys, xs]
			if _saliency.ndim == 1:
				_saliency = np.expand_dims(_saliency, axis=1)
			C = _saliency.shape[1]
			return [ _norm(_saliency[:, chan].ravel()) for chan in range(C) ]

		elif self == FeatureType.RGB:
			_im = im[ys, xs]
			return [ _norm(_im[:, chan].ravel()) for chan in range(3) ]


class FeatureComposition(object):

	Default = [ FeatureType.COORDS ]

	def __init__(self, composition):
		super(FeatureComposition, self).__init__()

		self.composition = [FeatureType.get(comp) for comp in composition]


	def __call__(self, *args, **kwargs):
		feats = []
		for comp in self.composition:
			feats.extend(comp(*args, **kwargs))

		return np.stack(feats).transpose()
