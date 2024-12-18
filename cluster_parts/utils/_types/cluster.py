import numpy as np
import enum

from skimage.feature import peak_local_max

from cvargparse.utils.enumerations import BaseChoiceType

class ClusterInitType(BaseChoiceType):
	NONE = enum.auto()
	MAXIMAS = enum.auto()
	MIN_MAX = enum.auto()

	Default = MAXIMAS

	def __call__(self, grad, K=None):
		assert grad.ndim == 2, \
			f"Incorrect input: ndim={grad.ndim} != 2!"

		if self == ClusterInitType.MAXIMAS:
			peaks = peak_local_max(grad, num_peaks=K).T
			return peaks if peaks.shape[1] == K else None


		elif self == ClusterInitType.MIN_MAX:

			max_loc = np.unravel_index(grad.argmax(), grad.shape)
			min_loc = np.unravel_index(grad.argmin(), grad.shape)

			return np.vstack([min_loc, max_loc]).T

		else:
			return None
