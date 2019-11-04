import numpy as np

from skimage.feature import peak_local_max

from cvargparse.utils.enumerations import BaseChoiceType

class ClusterInitType(BaseChoiceType):
	NONE = 0
	MAXIMAS = 0
	MIN_MAX = 2

	Default = MAXIMAS

	def __call__(self, grad, K=None):

		if self == ClusterInitType.MAXIMAS:
			return peak_local_max(grad, num_peaks=K).T

		elif self == ClusterInitType.MIN_MAX:

			max_loc = np.unravel_index(grad.argmax(), grad.shape)
			min_loc = np.unravel_index(grad.argmin(), grad.shape)

			return np.vstack([min_loc, max_loc]).T

		else:
			return None
