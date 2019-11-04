import numpy as np

from skimage.filters import threshold_otsu

from cvargparse.utils.enumerations import BaseChoiceType


class ThresholdType(BaseChoiceType):
	NONE = 0
	MEAN = 1
	PRECLUSTER = 2
	OTSU = 3

	Default = PRECLUSTER

	def __call__(self, im, grad):
		if self == ThresholdType.MEAN:
			return np.abs(grad).mean()

		elif self == ThresholdType.PRECLUSTER:
			from .clustering import cluster_gradient
			centers, labs = cluster_gradient(im, grad,
				K=2, thresh=None,
				cluster_init=ClusterInitType.MIN_MAX,
				# small fix, since it does not work with only one dimension
				# or at least, it has to be fixed
				feature_composition=["grad", "grad"]
			)


			# 1th cluster represents the cluster around the maximal peak
			return labs == 1

		elif self == ThresholdType.OTSU:
			thresh = threshold_otsu(grad)
			return grad > thresh
		else:
			return None
