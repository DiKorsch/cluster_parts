import numpy as np

from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

from cvargparse.utils.enumerations import BaseChoiceType

from cluster_parts.utils import ClusterInitType
from cluster_parts.utils import FeatureComposition
from cluster_parts.utils import FeatureType
from cluster_parts.utils import operations


class ThresholdType(BaseChoiceType):
	NONE = 0
	MEAN = 1
	PRECLUSTER = 2
	OTSU = 3

	Default = MEAN

	def __call__(self, im, grad):
		assert grad.ndim in (2, 3), \
			f"Incorrect input: {grad.ndim=} is neither 2 nor 3!"

		l2_grad = operations.l2_norm(grad)

		if self == ThresholdType.MEAN:
			return l2_grad > l2_grad.mean()

		elif self == ThresholdType.PRECLUSTER:
			K = 2 # background vs. foreground thresholding
			init_coords = ClusterInitType.MIN_MAX(l2_grad, K=K)
			feats = FeatureComposition([FeatureType.SALIENCY])

			init = feats(None, grad, init_coords)
			kmeans = KMeans(n_clusters=K, init=init, n_init=1)

			h, w = grad.shape[:2]
			idxs = np.arange(h * w)
			coords = np.unravel_index(idxs, (h, w))
			data = feats(None, grad, coords)

			kmeans.fit(data)

			labs = kmeans.labels_.reshape(h, w)

			# 1-cluster represents the cluster around the maximal peak
			return labs == 1

		elif self == ThresholdType.OTSU:
			thresh = threshold_otsu(l2_grad)
			return l2_grad > thresh

		else:
			return None
