import numpy as np

from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

from cvargparse.utils.enumerations import BaseChoiceType

from cluster_parts.utils import ClusterInitType
from cluster_parts.utils import FeatureComposition
from cluster_parts.utils import FeatureType


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
			K = 2 # background vs. foreground thresholding
			init_coords = ClusterInitType.MIN_MAX(grad, K=K)
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
			thresh = threshold_otsu(grad)
			return grad > thresh
		else:
			return None
