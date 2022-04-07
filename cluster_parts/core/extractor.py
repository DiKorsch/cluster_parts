import logging
import numpy as np

from functools import partial
from functools import wraps
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from cluster_parts.core.measures import FScore
from cluster_parts.core.measures import Precision
from cluster_parts.core.measures import Recall
from cluster_parts.utils import ClusterInitType
from cluster_parts.utils import FeatureComposition
from cluster_parts.utils import ThresholdType
from cluster_parts.utils import operations

def _check_min_bbox(bbox, min_bbox):
	y0, x0, y1, x1 = bbox
	h, w = y1 - y0, x1 - x0

	# if bbox is greater that min_bbox in both, the width and height
	if min(h, w) >= min_bbox:
		return bbox

	old_bbox = bbox.copy()
	dy, dx = max(min_bbox - h, 0), max(min_bbox - w, 0)

	bbox[0] -= int(dy / 2)
	bbox[1] -= int(dx / 2)
	bbox[2] += int(dy / 2)
	bbox[3] += int(dx / 2)

	if (bbox < 0).any():
		dy, dx, _, _ = np.minimum(bbox, 0)
		bbox[0] -= dy
		bbox[1] -= dx
		bbox[2] -= dy
		bbox[3] -= dx

	text = "Adjusted bbox from {} to {}".format(old_bbox, bbox)
	logging.debug("=" * len(text))
	logging.debug(text)
	logging.debug("=" * len(text))
	return bbox

class EnlargeBbox(object):
	def __init__(self, factor):
		self.factor = factor

	def __call__(self, func):

		@wraps(func)
		def inner(mask, *args, **kwargs):
			im_h, im_w = mask.shape
			bbox = func(mask, *args, **kwargs)
			if self.factor <= 0:
				return bbox

			y0, x0, y1, x1 = bbox

			w, h = x1 - x0, y1 - y0
			dx = w * self.factor / 2
			dy = h * self.factor / 2

			y0, x0 = max(y0 - dy, 0), max(x0 - dx, 0)
			y1, x1 = min(y1 + dy, im_h), min(x1 + dx, im_w)
			return y0, x0, y1, x1
		return inner


class BoundingBoxPartExtractor(object):
	"""Extracts bounding box parts from a saliency map and an image

		Arguments:
			image
				- input image

			optimal (default: True)
				- optimize bounding box selection

			gamma, sigma
				- saliency correction parameters

	"""

	def __init__(self, corrector, *,
		K=4, optimal=True,
		min_bbox=64, fit_object=False,
		thresh_type=ThresholdType.Default,
		cluster_init=ClusterInitType.Default,
		feature_composition=FeatureComposition.Default):
		super().__init__()

		self.corrector = corrector

		assert K is not None and K > 0, "Positive K is required!"
		self.K = K
		self.optimal = optimal
		self.thresh_type = ThresholdType.get(thresh_type)
		self.cluster_init = ClusterInitType.get(cluster_init)
		self.feature_composition = FeatureComposition(feature_composition)
		self.fit_object = fit_object
		self.min_bbox = min_bbox

	def __call__(self, image, saliency):
		if isinstance(saliency, (list, tuple)):
			return [self(image, sal) for sal in saliency]

		saliency = self.corrector(saliency)
		centers, labs = self.cluster_saliency(image, saliency)
		boxes = self.get_boxes(centers, labs, saliency)
		return boxes


	def get_boxes(self, centers, labels, saliency):
		if self.optimal:
			saliency = operations.l2_norm(saliency)
			saliency = operations.normalize(saliency)
		else:
			saliency = None

		obj_box = None

		res = []
		for i in range(self.K):
			mask = labels == i
			if mask.sum() == 0:
				# if there is no cluster for this label,
				# then take the extend of the whole object
				if obj_box is None:
					# lazy init this box to speed up the things a bit
					obj_mask = np.logical_not(np.isnan(labels))
					obj_box = self.fit_bbox(obj_mask, saliency)

				y0, x0, y1, x1 = obj_box
			else:
				y0, x0, y1, x1 = self.fit_bbox(mask, saliency)

			h, w = y1 - y0, x1 - x0
			res.append([i, ((x0, y0), w, h)])

		if self.fit_object:
			if obj_box is None:
				obj_mask = np.logical_not(np.isnan(labels))
				obj_box = self.fit_bbox(obj_mask, saliency)

			y0, x0, y1, x1 = obj_box
			h, w = y1 - y0, x1 - x0
			res.append([i + 1, ((x0, y0), w, h)])

		return res

	# @EnlargeBbox(factor=0.2)
	def fit_bbox(self, mask, saliency):
		ys, xs = np.where(mask)
		bbox = np.array([min(ys), min(xs), max(ys), max(xs)])

		bbox = _check_min_bbox(bbox, self.min_bbox)
		if not self.optimal:
			return bbox

		# (1) Our search area is [(x0,y0), (x1,y1)].
		# (2) If we  shift it with (x0,y0) it becomes [(0,0), (w,h)]
		# (3) We see it in normalized way, so it changes to [(0,0), (1,1)]
		# (4) Hence, the initial bbox is always [(0.25, 0.25), (0.75, 0.75)]  with width and height 0.5

		y0, x0, y1, x1 = bbox
		h, w = y1 - y0, x1 - x0
		search_area = mask[y0:y1, x0:x1].astype(np.float32)

		# the search area is weighted with the saliency values
		if saliency is not None:
			assert 0.0 <= saliency.max() <= 1.0, \
				f"Invalid max value: {saliency.max()=}"
			search_area *= saliency[y0:y1, x0:x1]

		scaler = np.array([h, w, h, w])

		init_bbox = np.array([0.25, 0.25, 0.75, 0.75])

		func = Recall(search_area, scaler=scaler)

		# func = Precision(search_area, scaler=scaler)
		# func = FScore(search_area, scaler=scaler, beta=1)
		# func = FScore(search_area, scaler=scaler, beta=2)
		# func = FScore(search_area, scaler=scaler, beta=0.5)

		res = minimize(
			fun=func,
			x0=init_bbox,
			options=dict(eps=1e-2, gtol=1e-1),
			bounds=Bounds(0, 1),
		)
		# scale back to (2) and shift to original values (1)
		bbox = res.x * scaler + np.array([y0, x0, y0, x0])
		bbox = _check_min_bbox(bbox, self.min_bbox)
		return bbox

	def cluster_saliency(self, image, saliency):
		thresh = self.thresh_type(image, saliency)
		l2_saliency = operations.l2_norm(saliency)

		thresh_mask = None
		if isinstance(thresh, (int, float, saliency.dtype.type)):
			thresh_mask = l2_saliency > thresh

		elif isinstance(thresh, np.ndarray):
			# thresh is already a mask
			thresh_mask = thresh

		init_coords = self.cluster_init(l2_saliency, self.K)

		if init_coords is None:
			clf = KMeans(self.K)

		else:
			init = self.feature_composition(image, saliency, init_coords)
			clf = KMeans(self.K, init=init, n_init=1)

		h, w, *_ = saliency.shape
		### get x,y coordinates of pixels to cluster
		if thresh_mask is not None:
			coords = np.where(thresh_mask)

		else:
			idxs = np.arange(h * w)
			coords = np.unravel_index(idxs, (h, w))

		data = self.feature_composition(image, saliency, coords)

		clf.fit(data)

		labels = np.full((h, w), np.nan)
		labels[coords] = clf.labels_
		centers = clf.cluster_centers_.copy()
		centers[:, 0] *= h
		centers[:, 1] *= w

		return centers, labels

