import abc
from skimage.transform.integral import integral_image

_recall     = lambda TP, FP, FN, TN: TP / (TP + FN)
_precision  = lambda TP, FP, FN, TN: TP / (TP + FP)

def integrate(img, y0, x0, y1, x1):
	A = img[y0, x0]
	B = img[y0, x1]
	C = img[y1, x0]
	D = img[y1, x1]
	return A + D - B - C

class BaseMeasure(abc.ABC):

	def __init__(self, search_area, scaler=1):
		super(BaseMeasure, self).__init__()
		self.scaler = scaler
		self.search_area = search_area

		self.search_area_sum = search_area.sum()
		self.neg_search_area_sum = ( 1 - search_area ).sum()

		self.integral_search_area = integral_image(search_area)
		self.integral_neg_search_area = integral_image(1 - search_area)

	@abc.abstractmethod
	def _measure(self, TP, FP, FN, TN, ratio):
		raise NotImplementedError

	def __call__(self, bbox):
		# scale back to the area mentioned in (2)
		y0, x0, y1, x1 = map(int, bbox * self.scaler)
		h, w = self.search_area.shape
		y1, x1 = min(y1, h-1), min(x1, w-1)
		y0, x0 = min(y0, y1), min(x0, x1)
		#area = self.search_area[y0:y1, x0:x1]
		_h, _w = y1 - y0, x1 - x0

		if _h < _w:
			ratio = (_h / _w) if _w != 0 else -100
		else:
			ratio = (_w / _h) if _h != 0 else -100

		##### THIS IS THE SLOWEST PART, BECAUSE IT IS CALLED MANY TIMES!

		# area.sum()
		# TP = integrate(self.integral_search_area, start=(y0,x0), end=(y1,x1))
		TP = integrate(self.integral_search_area, y0, x0, y1, x1)
		# (1-area).sum()
		# FP = integrate(self.integral_neg_search_area, start=(y0,x0), end=(y1,x1))
		FP = integrate(self.integral_neg_search_area, y0, x0, y1, x1)

		# search_area.sum() - TP
		FN = self.search_area_sum - TP
		# (1-search_area).sum() - FP
		TN = self.neg_search_area_sum - FP

		# ratio = 1
		return self._measure(TP, FP, FN, TN, ratio)

class Recall(BaseMeasure):

	def _measure(self, TP, FP, FN, TN, ratio):
		return -_recall(TP, FP, FN, TN) * ratio

class Precision(BaseMeasure):

	def _measure(self, TP, FP, FN, TN, ratio):
		return -_precision(TP, FP, FN, TN) * ratio

class FScore(BaseMeasure):

	def __init__(self, *args, beta=1, **kwargs):
	    super(FScore, self).__init__(*args, **kwargs)
	    self.beta = beta

	def _measure(self, TP, FP, FN, TN, ratio):
		recall = _recall(TP, FP, FN, TN)
		prec = _precision(TP, FP, FN, TN)

		return -((1 + self.beta**2) * (recall * prec) / (recall + self.beta**2 * prec) ) * self.scaler

