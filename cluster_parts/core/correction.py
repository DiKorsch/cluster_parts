from scipy.ndimage.filters import gaussian_filter

class Corrector(object):
	"""implementation of some image corrections: Gaussian blur and gamma correction

	Arguments:
		- sigma
			Gaussian filter parameter

		- gamma
			Gamma correction parameter

		- squeeze
			squeeze input after correction

	"""
	def __init__(self, *, sigma, gamma, squeeze=True):
		super(Corrector, self).__init__()
		assert gamma > 0, "Gamma should be greater than 0!"

		self.sigma = sigma
		self.squeeze = squeeze
		self.gamma = gamma

	def __call__(self, im):
		res = im.copy()

		if self.sigma is not None:
			res = gaussian_filter(res, sigma=self.sigma)

		if self.squeeze:
			res = res.squeeze()

		return res**self.gamma

	@staticmethod
	def identity(x):
		return x.copy()


