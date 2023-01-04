import chainer
import chainer.functions as F


class ImageGradient:

	def __init__(self, model, X):
		super().__init__()

		self.X = chainer.Variable(X, name="Input")
		self.model = model
		with chainer.force_backprop_mode():
			self.feats = model(self.X, model.meta.feature_layer)

	def __call__(self, cs: bool = False):

		""" cs = classification-specific """

		self.X.grad = None
		self.model.cleargrads()

		if cs:
			clf = self.model.clf_layer
			pred = chainer.as_array(clf(self.feats))
			cls_ids = pred.argmax(axis=1)
			weights = F.normalize(clf.W).array[cls_ids]
			F.sum(self.feats * weights).backward()
		else:
			F.sum(self.feats).backward()

		assert self.X.grad is not None, "Backprop mode is off?"
		return self.X.grad
