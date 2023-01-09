import chainer

from chainer.dataset.convert import concat_examples
from chainer.backends.cuda import to_cpu
from chainer.training import extension
from chainercv import transforms as tr
from tqdm.auto import tqdm

from cluster_parts.shortcuts.image_gradient import ImageGradient
from cluster_parts.utils.operations import grad2saliency


class CSPartEstimation(extension.Extension):

	name = "CSPartEstimation"
	trigger = None
	priority = extension.PRIORITY_WRITER + 1

	def __init__(self, dataset, model, extractor,
		*args,
		cs: bool = True, # classification-specific
		batch_size: int = 32,
		n_jobs: int = 0,
		output: str = None,
		**kwargs):
		super(CSPartEstimation, self).__init__()

		self.ds = dataset
		self.model = model
		self.extractor = extractor
		self.cs = cs

		self._it_kwargs = dict(
			batch_size=batch_size,
			n_jobs=n_jobs,
			repeat=False, shuffle=False,
		)

		self._ready = False

		self.__visualize = True


	@property
	def model(self):
		if self._model is None:
			self._model = self.__model.copy(mode="copy")

		return self._model

	@model.setter
	def model(self, model):
		self._model = None
		self.__model = model


	def _transform(self, im, size=None):
		if size is None:
			s = min(im.shape[:2])
			size = (s, s)
		im = im.transpose(2,0,1)
		if self.ds.center_cropped:
			return tr.center_crop(im, size).transpose(1,2,0)

		return tr.resize(im, size).transpose(1,2,0)

	def estimate_parts(self, it, n_batches):

		i = 0
		boxes = []

		for batch in tqdm(it, total=n_batches):
			X, *_, y = concat_examples(batch, device=self.model.device)
			grad = ImageGradient(self.model, X)(cs=self.cs)
			saliency = to_cpu(grad2saliency(grad))

			for idx, (sal, x) in enumerate(zip(saliency, X), i):
				im = self.ds.image_wrapped(idx).im_array
				I = self._transform(im)

				_boxes = self.extractor(I, sal)
				boxes.append(_boxes)

				if self.__visualize:
					_visualize(im, I, x, sal, _boxes)

			i += len(saliency)

			# import pdb; pdb.set_trace()


	def __call__(self, trainer):
		assert not self._ready, "Extension should only be called once!"

		it, n_batches = self.ds.new_iterator(**self._it_kwargs)
		with chainer.using_config("train", False):
			parts = self.estimate_parts(it, n_batches)

		self._ready = True



def _visualize(orig, transformed, cnn_input, saliency, boxes):
	from matplotlib import pyplot as plt

	# -1..1 -> 0..1
	cnn_input = (cnn_input + 1) / 2
	cnn_input = to_cpu(cnn_input.transpose(1,2,0))

	fig, axs = plt.subplots(2,2, squeeze=False)

	axs[0, 0].imshow(orig)
	axs[0, 1].imshow(transformed)

	axs[1, 0].imshow(cnn_input)
	_im = tr.resize(transformed.transpose(2,0,1), saliency.shape[:2]).transpose(1,2,0)
	axs[1, 1].imshow(_im, alpha=0.5)
	axs[1, 1].imshow(saliency, alpha=0.5)

	for i, box in boxes:
		axs[1,1].add_patch(plt.Rectangle(*box, fill=False, edgecolor="blue"))

	plt.show()
	plt.close()
