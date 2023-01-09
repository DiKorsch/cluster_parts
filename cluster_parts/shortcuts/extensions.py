import chainer
import gc
import numpy as np

from chainer.backends.cuda import to_cpu
from chainer.dataset.convert import concat_examples
from chainer.training import extension
from chainercv import transforms as tr
from pathlib import Path
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
		self.output = output

		self._it_kwargs = dict(
			batch_size=batch_size,
			n_jobs=n_jobs,
			repeat=False, shuffle=False,
		)

		self._ready = False
		self._manual_gc = True

		self.__visualize = False

	def initialize(self, trainer):
		if self.output is None:
			self.output = trainer.out

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

		parts = []
		for batch in tqdm(it, total=n_batches,
			desc="Estimating CS Parts"):
			X, *_, y = concat_examples(batch, device=self.model.device)
			grad = ImageGradient(self.model, X)(cs=self.cs)
			saliency = to_cpu(grad2saliency(grad))

			for idx, (sal, x) in enumerate(zip(saliency, X), i):
				im_obj = self.ds.image_wrapped(idx)
				im = im_obj.im_array
				I = self._transform(im)

				boxes = self.extractor(I, sal)
				boxes = [[int(_) for _ in (i,x,y,w,h)] for i, ((x,y), w, h) in boxes]

				parts.append(boxes)

				uuid = self.ds.uuids[idx]
				self.ds._annot.set_parts(uuid, np.array(boxes, dtype=np.int32))

				if self.__visualize:
					_visualize(im, I, x, sal, boxes)

			i += len(saliency)

			if self._manual_gc:
				gc.collect()

		self.dump_boxes(parts)

	def dump_boxes(self, parts):
		if self.output is None:
			return

		output_dir = Path(self.output) / "parts"
		output_dir.mkdir(parents=True, exist_ok=True)

		part_locs =  output_dir / "part_locs.txt"
		part_names =  output_dir / "parts.txt"
		input_size =  output_dir / "input_size"

		arr = []
		for i, boxes in enumerate(parts):
			uuid = self.ds.uuids[i]
			for j, *coords in boxes:
				arr.append([uuid, j] + coords)

		arr = np.array(arr)
		np.savetxt(part_locs, arr, fmt="%s %s %s %s %s %s")

		with open(part_names, "w") as f:
			for i in np.unique(arr[:, 1]):
				print(f"Part #{i}", file=f)

		with open(input_size, "w") as f:
			size = tuple(self.ds.size)[0]
			print(f"Input size: {size}", file=f)

		exit(0)

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

	for i, x, y, w, h in boxes:
		axs[1,1].add_patch(plt.Rectangle((x,y), w, h, fill=False, edgecolor="blue"))

	plt.show()
	plt.close()
