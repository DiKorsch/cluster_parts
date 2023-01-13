import chainer
import gc
import logging
import multiprocessing as mp
import numpy as np
import typing as T
import warnings

from chainer import functions as F
from chainer.backends.cuda import to_cpu
from chainer.dataset.convert import concat_examples
from chainer.training import extension
from chainercv import transforms as tr
from pathlib import Path
from tqdm.auto import tqdm

from cluster_parts.shortcuts.image_gradient import ImageGradient
from cluster_parts.utils.operations import grad2saliency

def identity(x):
	return x

class ExtractorPool:

	def __init__(self, extractor, dataset, *args,
		n_jobs: int = 0,
		transform: T.Callable = None):

		self.extractor = extractor
		self.ds = dataset
		self.transform = transform

		self.pool = None
		if n_jobs >= 1:
			self.pool = mp.Pool(n_jobs)

	def __getstate__(self):
		state = self.__dict__.copy()
		del state["pool"]
		return state

	# def __setstate__(self, state):
	# 	self.__dict__.update(state)

	@property
	def transform(self):
		return self._transform

	@transform.setter
	def transform(self, func):
		if func is None or not callable(func):
			reason = "None" if func is None else "not callable"
			warnings.warn(f"Setting transformation function to identity, because it was {reason}.")
			func = identity
		self._transform = func


	def __call__(self, saliency, offset) -> T.Tuple[int, T.List[int]]:

		mapper = map if self.pool is None else self.pool.map

		for idx, boxes in mapper(self.work, enumerate(saliency, offset)):
			yield idx, boxes

	def work(self, args):
		idx, sal = args

		im_obj = self.ds.image_wrapped(idx)
		im = im_obj.im_array
		I = self.transform(im)

		boxes = self.extractor(I, sal)
		boxes = [[int(_) for _ in (i,x,y,w,h)] for i, ((x,y), w, h) in boxes]
		return idx, np.array(boxes, dtype=np.int32)



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
		track_accuracy: bool = True,
		manual_gc: bool = True,
		sample_pos: int = 0,
		label_pos: int = 1,

		visualize: bool = False,
		**kwargs):
		super(CSPartEstimation, self).__init__()

		self._it_kwargs = dict(
			batch_size=batch_size,
			n_jobs=n_jobs,
			repeat=False, shuffle=False,
		)

		self.ds = dataset
		self.model = model
		self.extractor = extractor
		self.cs = cs
		self.output = output


		self._ready = False
		self._manual_gc = manual_gc
		self._sample_pos = sample_pos
		self._label_pos = label_pos
		self._track_accuracy = track_accuracy

		self.__visualize = visualize

	def __getstate__(self):
		state = self.__dict__.copy()
		del state["_model"]
		del state["_model_init"]
		return state

	def initialize(self, trainer):
		if self.output is None:
			self.output = trainer.out

	@property
	def model(self) -> T.Callable:
		if self._model is None:
			self._model = self._model_init.copy(mode="copy")

		return self._model

	@model.setter
	def model(self, model) -> None:
		self._model = None
		self._model_init = model

	@property
	def extractor(self):
		return self._extractor

	@extractor.setter
	def extractor(self, extr):
		self._extractor = ExtractorPool(
			extractor=extr,
			dataset=self.ds,
			transform=self._transform,
			n_jobs=0, #self._it_kwargs["n_jobs"]
		)


	@property
	def clf(self) -> T.Callable:
		return self.model.clf_layer

	def _transform(self, im: np.ndarray, size=None) -> np.ndarray:
		if size is None:
			s = min(im.shape[:2])
			size = (s, s)
		im = im.transpose(2,0,1)
		if self.ds.center_cropped:
			return tr.center_crop(im, size).transpose(1,2,0)

		return tr.resize(im, size).transpose(1,2,0)

	def unpack_batch(self, batch):
		batch = concat_examples(batch, device=self.model.device)
		return batch[self._sample_pos], batch[self._label_pos]

	def estimate_parts(self, it, n_batches) \
		-> T.List[T.List[int]]:

		offset = 0

		parts = []
		preds, labs = [], []
		for batch in tqdm(it, total=n_batches, desc="Estimating CS Parts"):
			X, y = self.unpack_batch(batch)
			grad = ImageGradient(self.model, X)(cs=self.cs)
			saliency = to_cpu(grad2saliency(grad))

			if self._manual_gc:
				gc.collect()

			if self._track_accuracy:
				feat = self.model.extract(X)
				pred = F.argmax(self.clf(feat), axis=1)
				preds.extend(to_cpu(chainer.as_array(pred)))
				labs.extend(to_cpu(chainer.as_array(y)))

			for idx, boxes in self.extractor(saliency, offset=offset):

				uuid = self.ds.uuids[idx]
				self.ds._annot.set_parts(uuid, boxes)
				parts.append(boxes)


				if self.__visualize:
					i = idx - offset
					im_obj = self.ds.image_wrapped(idx)
					im = im_obj.im_array
					I = self._transform(im)
					_visualize(im, I, X[i], saliency[i], boxes)

			offset += len(saliency)


		preds, labs = map(np.array, [preds, labs])
		self.evaluate(preds, labs)
		return parts

	def evaluate(self, preds, labs):
		if not self._track_accuracy:
			return
		tr_split = self.ds._annot.train_split

		for subset, split in [("Training", tr_split), ("Test", ~tr_split)]:
			split = split[:len(preds)]
			accu = np.mean(preds[split] == labs[split])

			logging.info(f"{subset} accuracy: {float(accu):.3%}")



	def dump_boxes(self, parts) -> None:
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


	def __call__(self, trainer) -> None:
		assert not self._ready, "Extension should only be called once!"

		it, n_batches = self.ds.new_iterator(**self._it_kwargs)
		with chainer.using_config("train", False):
			parts = self.estimate_parts(it, n_batches)

		self.dump_boxes(parts)
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
