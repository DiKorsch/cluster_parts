import chainer
import numpy as np

from chainercv import transforms as tr
from cvdatasets import dataset
from cvdatasets.dataset.image import ImageWrapper
from cvmodelz.models import BaseModel

from cluster_parts.core.extractor import BoundingBoxPartExtractor
from cluster_parts.shortcuts.image_gradient import ImageGradient
from cluster_parts.utils.operations import grad2saliency


class CSPartsDataset(
	dataset.ImageProfilerMixin,
	dataset.IteratorMixin,
	dataset.TransformMixin,
	dataset.AnnotationsReadMixin):

	def __init__(self, *args,
		model: BaseModel,
		extractor: BoundingBoxPartExtractor,
		clf = None,
		prepare = None,
		cs: bool = True, # classification-specific
		include_visualization: bool = False,
		**kwargs):
		parts = kwargs.pop("parts", None)
		kwargs["size"] = kwargs.pop("size", model.meta.input_size)
		kwargs["part_size"] = kwargs.pop("part_size", None)
		super().__init__(*args, **kwargs)

		self.model = model
		self.extractor = extractor
		self.cs = cs
		self._clf = clf
		self._prepare = prepare
		self._include_visualization = include_visualization
		self._visualization = {}

	@property
	def clf(self):
		if self._clf is None:
			return self.model.clf_layer

		return self._clf

	def _transform(self, im, size=None):
		if size is None:
			s = min(im.shape[:2])
			size = (s, s)
		im = im.transpose(2,0,1)
		if self.center_cropped:
			return tr.center_crop(im, size).transpose(1,2,0)

		return tr.resize(im, size).transpose(1,2,0)

	def estimate_parts(self, im_obj: ImageWrapper) -> np.ndarray:
		im = im_obj.im_array

		xp = self.model.xp
		X = xp.array([self.prepare(im)]) # CNN input
		I = self._transform(im) # image transformed same as the CNN input

		with chainer.using_config("train", False):
			grad = ImageGradient(self.model, X)(cs=self.cs)
		saliency = chainer.cuda.to_cpu(grad2saliency(grad))

		boxes = self.extractor(I, saliency)
		boxes = [[int(_) for _ in (i,x,y,w,h)] for i, ((x,y), w, h) in boxes]

		if self._include_visualization:
			sal = self.extractor.corrector(saliency)
			centers, clusters = self.extractor.cluster_saliency(im, sal)
			self._visualization[im_obj.uuid] = sal, centers, clusters

		return np.array(boxes, dtype=np.int32)

	def set_parts(self, im_obj: ImageWrapper) -> ImageWrapper:
		i = im_obj.uuid
		parts = self.estimate_parts(im_obj)
		self._annot.set_parts(self.uuids[i], parts)
		return self.image_wrapped(i)


	def transform(self, im_obj) -> tuple:
		if len(im_obj.parts) == 0 or any([(-1 in p.as_annotation) for p in im_obj.parts]):
			im_obj = self.set_parts(im_obj)

			if self._include_visualization:
				vis = self._visualization[im_obj.uuid]
				return im_obj.as_tuple() + vis

		return im_obj.as_tuple()

	def prepare(self, im):
		if self._prepare is not None:
			return self._prepare(im)

		if self.center_cropped:
			size = self.model.meta.input_size
			return tr.center_crop(self.model.prepare(im, size=size), size)

		return self.models.prepare(im, keep_ratio=False)
