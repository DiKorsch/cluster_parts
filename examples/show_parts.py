#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import numpy as np

from chainercv import transforms as tr
from cvargparse import Arg
from cvargparse import GPUParser
from cvdatasets import FileListAnnotations
from cvdatasets.dataset.image import Size
from cvdatasets.utils import new_iterator
from cvmodelz.models import ModelFactory
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from cluster_parts import core
from cluster_parts import utils
from cluster_parts.shortcuts import CSPartsDataset


def new_model(args, fc_params):

	weights = np.load(args.load)
	n_classes_found = False
	for key in fc_params:
		try:
			n_classes = weights[f"{args.load_path}{key}"].shape[0]
			n_classes_found = True
			break
		except KeyError as e:
			pass

	if not n_classes_found:
		raise KeyError("Could not find number of classes!")

	model = ModelFactory.new(args.model_type, input_size=Size(args.input_size))
	logging.info(f"Created {model.__class__.__name__} for {n_classes} classes")
	model.load_for_inference(args.load, n_classes,
		path=args.load_path, strict=args.load_strict)
	logging.info(f"Loaded weights from {args.load}")


	device = chainer.get_device(args.gpu[0])
	device.use()
	model.to_device(device)
	logging.info(f"Using {device}")

	return model


def main(args):

	model = new_model(args,
		fc_params=[
			"fc/b",
			"fc8/b",
			"fc6/b",
			"wrapped/output/fc/b",
			"wrapped/output/fc2/b",
		]
	)

	extractor = core.BoundingBoxPartExtractor(
		corrector=core.Corrector(gamma=args.gamma, sigma=args.sigma),

		K=args.n_parts,
		fit_object=args.fit_object,

		thresh_type=args.thresh_type,
		cluster_init=utils.ClusterInitType.MAXIMAS,
		feature_composition=args.feature_composition,

	)

	annot = FileListAnnotations(root_or_infofile=args.data)

	train, test = annot.new_train_test_datasets(
		dataset_cls=CSPartsDataset,
		model=model.copy(mode="copy"),
		extractor=extractor,
		cs=True,
		include_visualization=True,
	)
	# n_classes = get_n_classes(args.load)
	# n_classes = len(np.unique(annot._labels)) + args.label_shift

	logging.info(f"Loaded {len(train)} training and {len(test)} test images")

	input_size = tuple(model.meta.input_size)

	with chainer.using_config("train", False):
		for ds in [test, train]:

			it, n_batches = new_iterator(ds, args.n_jobs,
				args.batch_size, repeat=False, shuffle=args.shuffle,
				use_threads=True)

			for i, batch in tqdm(enumerate(it), total=n_batches):

				for im, parts, y, *vis in batch:
					# continue
					fig, axs = plt.subplots(1, 2, figsize=(16,9))

					axs[0].imshow(im)
					axs[0].set_title(f"class: {y}")

					for part in parts:
						box = utils.box_rescaled(im, part, input_size,
							center_cropped=ds.center_cropped)
						axs[0].add_patch(plt.Rectangle(*box, fill=False))

					if not vis:
						continue

					sal, centers, clusters = vis
					im = ds._transform(im)
					_im = tr.resize(im.transpose(2,0,1), sal.shape[:2]).transpose(1,2,0)
					axs[1].imshow(_im, alpha=0.8)
					axs[1].imshow(clusters, cmap=plt.cm.viridis, alpha=0.6)
					axs[1].imshow(sal, cmap=plt.cm.viridis, alpha=0.6)

					for part in parts:
						box = utils.box_rescaled(sal[..., None], part, input_size,
							center_cropped=ds.center_cropped)
						axs[1].add_patch(plt.Rectangle(*box, fill=False))

					plt.show()
					plt.close()


def parse_args():
	args = GPUParser()
	args.add_args([
		Arg("data"),

		Arg.int("--label_shift", "-ls", default=0,
			help="apply label shift"),

		Arg.int("--input_size", default=0,
			help="overrides default input size of the model, if greater than 0"),

		Arg.int("--batch_size", default=32),
		Arg.int("--n_jobs", "-j", default=4),
		Arg.flag("--shuffle")
	], group_name="Data arguments")


	args.add_args([

		Arg("--model_type", "-mt",
			required=True,
			choices=ModelFactory.get_models(["cvmodelz"]),
			help="type of the model"),

		Arg("--load",
			required=True,
			help="ignore weights and load already fine-tuned model (classifier will NOT be re-initialized and number of classes will be unchanged)"),

		Arg("--load_path", default="",
			help="load path within the weights archive"),

		Arg.flag("--load_strict",
			help="load weights in a strict mode"),
	], group_name="Model arguments")

	args.add_args([
		Arg.int("--n_parts", "-n", default=4,
			help="Number of parts to detect"),

		Arg.flag("--fit_object",
			help="Use the entire saliency as an extra part"),

		Arg.float("--sigma", default=5.0,
			help="Saliency correction parameter: Gaussian smoothing"),

		Arg.float("--gamma", default=0.7,
			help="Saliency correction parameter: gamma correction"),

		utils.ThresholdType.as_arg("thresh_type",
			help_text="type of gradient thresholding"),

		utils.FeatureType.as_arg("feature_composition",
			nargs="+", default=utils.FeatureComposition.Default,
			help_text="composition of features"),
	], group_name="Part configs")

	return args.parse_args()


main(parse_args())
