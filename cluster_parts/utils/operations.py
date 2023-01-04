import numpy as np

from cvdatasets.utils import rescale


def l2_norm(arr, axis=2, keepdims=False, xp=np):
	if arr.ndim == 2:
		arr = xp.expand_dims(arr, axis=axis)

	return xp.sqrt(xp.sum(arr**2, axis=axis, keepdims=keepdims))

def normalize(arr, axis=2, channel_wise=False):

	if arr.ndim == 3 and channel_wise:
		non_chan_axis = tuple([i for i in range(arr.ndim) if i != axis])
		arr -= arr.min(axis=non_chan_axis, keepdims=True)

		_max_vals = arr.max(axis=non_chan_axis, keepdims=True)
		mask = (_max_vals != 0).squeeze()
		if mask.any():
			arr[mask] /= _max_vals[mask]
		return arr

	arr -= arr.min()
	if arr.max() != 0:
		arr /= arr.max()

	return arr

def grad2saliency(grad, *, axis=1):
	if grad.ndim == 3:
		grad = grad[None]

	grad = abs(grad)
	other_axes = [a for a in range(1, 4) if a != axis]
	grad = grad - grad.min(axis=other_axes, keepdims=True)
	max_grad = grad.max(axis=other_axes, keepdims=True)
	max_grad[max_grad == 0] = 1
	grad = grad / max_grad

	return grad.mean(axis=axis)

def box_rescaled(im, part, init_size, *, center_cropped: bool):
	xywh = np.array(part.as_annotation[1:], dtype=np.int32)

	x, y = rescale(im, xywh[:2], init_size, center_cropped=center_cropped)
	w, h = rescale(im, xywh[2:], init_size, center_cropped=center_cropped)

	return (x, y), w, h
