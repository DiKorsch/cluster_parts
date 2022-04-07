import numpy as np

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
