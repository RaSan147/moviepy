from typing import Any, TypeAlias, Union, TYPE_CHECKING


import numpy as _np
from numpy import ndarray as np_ndarray


try:
	import cupy as cnp
	# test if cupy is working
	cnp.array([1, 2, 3])
	cnp.asarray([1, 2, 3])
	cnp.zeros((3, 3))
	__X = cnp.ones((3, 3))
	if not isinstance(__X, cnp.ndarray):
		raise Exception("cupy not available")

	# rename cupy to np
	import cupy as np

	np_ndarray_instance = (np.ndarray, np_ndarray)


except Exception:
	cnp = None
	import numpy as np

	np_ndarray_instance = (np_ndarray)



def np_get(arr: np_ndarray) -> _np.ndarray:
	"""Get the array from cupy or numpy.

	Parameters
	----------
	arr : Union[np.ndarray, _np.ndarray]
		The array to get.

	Returns
	-------
	_np.ndarray
		The array from cupy to numpy.
	"""
	if isinstance(arr, list):
		return [np_get(a) for a in arr]
	if cnp and isinstance(arr, cnp.ndarray):
		# if 0-d array, return the value
		if arr.ndim == 0:
			return arr.get().item()
		return arr.get()
	else:
		return arr

def np_convert(arr: Any) -> Union[Any, np.ndarray]:
	"""Convert the array to cupy or numpy.
	
	Parameters
	----------
	arr : Any
		The array to convert.
	Returns
	-------
	Union[Any, np.ndarray]
		The array from cupy to numpy.
	"""
	if cnp and isinstance(arr, _np.ndarray):
		return cnp.asarray(arr)
	if cnp and isinstance(arr, list):
		return [cnp.array(a) for a in arr]
	
	return arr

if __name__ == "__main__":
	# Test the np_get function
	if cnp is None:
		print("cupy not available")
	else:
		arr = cnp.array([1, 2, 3])
		print(np_get(arr))  # Should print: [1 2 3]

	# check isinstance
	arr = np.array([1, 2, 3])
	print(isinstance(arr, np.ndarray))  # Should print: True
	# check isinstance
	arr = np.array([1, 2, 3])
	print(isinstance(arr, np_ndarray_instance))  # Should print: True
	# check isinstance
	arr = _np.array([1, 2, 3])
	print(isinstance(arr, np_ndarray_instance))  # Should print: True

