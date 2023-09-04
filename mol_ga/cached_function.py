""" Code for functions which cache inputs for efficiency. """
from __future__ import annotations

from typing import Callable, Optional


class CachedFunction:
    """Function which caches previously computed values to avoid repeat computation."""

    def __init__(
        self,
        f: Callable,
        cache: Optional[dict] = None,
        transform: Optional[Callable] = None,
        max_cache_size: Optional[int] = None,
    ):
        """Init function

        :param f: The function to cache
        :type f: callable
        :param cache: dict mapping known inputs-> outputs of f, defaults to None
        :type cache: dict, optional
        :param transform: optional transform function to apply to values
            of f (e.g. scaling output to be in [0, 1]), defaults to None
        :type transform: callable, optional
        """
        self._f = f
        self.cache = cache or dict()
        self.transform = transform
        self.max_cache_size = max_cache_size
        self._trim_cache()

    def _batch_f_eval(self, input_list):
        return [self._f(x) for x in input_list]

    def _batch_transform(self, output_list):
        if self.transform is None:
            return output_list
        else:
            return [self.transform(x) for x in output_list]

    def _trim_cache(self) -> None:
        """Clears the cache if it is too large."""
        if self.max_cache_size is not None and len(self.cache) > self.max_cache_size:
            self.cache.clear()

    def eval_batch(self, inputs):
        # Eval function at non-cached inputs
        inputs_not_cached = [x for x in inputs if x not in self.cache]
        outputs_not_cached = self._batch_f_eval(inputs_not_cached)

        # Add new values to cache
        for x, y in zip(inputs_not_cached, outputs_not_cached):
            self.cache[x] = y

        # Get and transform outputs
        outputs = [self.cache[x] for x in inputs]
        outputs = self._batch_transform(outputs)
        return outputs

    def eval_non_batch(self, inputs):
        inputs = [inputs]
        outputs = self.eval_batch(inputs)
        assert len(outputs) == 1
        outputs = outputs[0]
        return outputs

    def __call__(self, inputs, batch=True):
        # Ensure it is in batch form
        if batch:
            output = self.eval_batch(inputs)
        else:
            output = self.eval_non_batch(inputs)

        self._trim_cache()
        return output


class CachedBatchFunction(CachedFunction):
    """
    Special kind of cached function where f takes a batch of inputs
    instead of a single input. The input to the underlying function
    (and transform function) will always be a batch of data.
    """

    def _batch_f_eval(self, input_list):
        return self._f(input_list)
