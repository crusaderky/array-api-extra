"""Pytest fixtures."""

from collections.abc import Callable, Generator
from functools import partial, wraps
from types import ModuleType
from typing import ParamSpec, TypeVar, cast

import numpy as np
import pytest

from array_api_extra._lib._backends import Backend
from array_api_extra._lib._testing import xfail
from array_api_extra._lib._utils._compat import array_namespace
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._typing import Device
from array_api_extra.testing import patch_lazy_xp_functions

T = TypeVar("T")
P = ParamSpec("P")

np_compat = array_namespace(np.empty(0))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


@pytest.fixture(params=[b.pytest_param() for b in Backend])
def library(request: pytest.FixtureRequest) -> Backend:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current Backend enum.
    """
    elem = cast(Backend, request.param)

    for marker_name, skip_or_xfail, allow_kwargs in (
        ("skip_xp_backend", pytest.skip, {"reason"}),
        ("xfail_xp_backend", partial(xfail, request), {"reason", "strict"}),
    ):
        for marker in request.node.iter_markers(marker_name):
            if len(marker.args) != 1:
                msg = f"Expected exactly one positional argument; got {marker.args}"
                raise TypeError(msg)
            if not isinstance(marker.args[0], Backend):
                msg = f"Argument of {marker_name} must be a Backend enum"
                raise TypeError(msg)
            if invalid_kwargs := set(marker.kwargs) - allow_kwargs:
                msg = f"Unexpected kwarg(s): {invalid_kwargs}"
                raise TypeError(msg)

            library: Backend = marker.args[0]
            reason: str | None = marker.kwargs.get("reason", None)
            strict: bool | None = marker.kwargs.get("strict", None)

            if library == elem:
                reason = f"{library}: {reason}" if reason else str(library)
                kwargs = {"strict": strict} if strict is not None else {}
                skip_or_xfail(reason=reason, **kwargs)

    return elem


class NumPyReadOnly:
    """
    Variant of array_api_compat.numpy producing read-only arrays.

    Read-only NumPy arrays fail on `__iadd__` etc., whereas read-only libraries such as
    JAX and Sparse simply don't define those methods, which makes calls to `+=` fall
    back to `__add__`.

    Note that this is not a full read-only Array API library. Notably,
    `array_namespace(x)` returns array_api_compat.numpy. This is actually the desired
    behaviour, so that when a tested function internally calls `xp =
    array_namespace(*args) or xp`, it will internally create writeable arrays.
    For this reason, tests that explicitly pass xp=xp to the tested functions may
    misbehave and should be skipped for NUMPY_READONLY.
    """

    def __getattr__(self, name: str) -> object:  # numpydoc ignore=PR01,RT01
        """Wrap all functions that return arrays to make their output read-only."""
        func = getattr(np_compat, name)
        if not callable(func) or isinstance(func, type):
            return func
        return self._wrap(func)

    @staticmethod
    def _wrap(func: Callable[P, T]) -> Callable[P, T]:  # numpydoc ignore=PR01,RT01
        """Wrap func to make all np.ndarrays it returns read-only."""

        def as_readonly(o: T) -> T:  # numpydoc ignore=PR01,RT01
            """Unset the writeable flag in o."""
            try:
                # Don't use is_numpy_array(o), as it includes np.generic
                if isinstance(o, np.ndarray):
                    o.flags.writeable = False
            except TypeError:
                # Cannot interpret as a data type
                return o

            # This works with namedtuples too
            if isinstance(o, tuple | list):
                return type(o)(*(as_readonly(i) for i in o))  # type: ignore[arg-type,return-value] # pyright: ignore[reportArgumentType]

            return o

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # numpydoc ignore=GL08
            return as_readonly(func(*args, **kwargs))

        return wrapper


@pytest.fixture
def xp(
    library: Backend, request: pytest.FixtureRequest
) -> Generator[ModuleType]:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current array namespace.
    """
    if library == Backend.NUMPY_READONLY:
        yield NumPyReadOnly()  # type: ignore[misc]  # pyright: ignore[reportReturnType]
        return

    xp = pytest.importorskip(library.modname)
    # Possibly wrap module with array_api_compat
    xp = array_namespace(xp.empty(0))

    if library == Backend.ARRAY_API_STRICTEST:
        with xp.ArrayAPIStrictFlags(
            boolean_indexing=False,
            data_dependent_shapes=False,
            # writeable=False,  # TODO implement in array-api-strict
            # lazy=True,  # TODO implement in array-api-strict
            enabled_extensions=(),
        ):
            yield xp
        return

    if library.like(Backend.JAX):
        _setup_jax(library)
    elif library.like(Backend.TORCH):
        _setup_torch(library)

    # On Dask and JAX, monkey-patch all functions tagged by `lazy_xp_function`
    # in the global scope of the module containing the test function.
    with patch_lazy_xp_functions(request, xp=xp):
        yield xp


def _setup_jax(library: Backend) -> None:
    import jax

    # suppress unused-ignore to run mypy in -e lint as well as -e dev
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call,unused-ignore]

    if library == Backend.JAX_GPU:
        try:
            device = jax.devices("cuda")[0]
        except RuntimeError:
            pytest.skip("no CUDA device available")
    else:
        device = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", device)


def _setup_torch(library: Backend) -> None:
    import torch

    # This is already the default, but some tests or env variables may change it.
    # TODO test both float32 and float64, like in scipy.
    torch.set_default_dtype(torch.float32)

    if library == Backend.TORCH_GPU:
        import torch.cuda

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device available")
        torch.set_default_device("cuda")
    else:
        assert library == Backend.TORCH
        torch.set_default_device("cpu")


# Can select the test with `pytest -k dask`
@pytest.fixture(params=[Backend.DASK.pytest_param()])
def da(
    request: pytest.FixtureRequest,
) -> Generator[ModuleType]:  # numpydoc ignore=PR01,RT01
    """Variant of the `xp` fixture that only yields dask.array."""
    xp = pytest.importorskip("dask.array")
    xp = array_namespace(xp.empty(0))
    with patch_lazy_xp_functions(request, xp=xp):
        yield xp


@pytest.fixture(params=[Backend.JAX.pytest_param(), Backend.JAX_GPU.pytest_param()])
def jnp(
    request: pytest.FixtureRequest,
) -> Generator[ModuleType]:  # numpydoc ignore=PR01,RT01
    """Variant of the `xp` fixture that only yields jax.numpy."""
    xp = pytest.importorskip("jax.numpy")
    _setup_jax(request.param)
    with patch_lazy_xp_functions(request, xp=xp):
        yield xp


@pytest.fixture(params=[Backend.TORCH, Backend.TORCH_GPU])
def torch(request: pytest.FixtureRequest) -> ModuleType:  # numpydoc ignore=PR01,RT01
    """Variant of the `xp` fixture that only yields torch."""
    xp = pytest.importorskip("torch")
    xp = array_namespace(xp.empty(0))
    _setup_torch(request.param)
    return xp


@pytest.fixture
def device(
    library: Backend, xp: ModuleType
) -> Device:  # numpydoc ignore=PR01,RT01,RT03
    """
    Return a valid device for the backend.

    Where possible, return a device that is not the default one.
    """
    if library == Backend.ARRAY_API_STRICT:
        return xp.Device("device1")
    if library == Backend.TORCH:
        return xp.device("meta")
    if library == Backend.TORCH_GPU:
        return xp.device("cpu")
    return get_device(xp.empty(0))
