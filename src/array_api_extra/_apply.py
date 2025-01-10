"""Public API Functions."""

# https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from ._lib._compat import (
    array_namespace,
    is_dask_namespace,
    is_jax_namespace,
)
from ._lib._typing import Array, DType

if TYPE_CHECKING:
    # https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
    from typing import TypeAlias

    import numpy as np
    import numpy.typing as npt

    NumPyObject: TypeAlias = npt.NDArray[DType] | np.generic  # type: ignore[no-any-explicit]


def apply_numpy_func(  # type: ignore[no-any-explicit]
    func: Callable[..., NumPyObject | Sequence[NumPyObject]],
    *args: Array,
    shapes: Sequence[tuple[int, ...]] | None = None,
    dtypes: Sequence[DType] | None = None,
    xp: ModuleType | None = None,
    input_indices: Sequence[Sequence[Hashable]] | None = None,
    core_indices: Sequence[Hashable] | None = None,
    output_indices: Sequence[Sequence[Hashable]] | None = None,
    adjust_chunks: Sequence[dict[Hashable, Callable[[int], int]]] | None = None,
    new_axes: Sequence[dict[Hashable, int]] | None = None,
    **kwargs: Any,
) -> tuple[Array, ...]:
    """
    Apply a function that operates on NumPy arrays to Array API compliant arrays.

    Parameters
    ----------
    func : callable
        The function to apply. It must accept one or more NumPy arrays or generics as
        positional arguments and return either a single NumPy array or generic, or a
        tuple or list thereof.

        It must be a pure function, i.e. without side effects such as disk output,
        as depending on the backend it may be executed more than once.
    *args : Array
        One or more Array API compliant arrays. You need to be able to apply
        ``np.asarray()`` to them to convert them to numpy; read notes below about
        specific backends.
    shapes : Sequence[tuple[int, ...]], optional
        Sequence of output shapes, one for each output of `func`.
        If `func` returns a single (non-sequence) output, this must be a sequence
        with a single element.
        Default: assume a single output and broadcast shapes of the input arrays.
    dtypes : Sequence[DType], optional
        Sequence of output dtypes, one for each output of `func`.
        If `func` returns a single (non-sequence) output, this must be a sequence
        with a single element.
        Default: infer the result type(s) from the input arrays.
    xp : array_namespace, optional
        The standard-compatible namespace for `args`. Default: infer.
    input_indices : Sequence[Sequence[Hashable]], optional
        Dask specific.
        Axes labels for each input array, e.g. if there are two args with respectively
        ndim=3 and 1, `input_indices` could be ``['ijk', 'j']`` or ``[(0, 1, 2),
        (1,)]``.
        Default: disallow Dask.
    core_indices : Sequence[Hashable], optional
        **Dask specific.**
        Axes of the input arrays that cannot be broken into chunks.
        Default: disallow Dask.
    output_indices : Sequence[Sequence[Hashable]], optional
        **Dask specific.**
        Axes labels for each output array. If `func` returns a single (non-sequence)
        output, this must be a sequence containing a single sequence of labels, e.g.
        ``['ijk']``.
        Default: disallow Dask.
    adjust_chunks : Sequence[Mapping[Hashable, Callable[[int], int]]], optional
        **Dask specific.**
        Sequence of dicts, one per output, mapping index to function to be applied to
        each chunk to determine the output size. The total must add up to the output
        shape.
        Default: on Dask, the size along each index cannot change.
    new_axes : Sequence[Mapping[Hashable, int]], optional
        **Dask specific.**
        New indexes and their dimension lengths, one per output.
        Default: on Dask, there can't be `output_indices` that don't appear in
        `input_indices`.
    **kwargs : Any, optional
        Additional keyword arguments to pass verbatim to `func`.
        Any array objects in them won't be converted to NumPy.

    Returns
    -------
    tuple[Array, ...]
        The result(s) of `func` applied to the input arrays.
        This is always a tuple, even if `func` returns a single output.

    Notes
    -----
    JAX
        This allows applying eager functions to jitted JAX arrays, which are lazy.
        The function won't be applied until the JAX array is materialized.

        The `JAX transfer guard
        <https://jax.readthedocs.io/en/latest/transfer_guard.html>`_
        may prevent arrays on a GPU device from being transferred back to CPU.
        This is treated as an implicit transfer.

    PyTorch, CuPy
        These backends raise by default if you attempt to convert arrays on a GPU device
        to NumPy.

    Sparse
        By default, sparse prevents implicit densification through ``np.asarray`.
        `This safety mechanism can be disabled
        <https://sparse.pydata.org/en/stable/operations.html#package-configuration>`_.

    Dask
        This allows applying eager functions to the individual chunks of dask arrays.
        The dask graph won't be computed. As a special limitation, `func` must return
        exactly one output.

        In order to enable running on Dask you need to specify at least
        `input_indices`, `output_indices`, and `core_indices`, but you may also need
        `adjust_chunks` and `new_axes` depending on the function.

        Read `dask.array.blockwise`:
        - ``input_indices`` map to the even ``*args`` of `dask.array.blockwise`
        - ``output_indices[0]`` maps to the ``out_ind`` parameter
        - ``adjust_chunks[0]`` maps to the ``adjust_chunks`` parameter
        - ``new_axes[0]`` maps to the ``new_axes`` parameter

        ``core_indices`` is a safety measure to prevent incorrect results on
        Dask along chunked axes. Consider this::

            >>> apply_numpy_func(lambda x: x + x.sum(axis=0), x,
            ...                  input_indices=['ij'], output_indices=['ij'])

        The above example would produce incorrect results if x is a dask array with more
        than one chunk along axis 0, as each chunk will calculate its own local
        subtotal. To prevent this, we need to declare the first axis of ``args[0]`` as a
        *core axis*::

            >>> apply_numpy_func(lambda x: x + x.sum(axis=0), x,
            ...                  input_indices=['ij'], output_indices=['ij'],
            ...                  core_indices='i')

        This will cause `apply_numpy_func` to raise if the first axis of `x` is broken
        along multiple chunks, thus forcing the final user to rechunk ahead of time:

            >>> x = x.chunk({0: -1})

        This needs to always be a conscious decision on behalf of the final user, as the
        new chunks will be larger than the old and may cause memory issues, unless chunk
        size is reduced along a different, non-core axis.
    """
    if xp is None:
        xp = array_namespace(*args)
    if shapes is None:
        shapes = [xp.broadcast_shapes(*(arg.shape for arg in args))]
    if dtypes is None:
        dtypes = [xp.result_type(*args)] * len(shapes)

    if len(shapes) != len(dtypes):
        msg = f"got {len(shapes)} shapes and {len(dtypes)} dtypes"
        raise ValueError(msg)
    if len(shapes) == 0:
        msg = "Must have at least one output array"
        raise ValueError(msg)

    if is_dask_namespace(xp):
        # General validation
        if len(shapes) > 1:
            msg = "dask.array.map_blocks() does not support multiple outputs"
            raise NotImplementedError(msg)
        if input_indices is None or output_indices is None or core_indices is None:
            msg = (
                "Dask is disallowed unless one declares input_indices, "
                "output_indices, and core_indices"
            )
            raise ValueError(msg)
        if len(input_indices) != len(args):
            msg = f"got {len(input_indices)} input_indices and {len(args)} args"
            raise ValueError(msg)
        if len(output_indices) != len(shapes):
            msg = f"got {len(output_indices)} input_indices and {len(shapes)} shapes"
            raise NotImplementedError(msg)
        if isinstance(adjust_chunks, Mapping):
            msg = "adjust_chunks must be a sequence of mappings"
            raise ValueError(msg)
        if adjust_chunks is not None and len(adjust_chunks) != len(shapes):
            msg = f"got {len(adjust_chunks)} adjust_chunks and {len(shapes)} shapes"
            raise ValueError(msg)
        if isinstance(new_axes, Mapping):
            msg = "new_axes must be a sequence of mappings"
            raise ValueError(msg)
        if new_axes is not None and len(new_axes) != len(shapes):
            msg = f"got {len(new_axes)} new_axes and {len(shapes)} shapes"
            raise ValueError(msg)

        # core_indices validation
        for inp_idx, arg in zip(input_indices, args, strict=True):
            for i, chunks in zip(inp_idx, arg.chunks, strict=True):
                if i in core_indices and len(chunks) > 1:
                    msg = f"Core index {i} is broken into multiple chunks"
                    raise ValueError(msg)

        meta_xp = array_namespace(*(getattr(arg, "meta", None) for arg in args))
        wrapped = _npfunc_single_output_wrapper(func, meta_xp)
        dask_args = []
        for arg, inp_idx in zip(args, input_indices, strict=True):
            dask_args += [arg, inp_idx]

        out = xp.blockwise(
            wrapped,
            output_indices[0],
            *dask_args,
            dtype=dtypes[0],
            adjust_chunks=adjust_chunks[0] if adjust_chunks is not None else None,
            new_axes=new_axes[0] if new_axes is not None else None,
            **kwargs,
        )
        if out.shape != shapes[0]:
            msg = f"expected shape {shapes[0]}, but got {out.shape} from indices"
            raise ValueError(msg)
        return (out,)

    wrapped = _npfunc_tuple_output_wrapper(func, xp)
    if is_jax_namespace(xp):
        # If we're inside jax.jit, we can't eagerly convert
        # the JAX tracer objects to numpy.
        # Instead, we delay calling wrapped, which will receive
        # as arguments and will return JAX eager arrays.
        import jax  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error  # pyright: ignore[reportMissingImports]

        return cast(
            tuple[Array, ...],
            jax.pure_callback(
                wrapped,
                tuple(
                    jax.ShapeDtypeStruct(s, dt)  # pyright: ignore[reportUnknownArgumentType]
                    for s, dt in zip(shapes, dtypes, strict=True)
                ),
                *args,
                **kwargs,
            ),
        )

    # Eager backends
    out = wrapped(*args, **kwargs)

    # Output validation
    if len(out) != len(shapes):
        msg = f"func was declared to return {len(shapes)} outputs, got {len(out)}"
        raise ValueError(msg)
    for out_i, shape_i, dtype_i in zip(out, shapes, dtypes, strict=True):
        if out_i.shape != shape_i:
            msg = f"expected shape {shape_i}, got {out_i.shape}"
            raise ValueError(msg)
        if not xp.isdtype(out_i.dtype, dtype_i):
            msg = f"expected dtype {dtype_i}, got {out_i.dtype}"
            raise ValueError(msg)
    return out  # type: ignore[no-any-return]


def _npfunc_tuple_output_wrapper(  # type: ignore[no-any-explicit]  # numpydoc ignore=PR01,RT01
    func: Callable[..., NumPyObject | Sequence[NumPyObject]],
    xp: ModuleType,
) -> Callable[..., tuple[Array, ...]]:
    """
    Helper of `apply_numpy_func`.

    Given a function that accepts one or more numpy arrays as positional arguments and
    returns a single numpy array or a sequence of numpy arrays,
    return a function that accepts the same number of Array API arrays and always
    returns a tuple of Array API array.

    Any keyword arguments are passed through verbatim to the wrapped function.

    Raise if np.asarray() raises on any input. This typically happens if the input is
    lazy and has a guard against being implicitly turned into a NumPy array (e.g.
    densification for sparse arrays, device->host transfer for cupy and torch arrays).
    """

    @wraps(func)
    def wrapper(  # type: ignore[no-any-decorated,no-any-explicit]
        *args: Array, **kwargs: Any
    ) -> tuple[Array, ...]:  # numpydoc ignore=GL08
        import numpy as np  # pylint: disable=import-outside-toplevel

        args = tuple(np.asarray(arg) for arg in args)
        out = func(*args, **kwargs)

        if isinstance(out, np.ndarray | np.generic):
            out = (out,)
        elif not isinstance(out, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = (
                "apply_numpy_func: func must return a numpy object or a "
                f"sequence of numpy objects; got {out}"
            )
            raise TypeError(msg)

        return tuple(xp.asarray(o) for o in out)

    return wrapper


def _npfunc_single_output_wrapper(  # type: ignore[no-any-explicit]  # numpydoc ignore=PR01,RT01
    func: Callable[..., NumPyObject | Sequence[NumPyObject]],
    xp: ModuleType,
) -> Callable[..., Array]:
    """
    Dask-specific helper of `apply_numpy_func`.

    Variant of `_npfunc_tuple_output_wrapper`, to be used with Dask which, at the time
    of writing, does not support multiple outputs in `dask.array.blockwise`.

    func may return a single numpy object or a sequence with exactly one numpy object.
    The wrapper returns a single Array object, with no tuple wrapping.
    """

    # @wraps causes the generated dask key to contain the name of the wrapped function
    @wraps(func)
    def wrapper(  # type: ignore[no-any-decorated,no-any-explicit]  # numpydoc ignore=GL08
        *args: Array, **kwargs: Any
    ) -> Array:
        import numpy as np  # pylint: disable=import-outside-toplevel

        args = tuple(np.asarray(arg) for arg in args)
        out = func(*args, **kwargs)

        if not isinstance(out, np.ndarray | np.generic):
            if not isinstance(out, Sequence) or len(out) != 1:  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = (
                    "apply_numpy_func: func must return a single numpy object or a "
                    f"sequence with exactly one numpy object; got {out}"
                )
                raise ValueError(msg)
            out = out[0]

        return xp.asarray(out)

    return wrapper
