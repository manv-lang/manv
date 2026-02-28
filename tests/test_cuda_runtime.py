from __future__ import annotations

import ctypes
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.backends.cuda.runtime import CudaRuntime
from manv.gpu_dispatch import dispatch_kernel_ir
from manv.intrinsics import invoke_intrinsic


def _prov() -> dict[str, object]:
    return {
        "graph_node_id": "g1",
        "hlir_op_id": "main.i1",
        "source_span": {
            "uri": "runtime_test.mv",
            "start_line": 1,
            "start_col": 1,
            "end_line": 1,
            "end_col": 16,
        },
        "inline_chain": [],
    }


def _vector_add_kernel_ir() -> dict[str, object]:
    return {
        "version": "0.1",
        "source": "runtime_test.mv",
        "kernels": [
            {
                "kernel_name": "vec_add_i32",
                "signature": {
                    "params": [
                        {"index": 0, "name": "out", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                        {"index": 1, "name": "a", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                        {"index": 2, "name": "b", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                    ],
                    "return_policy": "void",
                },
                "launch_model": {"grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": 4, "block_y": 1, "block_z": 1, "shared_bytes": 0},
                "memory_regions": [
                    {"name": "out", "dtype": "i32", "space": "global"},
                    {"name": "a", "dtype": "i32", "space": "global"},
                    {"name": "b", "dtype": "i32", "space": "global"},
                ],
                "blocks": [
                    {
                        "id": "entry",
                        "ops": [
                            {"id": "t0", "opcode": "thread_id_x", "inputs": [], "outputs": ["t0"], "attrs": {}, "dtype": "i32", "memory_space": "private", "provenance": _prov()},
                            {"id": "a0", "opcode": "buffer_load", "inputs": ["t0"], "outputs": ["a0"], "attrs": {"buffer": "a"}, "dtype": "i32", "memory_space": "global", "provenance": _prov()},
                            {"id": "b0", "opcode": "buffer_load", "inputs": ["t0"], "outputs": ["b0"], "attrs": {"buffer": "b"}, "dtype": "i32", "memory_space": "global", "provenance": _prov()},
                            {"id": "s0", "opcode": "binary::+", "inputs": ["a0", "b0"], "outputs": ["s0"], "attrs": {}, "dtype": "i32", "memory_space": "private", "provenance": _prov()},
                            {"id": "w0", "opcode": "buffer_store", "inputs": ["t0", "s0"], "outputs": [], "attrs": {"buffer": "out"}, "dtype": "void", "memory_space": "global", "provenance": _prov()},
                        ],
                        "terminator": "ret",
                    }
                ],
            }
        ],
    }


def _saxpy_kernel_ir() -> dict[str, object]:
    return {
        "version": "0.1",
        "source": "runtime_test.mv",
        "kernels": [
            {
                "kernel_name": "saxpy_i32",
                "signature": {
                    "params": [
                        {"index": 0, "name": "out", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                        {"index": 1, "name": "a", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                        {"index": 2, "name": "alpha", "kind": "scalar", "dtype": "i32", "by_ref": False, "alignment": 4, "address_space": "private"},
                    ],
                    "return_policy": "void",
                },
                "launch_model": {"grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": 4, "block_y": 1, "block_z": 1, "shared_bytes": 0},
                "memory_regions": [
                    {"name": "out", "dtype": "i32", "space": "global"},
                    {"name": "a", "dtype": "i32", "space": "global"},
                ],
                "blocks": [
                    {
                        "id": "entry",
                        "ops": [
                            {"id": "t0", "opcode": "thread_id_x", "inputs": [], "outputs": ["t0"], "attrs": {}, "dtype": "i32", "memory_space": "private", "provenance": _prov()},
                            {"id": "a0", "opcode": "buffer_load", "inputs": ["t0"], "outputs": ["a0"], "attrs": {"buffer": "a"}, "dtype": "i32", "memory_space": "global", "provenance": _prov()},
                            {"id": "s0", "opcode": "binary::*", "inputs": ["a0", "alpha"], "outputs": ["s0"], "attrs": {}, "dtype": "i32", "memory_space": "private", "provenance": _prov()},
                            {"id": "w0", "opcode": "buffer_store", "inputs": ["t0", "s0"], "outputs": [], "attrs": {"buffer": "out"}, "dtype": "void", "memory_space": "global", "provenance": _prov()},
                        ],
                        "terminator": "ret",
                    }
                ],
            }
        ],
    }


def _reduction_kernel_ir() -> dict[str, object]:
    return {
        "version": "0.1",
        "source": "runtime_test.mv",
        "kernels": [
            {
                "kernel_name": "reduce_partial",
                "signature": {
                    "params": [
                        {"index": 0, "name": "__partial", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                        {"index": 1, "name": "a", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                    ],
                    "return_policy": "void",
                },
                "launch_model": {"grid_x": 2, "grid_y": 1, "grid_z": 1, "block_x": 4, "block_y": 1, "block_z": 1, "shared_bytes": 0},
                "memory_regions": [
                    {"name": "__partial", "dtype": "i32", "space": "global"},
                    {"name": "a", "dtype": "i32", "space": "global"},
                ],
                "debug_meta": {
                    "kernel_kind": "reduction_partial",
                    "reduction_value": "partial_load",
                    "output_buffer": "__partial",
                    "value_dtype": "i32",
                },
                "blocks": [
                    {
                        "id": "entry",
                        "ops": [
                            {"id": "tid0", "opcode": "thread_id_x", "inputs": [], "outputs": ["tid0"], "attrs": {}, "dtype": "i32", "memory_space": "private", "provenance": _prov()},
                            {"id": "partial_load", "opcode": "buffer_load", "inputs": ["tid0"], "outputs": ["partial_load"], "attrs": {"buffer": "a"}, "dtype": "i32", "memory_space": "global", "provenance": _prov()},
                        ],
                        "terminator": "ret",
                    }
                ],
            },
            {
                "kernel_name": "reduce_finalize",
                "signature": {
                    "params": [
                        {"index": 0, "name": "__out", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                        {"index": 1, "name": "__partial", "kind": "buffer", "dtype": "i32", "by_ref": True, "alignment": 4, "address_space": "global"},
                    ],
                    "return_policy": "void",
                },
                "launch_model": {"grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": 2, "block_y": 1, "block_z": 1, "shared_bytes": 0},
                "memory_regions": [
                    {"name": "__out", "dtype": "i32", "space": "global"},
                    {"name": "__partial", "dtype": "i32", "space": "global"},
                ],
                "debug_meta": {
                    "kernel_kind": "reduction_finalize",
                    "reduction_value": "partial_load",
                    "output_buffer": "__out",
                    "value_dtype": "i32",
                },
                "blocks": [
                    {
                        "id": "entry",
                        "ops": [
                            {"id": "tid0", "opcode": "thread_id_x", "inputs": [], "outputs": ["tid0"], "attrs": {}, "dtype": "i32", "memory_space": "private", "provenance": _prov()},
                            {"id": "partial_load", "opcode": "buffer_load", "inputs": ["tid0"], "outputs": ["partial_load"], "attrs": {"buffer": "__partial"}, "dtype": "i32", "memory_space": "global", "provenance": _prov()},
                        ],
                        "terminator": "ret",
                    }
                ],
            },
        ],
    }


def _compiled_bundle() -> SimpleNamespace:
    # The runtime tests focus on driver interaction, not NVRTC. Supplying a
    # prebuilt bundle keeps the test deterministic and hardware-independent.
    return SimpleNamespace(
        binaries={"ptx": "// fake ptx", "cuda_cpp": "// fake cuda"},
        reflection={"arch": "sm_80"},
        cache_key="fake-cache-key",
    )


class _FakeFn:
    def __init__(self, impl):
        self.impl = impl
        self.argtypes = None

    def __call__(self, *args):
        return self.impl(*args)


class FakeCudaDriver:
    def __init__(self, *, fail_launch: bool = False) -> None:
        self.fail_launch = fail_launch
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.allocations: dict[int, bytearray] = {}
        self._next_ptr = 0x1000
        self._function_names: dict[int, str] = {}
        self._next_function = 0x2000

        self.cuInit = _FakeFn(self._cuInit)
        self.cuDeviceGetCount = _FakeFn(self._cuDeviceGetCount)
        self.cuDeviceGet = _FakeFn(self._cuDeviceGet)
        self.cuCtxCreate_v2 = _FakeFn(self._cuCtxCreate_v2)
        self.cuCtxDestroy_v2 = _FakeFn(self._cuCtxDestroy_v2)
        self.cuCtxSynchronize = _FakeFn(self._cuCtxSynchronize)
        self.cuDriverGetVersion = _FakeFn(self._cuDriverGetVersion)
        self.cuModuleLoadDataEx = _FakeFn(self._cuModuleLoadDataEx)
        self.cuModuleGetFunction = _FakeFn(self._cuModuleGetFunction)
        self.cuModuleUnload = _FakeFn(self._cuModuleUnload)
        self.cuMemAlloc_v2 = _FakeFn(self._cuMemAlloc_v2)
        self.cuMemFree_v2 = _FakeFn(self._cuMemFree_v2)
        self.cuMemcpyHtoD_v2 = _FakeFn(self._cuMemcpyHtoD_v2)
        self.cuMemcpyDtoH_v2 = _FakeFn(self._cuMemcpyDtoH_v2)
        self.cuMemcpyDtoD_v2 = _FakeFn(self._cuMemcpyDtoD_v2)
        self.cuLaunchKernel = _FakeFn(self._cuLaunchKernel)
        self.cuGetErrorName = _FakeFn(self._cuGetErrorName)
        self.cuGetErrorString = _FakeFn(self._cuGetErrorString)

    def _record(self, name: str, *args: object) -> None:
        self.calls.append((name, args))

    def _cuInit(self, flags: int) -> int:
        self._record("cuInit", flags)
        return 0

    def _cuDeviceGetCount(self, out_ptr: object) -> int:
        self._record("cuDeviceGetCount")
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = 1
        return 0

    def _cuDeviceGet(self, out_ptr: object, ordinal: int) -> int:
        self._record("cuDeviceGet", ordinal)
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = int(ordinal)
        return 0

    def _cuCtxCreate_v2(self, out_ptr: object, flags: int, device: object) -> int:
        self._record("cuCtxCreate_v2", flags, getattr(device, "value", device))
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0xCAFE
        return 0

    def _cuCtxDestroy_v2(self, context: object) -> int:
        self._record("cuCtxDestroy_v2", getattr(context, "value", context))
        return 0

    def _cuCtxSynchronize(self) -> int:
        self._record("cuCtxSynchronize")
        return 0

    def _cuDriverGetVersion(self, out_ptr: object) -> int:
        self._record("cuDriverGetVersion")
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = 12040
        return 0

    def _cuModuleLoadDataEx(self, out_ptr: object, ptx_ptr: object, *_rest: object) -> int:
        self._record("cuModuleLoadDataEx", ctypes.cast(ptx_ptr, ctypes.c_char_p).value)
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0xBEEF
        return 0

    def _cuModuleGetFunction(self, out_ptr: object, module: object, name: bytes) -> int:
        self._record("cuModuleGetFunction", getattr(module, "value", module), name)
        handle = self._next_function
        self._next_function += 0x100
        self._function_names[handle] = name.decode("utf-8")
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value = handle
        return 0

    def _cuModuleUnload(self, module: object) -> int:
        self._record("cuModuleUnload", getattr(module, "value", module))
        return 0

    def _cuMemAlloc_v2(self, out_ptr: object, nbytes: int) -> int:
        self._record("cuMemAlloc_v2", nbytes)
        pointer = self._next_ptr
        self._next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[pointer] = bytearray(int(nbytes))
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_uint64)).contents.value = pointer
        return 0

    def _cuMemFree_v2(self, device_ptr: object) -> int:
        ptr = int(getattr(device_ptr, "value", device_ptr))
        self._record("cuMemFree_v2", ptr)
        self.allocations.pop(ptr, None)
        return 0

    def _cuMemcpyHtoD_v2(self, device_ptr: object, src_ptr: int, nbytes: int) -> int:
        ptr = int(getattr(device_ptr, "value", device_ptr))
        self._record("cuMemcpyHtoD_v2", ptr, nbytes)
        self.allocations[ptr][: int(nbytes)] = ctypes.string_at(src_ptr, int(nbytes))
        return 0

    def _cuMemcpyDtoH_v2(self, dst_ptr: int, device_ptr: object, nbytes: int) -> int:
        ptr = int(getattr(device_ptr, "value", device_ptr))
        self._record("cuMemcpyDtoH_v2", ptr, nbytes)
        ctypes.memmove(dst_ptr, bytes(self.allocations[ptr][: int(nbytes)]), int(nbytes))
        return 0

    def _cuMemcpyDtoD_v2(self, dst_ptr: object, src_ptr: object, nbytes: int) -> int:
        dst = int(getattr(dst_ptr, "value", dst_ptr))
        src = int(getattr(src_ptr, "value", src_ptr))
        self._record("cuMemcpyDtoD_v2", dst, src, nbytes)
        self.allocations[dst][: int(nbytes)] = self.allocations[src][: int(nbytes)]
        return 0

    def _cuLaunchKernel(self, fn_handle: object, *_dims: object) -> int:
        handle = int(getattr(fn_handle, "value", fn_handle))
        kernel_name = self._function_names.get(handle, "")
        self._record("cuLaunchKernel", kernel_name)
        if self.fail_launch:
            return 700

        kernel_params = _dims[-2]
        # The runtime ABI appends the synthetic grid-stride extent after the
        # declared kernel parameters. The fake driver must discover that slot
        # from the actual launch payload instead of hard-coding an index that
        # only works for one signature width.
        extent = ctypes.cast(kernel_params[len(kernel_params) - 1], ctypes.POINTER(ctypes.c_int)).contents.value
        out_ptr = ctypes.cast(kernel_params[0], ctypes.POINTER(ctypes.c_uint64)).contents.value

        if kernel_name == "vec_add_i32":
            a_ptr = ctypes.cast(kernel_params[1], ctypes.POINTER(ctypes.c_uint64)).contents.value
            b_ptr = ctypes.cast(kernel_params[2], ctypes.POINTER(ctypes.c_uint64)).contents.value
            in_a = _decode_i32(self.allocations[a_ptr], extent)
            in_b = _decode_i32(self.allocations[b_ptr], extent)
            out = [lhs + rhs for lhs, rhs in zip(in_a, in_b, strict=True)]
            self.allocations[out_ptr][: 4 * extent] = _encode_i32(out)
            return 0

        if kernel_name == "saxpy_i32":
            a_ptr = ctypes.cast(kernel_params[1], ctypes.POINTER(ctypes.c_uint64)).contents.value
            alpha = ctypes.cast(kernel_params[2], ctypes.POINTER(ctypes.c_int)).contents.value
            in_a = _decode_i32(self.allocations[a_ptr], extent)
            out = [alpha * value for value in in_a]
            self.allocations[out_ptr][: 4 * extent] = _encode_i32(out)
            return 0

        if kernel_name == "reduce_partial":
            a_ptr = ctypes.cast(kernel_params[1], ctypes.POINTER(ctypes.c_uint64)).contents.value
            in_a = _decode_i32(self.allocations[a_ptr], extent)
            out_len = len(self.allocations[out_ptr]) // 4
            chunks = _chunk_sum(in_a, max(1, out_len))
            self.allocations[out_ptr][: 4 * len(chunks)] = _encode_i32(chunks)
            return 0

        if kernel_name == "reduce_finalize":
            partial_ptr = ctypes.cast(kernel_params[1], ctypes.POINTER(ctypes.c_uint64)).contents.value
            partial = _decode_i32(self.allocations[partial_ptr], extent)
            self.allocations[out_ptr][:4] = _encode_i32([sum(partial)])
            return 0

        return 0

    def _cuGetErrorName(self, code: int, out_ptr: object) -> int:
        name = b"CUDA_ERROR_LAUNCH_FAILED" if int(code) == 700 else b"CUDA_ERROR_UNKNOWN"
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_char_p)).contents.value = name
        return 0

    def _cuGetErrorString(self, code: int, out_ptr: object) -> int:
        message = b"launch failed" if int(code) == 700 else b"unknown error"
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_char_p)).contents.value = message
        return 0


def _encode_i32(values: list[int]) -> bytes:
    array = (ctypes.c_int * len(values))(*values)
    return ctypes.string_at(ctypes.addressof(array), ctypes.sizeof(array))


def _decode_i32(payload: bytearray, count: int) -> list[int]:
    if count <= 0:
        return []
    array_type = ctypes.c_int * count
    values = array_type.from_buffer_copy(bytes(payload[: 4 * count]))
    return [int(value) for value in values]


def _chunk_sum(values: list[int], chunks: int) -> list[int]:
    if chunks <= 0:
        return []
    width = max(1, math.ceil(len(values) / chunks))
    out: list[int] = []
    for start in range(0, len(values), width):
        out.append(sum(values[start : start + width]))
    return out


def test_cuda_runtime_executes_kir_with_fake_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_driver = FakeCudaDriver()
    monkeypatch.setattr("manv.backends.cuda.runtime._load_cuda_driver", lambda: fake_driver)

    runtime = CudaRuntime()
    result = runtime.execute_kir(
        _vector_add_kernel_ir(),
        inputs={"out": [0, 0, 0, 0], "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        allow_cpu_fallback=False,
        compiled_bundle=_compiled_bundle(),
    )

    assert runtime.available is True
    assert result["_runtime"]["executed_on_gpu"] is True
    assert result["buffers"]["out"] == [6, 8, 10, 12]
    assert any(name == "cuLaunchKernel" for name, _args in fake_driver.calls)


def test_cuda_runtime_falls_back_to_cpu_reference_on_launch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_driver = FakeCudaDriver(fail_launch=True)
    monkeypatch.setattr("manv.backends.cuda.runtime._load_cuda_driver", lambda: fake_driver)

    runtime = CudaRuntime()
    result = runtime.execute_kir(
        _vector_add_kernel_ir(),
        inputs={"out": [0, 0, 0, 0], "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        allow_cpu_fallback=True,
        compiled_bundle=_compiled_bundle(),
    )

    # The fallback path is intentional: backend dispatch may choose CPU KIR
    # fallback, while `@gpu` best-effort mode can decide to fall back even
    # earlier at the HLIR boundary.
    assert result["_runtime"]["executed_on_gpu"] is False
    assert result["_runtime"]["backend"] == "cpu"
    assert "CUDA_ERROR_LAUNCH_FAILED" in result["_runtime"]["fallback_reason"]
    assert result["buffers"]["out"] == [6, 8, 10, 12]


def test_cuda_runtime_marshals_scalar_kernel_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_driver = FakeCudaDriver()
    monkeypatch.setattr("manv.backends.cuda.runtime._load_cuda_driver", lambda: fake_driver)

    runtime = CudaRuntime()
    result = runtime.execute_kir(
        _saxpy_kernel_ir(),
        inputs={"out": [0, 0, 0, 0], "a": [1, 2, 3, 4], "alpha": 3},
        allow_cpu_fallback=False,
        compiled_bundle=_compiled_bundle(),
    )

    assert result["_runtime"]["executed_on_gpu"] is True
    assert result["buffers"]["out"] == [3, 6, 9, 12]


def test_cuda_runtime_executes_multi_kernel_reduction_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_driver = FakeCudaDriver()
    monkeypatch.setattr("manv.backends.cuda.runtime._load_cuda_driver", lambda: fake_driver)

    runtime = CudaRuntime()
    result = runtime.execute_kir(
        _reduction_kernel_ir(),
        inputs={"a": [1, 2, 3, 4], "__partial": [0, 0], "__out": [0]},
        allow_cpu_fallback=False,
        compiled_bundle=_compiled_bundle(),
    )

    assert result["_runtime"]["executed_on_gpu"] is True
    assert result["buffers"]["__partial"] == [3, 7]
    assert result["buffers"]["__out"] == [10]


def test_dispatch_reports_cpu_when_cuda_runtime_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_driver = FakeCudaDriver(fail_launch=True)
    monkeypatch.setattr("manv.backends.cuda.runtime._load_cuda_driver", lambda: fake_driver)
    monkeypatch.setattr(
        "manv.gpu_backends.compile_cuda_source",
        lambda source, arch: SimpleNamespace(success=True, ptx="// fake ptx", log="", version="fake", flags=[arch]),
    )
    monkeypatch.chdir(tmp_path)

    result = dispatch_kernel_ir(
        _vector_add_kernel_ir(),
        backend="cuda",
        inputs={"out": [0, 0, 0, 0], "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
    )

    assert result.selected_backend == "cuda"
    assert result.executed_backend == "cpu"
    assert result.outputs["buffers"]["out"] == [6, 8, 10, 12]


def test_cuda_intrinsic_buffer_roundtrip_uses_runtime_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_driver = FakeCudaDriver()
    monkeypatch.setattr("manv.backends.cuda.runtime._load_cuda_driver", lambda: fake_driver)
    monkeypatch.setattr("manv.intrinsics._CUDA_RUNTIME", None)

    handle = invoke_intrinsic("cuda_alloc", ["buf", 16])
    invoke_intrinsic("cuda_memcpy_h2d", [handle, [9, 8, 7, 6]])
    copied = invoke_intrinsic("cuda_memcpy_d2h", [handle])

    assert copied == [9, 8, 7, 6]
    assert invoke_intrinsic("cuda_last_error", []) == ""

    invoke_intrinsic("cuda_free", [handle])
