"""System-LLVM toolchain helpers for ManV host compilation.

Why this module exists:
- The host backend uses textual LLVM IR plus the system toolchain instead of
  bundling a Python LLVM binding.
- Keeping toolchain discovery and subprocess calls here prevents the compiler
  pipeline from being littered with platform-specific command assembly.

Important invariants:
- Output naming is deterministic.
- LLVM failure surfaces are translated into stable ManV diagnostics.
- Native emission is LLVM-only. Interpreter-host compatibility remains a
  separate explicit mode; this module does not silently fall back to the older
  assembly-based path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Iterable

from .diagnostics import ManvError, diag
from .targets import TargetSpec


@dataclass(frozen=True)
class LlvmToolchain:
    clang: str
    version_text: str


def detect_llvm_toolchain() -> LlvmToolchain | None:
    clang = shutil.which("clang")
    if clang is None:
        return None
    proc = subprocess.run([clang, "--version"], capture_output=True, text=True, check=False)
    version_text = proc.stdout.splitlines()[0].strip() if proc.returncode == 0 and proc.stdout else "unknown"
    return LlvmToolchain(clang=clang, version_text=version_text)


def build_llvm_artifacts(
    *,
    llvm_ir: str,
    out_dir: Path,
    stem: str,
    target: TargetSpec,
    emit_ir: bool,
    emit_object: bool,
    emit_executable: bool,
    emit_asm: bool = False,
    link_libs: tuple[str, ...] = (),
    link_paths: tuple[str, ...] = (),
    link_args: tuple[str, ...] = (),
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ll_path = out_dir / f"{stem}.{target.name}.ll"
    ll_path.write_text(llvm_ir, encoding="utf-8")
    if emit_ir:
        paths["llvm_ir"] = ll_path

    toolchain = detect_llvm_toolchain()
    if toolchain is None:
        raise ManvError(diag("E5201", "no LLVM toolchain found (clang)", str(out_dir), 1, 1))

    obj_ext = ".obj" if platform.system().lower() == "windows" else ".o"
    obj_path = out_dir / f"{stem}.{target.name}{obj_ext}"
    asm_path = out_dir / f"{stem}.{target.name}.llvm.s"
    runtime_c_path = out_dir / f"{stem}.{target.name}.runtime.c"
    runtime_obj_path = out_dir / f"{stem}.{target.name}.runtime{obj_ext}"
    exe_suffix = ".exe" if platform.system().lower() == "windows" else ""
    exe_path = out_dir / f"{stem}.{target.name}{exe_suffix}"

    if emit_asm:
        _run(
            [toolchain.clang, "-S", "-x", "ir", str(ll_path), "-o", str(asm_path), "-target", _target_triple(target)],
            cwd=out_dir,
            err_code="E5202",
            err_prefix="llvm assembly emission failed",
        )
        paths["asm"] = asm_path

    if emit_object or emit_executable:
        _run(
            [toolchain.clang, "-c", "-x", "ir", str(ll_path), "-o", str(obj_path), "-target", _target_triple(target)],
            cwd=out_dir,
            err_code="E5203",
            err_prefix="llvm object emission failed",
        )
        paths["native_obj"] = obj_path

    if emit_executable:
        runtime_c_path.write_text(_runtime_support_c(), encoding="utf-8")
        _run(
            [toolchain.clang, "-c", str(runtime_c_path), "-o", str(runtime_obj_path), "-target", _target_triple(target)],
            cwd=out_dir,
            err_code="E5204",
            err_prefix="runtime support compilation failed",
        )
        command = [toolchain.clang, str(obj_path), str(runtime_obj_path), "-o", str(exe_path), "-target", _target_triple(target)]
        for path in link_paths:
            command.extend(["-L", path])
        for lib in link_libs:
            command.append(f"-l{lib}")
        command.extend(link_args)
        _run(command, cwd=out_dir, err_code="E5205", err_prefix="llvm link failed")
        paths["native_exe"] = exe_path

    return paths


def _run(command: Iterable[str], *, cwd: Path, err_code: str, err_prefix: str) -> None:
    proc = subprocess.run(list(command), cwd=str(cwd), capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return
    message = err_prefix
    stderr = proc.stderr.strip()
    stdout = proc.stdout.strip()
    if stderr:
        message = f"{message}: {stderr}"
    elif stdout:
        message = f"{message}: {stdout}"
    raise ManvError(diag(err_code, message, str(cwd), 1, 1))


def _runtime_support_c() -> str:
    return (
        "#include <stdint.h>\n"
        "#include <stdio.h>\n\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        "#if defined(_WIN32)\n"
        "#include <windows.h>\n"
        "#else\n"
        "#include <unistd.h>\n"
        "#endif\n\n"
        "enum {\n"
        "    MANV_TAG_BOOL = 1,\n"
        "    MANV_TAG_I64 = 2,\n"
        "    MANV_TAG_F64 = 3,\n"
        "    MANV_TAG_PTR = 4\n"
        "};\n\n"
        "typedef struct ManvArray {\n"
        "    int32_t element_tag;\n"
        "    int32_t reserved;\n"
        "    int64_t length;\n"
        "    void* data;\n"
        "} ManvArray;\n\n"
        "typedef struct ManvMap {\n"
        "    int32_t key_tag;\n"
        "    int32_t value_tag;\n"
        "    int64_t length;\n"
        "    int64_t capacity;\n"
        "    void* keys;\n"
        "    void* values;\n"
        "} ManvMap;\n\n"
        "typedef struct ManvSyscallResult {\n"
        "    int64_t ok;\n"
        "    int64_t result_i64;\n"
        "    const char* platform;\n"
        "} ManvSyscallResult;\n\n"
        "static void manv_rt_abort(const char* message) {\n"
        "    fprintf(stderr, \"%s\\n\", message);\n"
        "    fflush(stderr);\n"
        "    exit(1);\n"
        "}\n\n"
        "static void* manv_rt_xcalloc(size_t count, size_t size, const char* what) {\n"
        "    void* ptr = calloc(count, size);\n"
        "    if (ptr == NULL) {\n"
        "        fprintf(stderr, \"out of memory while allocating %s\\n\", what);\n"
        "        fflush(stderr);\n"
        "        exit(1);\n"
        "    }\n"
        "    return ptr;\n"
        "}\n\n"
        "static size_t manv_rt_tag_size(int32_t tag) {\n"
        "    switch (tag) {\n"
        "        case MANV_TAG_BOOL:\n"
        "        case MANV_TAG_I64:\n"
        "            return sizeof(int64_t);\n"
        "        case MANV_TAG_F64:\n"
        "            return sizeof(double);\n"
        "        case MANV_TAG_PTR:\n"
        "            return sizeof(void*);\n"
        "        default:\n"
        "            manv_rt_abort(\"unsupported ManV runtime value tag\");\n"
        "            return sizeof(int64_t);\n"
        "    }\n"
        "}\n\n"
        "static void manv_rt_check_array_bounds(const ManvArray* array, int64_t index) {\n"
        "    if (array == NULL) {\n"
        "        manv_rt_abort(\"array operation received a null array\");\n"
        "    }\n"
        "    if (index < 0 || index >= array->length) {\n"
        "        fprintf(stderr, \"array index out of bounds: %lld\\n\", (long long)index);\n"
        "        fflush(stderr);\n"
        "        exit(1);\n"
        "    }\n"
        "}\n\n"
        "static void manv_rt_check_map(const ManvMap* map) {\n"
        "    if (map == NULL) {\n"
        "        manv_rt_abort(\"map operation received a null map\");\n"
        "    }\n"
        "}\n\n"
        "static void manv_rt_ensure_map_capacity(ManvMap* map) {\n"
        "    if (map->length < map->capacity) {\n"
        "        return;\n"
        "    }\n"
        "    int64_t next_capacity = map->capacity <= 0 ? 4 : map->capacity * 2;\n"
        "    size_t key_size = manv_rt_tag_size(map->key_tag);\n"
        "    size_t value_size = manv_rt_tag_size(map->value_tag);\n"
        "    void* next_keys = manv_rt_xcalloc((size_t)next_capacity, key_size, \"map keys\");\n"
        "    void* next_values = manv_rt_xcalloc((size_t)next_capacity, value_size, \"map values\");\n"
        "    if (map->length > 0) {\n"
        "        memcpy(next_keys, map->keys, (size_t)map->length * key_size);\n"
        "        memcpy(next_values, map->values, (size_t)map->length * value_size);\n"
        "    }\n"
        "    free(map->keys);\n"
        "    free(map->values);\n"
        "    map->keys = next_keys;\n"
        "    map->values = next_values;\n"
        "    map->capacity = next_capacity;\n"
        "}\n\n"
        "static int64_t manv_rt_find_map_slot_i64(const ManvMap* map, int64_t key) {\n"
        "    const int64_t* keys = (const int64_t*)map->keys;\n"
        "    for (int64_t i = 0; i < map->length; ++i) {\n"
        "        if (keys[i] == key) {\n"
        "            return i;\n"
        "        }\n"
        "    }\n"
        "    return -1;\n"
        "}\n\n"
        "static int64_t manv_rt_find_map_slot_ptr(const ManvMap* map, const void* key) {\n"
        "    void* const* keys = (void* const*)map->keys;\n"
        "    for (int64_t i = 0; i < map->length; ++i) {\n"
        "        if (keys[i] == key) {\n"
        "            return i;\n"
        "        }\n"
        "        if (keys[i] != NULL && key != NULL && strcmp((const char*)keys[i], (const char*)key) == 0) {\n"
        "            return i;\n"
        "        }\n"
        "    }\n"
        "    return -1;\n"
        "}\n\n"
        "static const char* manv_rt_platform_name(void) {\n"
        "#if defined(_WIN32)\n"
        "    return \"nt\";\n"
        "#else\n"
        "    return \"posix\";\n"
        "#endif\n"
        "}\n\n"
        "static void manv_rt_gpu_required_abort(const char* callee) {\n"
        "    fprintf(stderr, \"GPU backend unavailable for required @gpu call: %s\\n\", callee ? callee : \"<unknown>\");\n"
        "    fflush(stderr);\n"
        "    exit(1);\n"
        "}\n\n"
        "void manv_rt_print_i64(int64_t value) {\n"
        "    printf(\"%lld\\n\", (long long)value);\n"
        "}\n\n"
        "void manv_rt_print_f64(double value) {\n"
        "    printf(\"%.17g\\n\", value);\n"
        "}\n\n"
        "void manv_rt_print_bool(_Bool value) {\n"
        "    puts(value ? \"True\" : \"False\");\n"
        "}\n\n"
        "void manv_rt_print_cstr(const char* value) {\n"
        "    puts(value ? value : \"\");\n"
        "}\n\n"
        "ManvArray* manv_rt_array_new(int64_t length, int32_t element_tag, int32_t reserved) {\n"
        "    if (length < 0) {\n"
        "        manv_rt_abort(\"array length must be non-negative\");\n"
        "    }\n"
        "    if (element_tag == 0) {\n"
        "        element_tag = MANV_TAG_I64;\n"
        "    }\n"
        "    ManvArray* array = (ManvArray*)manv_rt_xcalloc(1, sizeof(ManvArray), \"array header\");\n"
        "    array->element_tag = element_tag;\n"
        "    array->reserved = reserved;\n"
        "    array->length = length;\n"
        "    array->data = length == 0 ? NULL : manv_rt_xcalloc((size_t)length, manv_rt_tag_size(element_tag), \"array data\");\n"
        "    return array;\n"
        "}\n\n"
        "int64_t manv_rt_array_len(const ManvArray* array) {\n"
        "    return array == NULL ? 0 : array->length;\n"
        "}\n\n"
        "ManvArray* manv_rt_array_clone_sized(const ManvArray* seed, int64_t size) {\n"
        "    if (seed == NULL) {\n"
        "        manv_rt_abort(\"array_init_sized requires a non-null seed array\");\n"
        "    }\n"
        "    if (size < seed->length) {\n"
        "        manv_rt_abort(\"initializer exceeds static size\");\n"
        "    }\n"
        "    ManvArray* out = manv_rt_array_new(size, seed->element_tag, seed->reserved);\n"
        "    if (seed->length > 0 && seed->data != NULL) {\n"
        "        memcpy(out->data, seed->data, (size_t)seed->length * manv_rt_tag_size(seed->element_tag));\n"
        "    }\n"
        "    return out;\n"
        "}\n\n"
        "void manv_rt_array_set_i64(ManvArray* array, int64_t index, int64_t value) {\n"
        "    manv_rt_check_array_bounds(array, index);\n"
        "    if (array->element_tag != MANV_TAG_BOOL && array->element_tag != MANV_TAG_I64) {\n"
        "        manv_rt_abort(\"array i64 store used on non-integer array\");\n"
        "    }\n"
        "    ((int64_t*)array->data)[index] = value;\n"
        "}\n\n"
        "void manv_rt_array_set_f64(ManvArray* array, int64_t index, double value) {\n"
        "    manv_rt_check_array_bounds(array, index);\n"
        "    if (array->element_tag != MANV_TAG_F64) {\n"
        "        manv_rt_abort(\"array f64 store used on non-float array\");\n"
        "    }\n"
        "    ((double*)array->data)[index] = value;\n"
        "}\n\n"
        "void manv_rt_array_set_ptr(ManvArray* array, int64_t index, void* value) {\n"
        "    manv_rt_check_array_bounds(array, index);\n"
        "    if (array->element_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"array ptr store used on non-pointer array\");\n"
        "    }\n"
        "    ((void**)array->data)[index] = value;\n"
        "}\n\n"
        "int64_t manv_rt_array_get_i64(const ManvArray* array, int64_t index) {\n"
        "    manv_rt_check_array_bounds(array, index);\n"
        "    if (array->element_tag != MANV_TAG_BOOL && array->element_tag != MANV_TAG_I64) {\n"
        "        manv_rt_abort(\"array i64 load used on non-integer array\");\n"
        "    }\n"
        "    return ((const int64_t*)array->data)[index];\n"
        "}\n\n"
        "double manv_rt_array_get_f64(const ManvArray* array, int64_t index) {\n"
        "    manv_rt_check_array_bounds(array, index);\n"
        "    if (array->element_tag != MANV_TAG_F64) {\n"
        "        manv_rt_abort(\"array f64 load used on non-float array\");\n"
        "    }\n"
        "    return ((const double*)array->data)[index];\n"
        "}\n\n"
        "void* manv_rt_array_get_ptr(const ManvArray* array, int64_t index) {\n"
        "    manv_rt_check_array_bounds(array, index);\n"
        "    if (array->element_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"array ptr load used on non-pointer array\");\n"
        "    }\n"
        "    return ((void* const*)array->data)[index];\n"
        "}\n\n"
        "ManvMap* manv_rt_map_new(int32_t key_tag, int32_t value_tag) {\n"
        "    if (key_tag == 0) {\n"
        "        key_tag = MANV_TAG_PTR;\n"
        "    }\n"
        "    if (value_tag == 0) {\n"
        "        value_tag = MANV_TAG_I64;\n"
        "    }\n"
        "    ManvMap* map = (ManvMap*)manv_rt_xcalloc(1, sizeof(ManvMap), \"map header\");\n"
        "    map->key_tag = key_tag;\n"
        "    map->value_tag = value_tag;\n"
        "    map->length = 0;\n"
        "    map->capacity = 0;\n"
        "    map->keys = NULL;\n"
        "    map->values = NULL;\n"
        "    return map;\n"
        "}\n\n"
        "static int64_t manv_rt_find_map_slot(const ManvMap* map, int32_t key_tag, int64_t key_i64, const void* key_ptr) {\n"
        "    if (key_tag == MANV_TAG_I64 || key_tag == MANV_TAG_BOOL) {\n"
        "        return manv_rt_find_map_slot_i64(map, key_i64);\n"
        "    }\n"
        "    if (key_tag == MANV_TAG_PTR) {\n"
        "        return manv_rt_find_map_slot_ptr(map, key_ptr);\n"
        "    }\n"
        "    manv_rt_abort(\"unsupported map key tag\");\n"
        "    return -1;\n"
        "}\n\n"
        "static int64_t manv_rt_claim_map_slot(ManvMap* map, int32_t key_tag, int64_t key_i64, void* key_ptr) {\n"
        "    manv_rt_check_map(map);\n"
        "    int64_t slot = manv_rt_find_map_slot(map, key_tag, key_i64, key_ptr);\n"
        "    if (slot >= 0) {\n"
        "        return slot;\n"
        "    }\n"
        "    manv_rt_ensure_map_capacity(map);\n"
        "    slot = map->length;\n"
        "    map->length += 1;\n"
        "    if (key_tag == MANV_TAG_I64 || key_tag == MANV_TAG_BOOL) {\n"
        "        ((int64_t*)map->keys)[slot] = key_i64;\n"
        "    } else if (key_tag == MANV_TAG_PTR) {\n"
        "        ((void**)map->keys)[slot] = key_ptr;\n"
        "    } else {\n"
        "        manv_rt_abort(\"unsupported map key tag\");\n"
        "    }\n"
        "    return slot;\n"
        "}\n\n"
        "static int64_t manv_rt_require_map_slot_i64(const ManvMap* map, int64_t key) {\n"
        "    manv_rt_check_map(map);\n"
        "    int64_t slot = manv_rt_find_map_slot_i64(map, key);\n"
        "    if (slot < 0) {\n"
        "        fprintf(stderr, \"missing integer map key: %lld\\n\", (long long)key);\n"
        "        fflush(stderr);\n"
        "        exit(1);\n"
        "    }\n"
        "    return slot;\n"
        "}\n\n"
        "static int64_t manv_rt_require_map_slot_ptr(const ManvMap* map, const void* key) {\n"
        "    manv_rt_check_map(map);\n"
        "    int64_t slot = manv_rt_find_map_slot_ptr(map, key);\n"
        "    if (slot < 0) {\n"
        "        fprintf(stderr, \"missing pointer map key\\n\");\n"
        "        fflush(stderr);\n"
        "        exit(1);\n"
        "    }\n"
        "    return slot;\n"
        "}\n\n"
        "void manv_rt_map_set_i64_i64(ManvMap* map, int64_t key, int64_t value) {\n"
        "    if (map->key_tag != MANV_TAG_I64 && map->key_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map key store used on non-integer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_I64 && map->value_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map value store used on non-integer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_claim_map_slot(map, map->key_tag, key, NULL);\n"
        "    ((int64_t*)map->values)[slot] = value;\n"
        "}\n\n"
        "void manv_rt_map_set_i64_f64(ManvMap* map, int64_t key, double value) {\n"
        "    if (map->key_tag != MANV_TAG_I64 && map->key_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map key store used on non-integer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_F64) {\n"
        "        manv_rt_abort(\"float map value store used on non-float-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_claim_map_slot(map, map->key_tag, key, NULL);\n"
        "    ((double*)map->values)[slot] = value;\n"
        "}\n\n"
        "void manv_rt_map_set_i64_ptr(ManvMap* map, int64_t key, void* value) {\n"
        "    if (map->key_tag != MANV_TAG_I64 && map->key_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map key store used on non-integer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map value store used on non-pointer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_claim_map_slot(map, map->key_tag, key, NULL);\n"
        "    ((void**)map->values)[slot] = value;\n"
        "}\n\n"
        "void manv_rt_map_set_ptr_i64(ManvMap* map, void* key, int64_t value) {\n"
        "    if (map->key_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map key store used on non-pointer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_I64 && map->value_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map value store used on non-integer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_claim_map_slot(map, map->key_tag, 0, key);\n"
        "    ((int64_t*)map->values)[slot] = value;\n"
        "}\n\n"
        "void manv_rt_map_set_ptr_f64(ManvMap* map, void* key, double value) {\n"
        "    if (map->key_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map key store used on non-pointer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_F64) {\n"
        "        manv_rt_abort(\"float map value store used on non-float-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_claim_map_slot(map, map->key_tag, 0, key);\n"
        "    ((double*)map->values)[slot] = value;\n"
        "}\n\n"
        "void manv_rt_map_set_ptr_ptr(ManvMap* map, void* key, void* value) {\n"
        "    if (map->key_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map key store used on non-pointer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map value store used on non-pointer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_claim_map_slot(map, map->key_tag, 0, key);\n"
        "    ((void**)map->values)[slot] = value;\n"
        "}\n\n"
        "int64_t manv_rt_map_get_i64_i64(const ManvMap* map, int64_t key) {\n"
        "    if (map->key_tag != MANV_TAG_I64 && map->key_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map key load used on non-integer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_I64 && map->value_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map load used on non-integer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_require_map_slot_i64(map, key);\n"
        "    return ((const int64_t*)map->values)[slot];\n"
        "}\n\n"
        "double manv_rt_map_get_i64_f64(const ManvMap* map, int64_t key) {\n"
        "    if (map->key_tag != MANV_TAG_I64 && map->key_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map key load used on non-integer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_F64) {\n"
        "        manv_rt_abort(\"float map load used on non-float-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_require_map_slot_i64(map, key);\n"
        "    return ((const double*)map->values)[slot];\n"
        "}\n\n"
        "void* manv_rt_map_get_i64_ptr(const ManvMap* map, int64_t key) {\n"
        "    if (map->key_tag != MANV_TAG_I64 && map->key_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map key load used on non-integer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map load used on non-pointer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_require_map_slot_i64(map, key);\n"
        "    return ((void* const*)map->values)[slot];\n"
        "}\n\n"
        "int64_t manv_rt_map_get_ptr_i64(const ManvMap* map, void* key) {\n"
        "    if (map->key_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map key load used on non-pointer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_I64 && map->value_tag != MANV_TAG_BOOL) {\n"
        "        manv_rt_abort(\"integer map load used on non-integer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_require_map_slot_ptr(map, key);\n"
        "    return ((const int64_t*)map->values)[slot];\n"
        "}\n\n"
        "double manv_rt_map_get_ptr_f64(const ManvMap* map, void* key) {\n"
        "    if (map->key_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map key load used on non-pointer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_F64) {\n"
        "        manv_rt_abort(\"float map load used on non-float-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_require_map_slot_ptr(map, key);\n"
        "    return ((const double*)map->values)[slot];\n"
        "}\n\n"
        "void* manv_rt_map_get_ptr_ptr(const ManvMap* map, void* key) {\n"
        "    if (map->key_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map key load used on non-pointer-key map\");\n"
        "    }\n"
        "    if (map->value_tag != MANV_TAG_PTR) {\n"
        "        manv_rt_abort(\"pointer map load used on non-pointer-value map\");\n"
        "    }\n"
        "    int64_t slot = manv_rt_require_map_slot_ptr(map, key);\n"
        "    return ((void* const*)map->values)[slot];\n"
        "}\n\n"
        "int64_t manv_rt_map_len(const ManvMap* map) {\n"
        "    return map == NULL ? 0 : map->length;\n"
        "}\n\n"
        "int64_t manv_rt_cstr_len(const char* value) {\n"
        "    return value == NULL ? 0 : (int64_t)strlen(value);\n"
        "}\n\n"
        "ManvSyscallResult* manv_rt_syscall_invoke_cstr(const char* target) {\n"
        "    if (target == NULL) {\n"
        "        fprintf(stderr, \"syscall target must be a non-null string\\n\");\n"
        "        exit(1);\n"
        "    }\n"
        "    ManvSyscallResult* out = (ManvSyscallResult*)calloc(1, sizeof(ManvSyscallResult));\n"
        "    if (out == NULL) {\n"
        "        fprintf(stderr, \"out of memory while creating syscall result\\n\");\n"
        "        exit(1);\n"
        "    }\n"
        "    out->platform = manv_rt_platform_name();\n"
        "    if (strcmp(target, \"getpid\") == 0) {\n"
        "#if defined(_WIN32)\n"
        "        out->result_i64 = (int64_t)GetCurrentProcessId();\n"
        "#else\n"
        "        out->result_i64 = (int64_t)getpid();\n"
        "#endif\n"
        "        out->ok = 1;\n"
        "        return out;\n"
        "    }\n"
        "    fprintf(stderr, \"unsupported syscall alias: %s\\n\", target);\n"
        "    exit(1);\n"
        "}\n\n"
        "ManvSyscallResult* manv_rt_syscall_invoke_i64(int64_t target) {\n"
        "    fprintf(stderr, \"numeric syscall is not implemented in the LLVM host runtime: %lld\\n\", (long long)target);\n"
        "    exit(1);\n"
        "}\n\n"
        "_Bool manv_rt_syscall_result_ok(const ManvSyscallResult* value) {\n"
        "    return value != NULL && value->ok != 0;\n"
        "}\n\n"
        "int64_t manv_rt_syscall_result_i64(const ManvSyscallResult* value) {\n"
        "    return value == NULL ? 0 : value->result_i64;\n"
        "}\n\n"
        "const char* manv_rt_syscall_result_platform(const ManvSyscallResult* value) {\n"
        "    return value == NULL ? \"\" : value->platform;\n"
        "}\n\n"
        "void manv_rt_gpu_required_void(const char* callee) {\n"
        "    manv_rt_gpu_required_abort(callee);\n"
        "}\n\n"
        "int64_t manv_rt_gpu_required_i64(const char* callee) {\n"
        "    manv_rt_gpu_required_abort(callee);\n"
        "    return 0;\n"
        "}\n\n"
        "double manv_rt_gpu_required_f64(const char* callee) {\n"
        "    manv_rt_gpu_required_abort(callee);\n"
        "    return 0.0;\n"
        "}\n\n"
        "float manv_rt_gpu_required_f32(const char* callee) {\n"
        "    manv_rt_gpu_required_abort(callee);\n"
        "    return 0.0f;\n"
        "}\n\n"
        "void* manv_rt_gpu_required_ptr(const char* callee) {\n"
        "    manv_rt_gpu_required_abort(callee);\n"
        "    return NULL;\n"
        "}\n"
    )


def _target_triple(target: TargetSpec) -> str:
    if target.name == "x86_64-sysv":
        return "x86_64-unknown-linux-gnu"
    if target.name == "x86_64-win64":
        return "x86_64-pc-windows-msvc"
    if target.name == "aarch64-aapcs64":
        return "aarch64-unknown-linux-gnu"
    return "unknown-unknown-unknown"
