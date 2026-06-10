#!/usr/bin/env python3
"""
Patch ggml.c to add big-endian byte-swapping to the GGUF reader.
GGUF format is always little-endian on disk. On big-endian systems (PowerPC G5),
all multi-byte scalar values must be byte-swapped when reading.
"""
import sys

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # === PATCH 1: Add byte-swap helpers and gguf_fread_val above gguf_fread_el ===
    old_fread_el = """static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}"""

    new_fread_section = """// --- Big-endian byte-swap support for GGUF ---
// GGUF is always little-endian on disk. On big-endian hosts, swap multi-byte values.
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define GGUF_IS_BIG_ENDIAN 1
#else
#define GGUF_IS_BIG_ENDIAN 0
#endif

#if GGUF_IS_BIG_ENDIAN
static inline void gguf_bswap_2(void * p) {
    uint8_t * b = (uint8_t *)p;
    uint8_t t = b[0]; b[0] = b[1]; b[1] = t;
}
static inline void gguf_bswap_4(void * p) {
    uint8_t * b = (uint8_t *)p;
    uint8_t t;
    t = b[0]; b[0] = b[3]; b[3] = t;
    t = b[1]; b[1] = b[2]; b[2] = t;
}
static inline void gguf_bswap_8(void * p) {
    uint8_t * b = (uint8_t *)p;
    uint8_t t;
    t = b[0]; b[0] = b[7]; b[7] = t;
    t = b[1]; b[1] = b[6]; b[6] = t;
    t = b[2]; b[2] = b[5]; b[5] = t;
    t = b[3]; b[3] = b[4]; b[4] = t;
}
// Swap a single element of known size
static inline void gguf_bswap(void * data, size_t size) {
    switch (size) {
        case 2: gguf_bswap_2(data); break;
        case 4: gguf_bswap_4(data); break;
        case 8: gguf_bswap_8(data); break;
        default: break;
    }
}
// Swap N elements of given element size
static inline void gguf_bswap_n(void * data, size_t n, size_t elem_size) {
    if (elem_size <= 1) return;
    uint8_t * p = (uint8_t *)data;
    for (size_t i = 0; i < n; i++) {
        gguf_bswap(p + i * elem_size, elem_size);
    }
}
#endif

// Raw read - no byte-swapping (used for string data, bulk tensor data)
static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

// Read a scalar value with byte-swap on big-endian
// Use ONLY for numeric scalars (uint32, uint64, float32, etc.), NOT for string data
static bool gguf_fread_val(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    if (n != size) return false;
#if GGUF_IS_BIG_ENDIAN
    gguf_bswap(dst, size);
#endif
    return true;
}"""

    if old_fread_el not in content:
        print("ERROR: Could not find gguf_fread_el to patch")
        sys.exit(1)
    content = content.replace(old_fread_el, new_fread_section)
    print("PATCH 1: Added byte-swap helpers and gguf_fread_val")

    # === PATCH 2: Fix gguf_fread_str to swap string length ===
    old_fread_str = """static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%" PRIu64 ")\\n", __func__, p->n);
        return false;
    }

    p->data = GGML_CALLOC(p->n + 1, 1);

    ok = ok && gguf_fread_el(file,  p->data, p->n, offset);"""

    new_fread_str = """static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    // Read string length as scalar (needs byte-swap on BE)
    ok = ok && gguf_fread_val(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%" PRIu64 ")\\n", __func__, p->n);
        return false;
    }

    p->data = GGML_CALLOC(p->n + 1, 1);

    // Read string data as raw bytes (no swap needed for character data)
    ok = ok && gguf_fread_el(file,  p->data, p->n, offset);"""

    if old_fread_str not in content:
        print("ERROR: Could not find gguf_fread_str to patch")
        sys.exit(1)
    content = content.replace(old_fread_str, new_fread_str)
    print("PATCH 2: Fixed gguf_fread_str to swap string length")

    # === PATCH 3: Header reads - use gguf_fread_val for version, n_tensors, n_kv ===
    old_header = """        ok = ok && gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);"""

    new_header = """        ok = ok && gguf_fread_val(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_val(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_val(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);"""

    if old_header not in content:
        print("ERROR: Could not find header reads to patch")
        sys.exit(1)
    content = content.replace(old_header, new_header)
    print("PATCH 3: Header reads now use gguf_fread_val")

    # === PATCH 4: KV type read ===
    old_kv_type = """            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);"""

    new_kv_type = """            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_val(file, &kv->type, sizeof(kv->type), &offset);"""

    if old_kv_type not in content:
        print("ERROR: Could not find KV type read to patch")
        sys.exit(1)
    content = content.replace(old_kv_type, new_kv_type)
    print("PATCH 4: KV type reads now use gguf_fread_val")

    # === PATCH 5: KV scalar value reads ===
    kv_scalar_types = [
        "GGUF_TYPE_UINT8",  "GGUF_TYPE_INT8",
        "GGUF_TYPE_UINT16", "GGUF_TYPE_INT16",
        "GGUF_TYPE_UINT32", "GGUF_TYPE_INT32",
        "GGUF_TYPE_FLOAT32",
        "GGUF_TYPE_UINT64", "GGUF_TYPE_INT64",
        "GGUF_TYPE_FLOAT64",
        "GGUF_TYPE_BOOL",
    ]

    kv_value_fields = {
        "GGUF_TYPE_UINT8":   "uint8",
        "GGUF_TYPE_INT8":    "int8",
        "GGUF_TYPE_UINT16":  "uint16",
        "GGUF_TYPE_INT16":   "int16",
        "GGUF_TYPE_UINT32":  "uint32",
        "GGUF_TYPE_INT32":   "int32",
        "GGUF_TYPE_FLOAT32": "float32",
        "GGUF_TYPE_UINT64":  "uint64",
        "GGUF_TYPE_INT64":   "int64",
        "GGUF_TYPE_FLOAT64": "float64",
        "GGUF_TYPE_BOOL":    "bool_",
    }

    count = 0
    for typ in kv_scalar_types:
        field = kv_value_fields[typ]
        # Match the pattern for each KV scalar read
        old_line = f"case {typ}:   ok = ok && gguf_fread_el (file, &kv->value.{field},"
        new_line = f"case {typ}:   ok = ok && gguf_fread_val(file, &kv->value.{field},"
        if old_line not in content:
            # Try without extra spaces
            old_line = f"case {typ}: ok = ok && gguf_fread_el (file, &kv->value.{field},"
            new_line = f"case {typ}: ok = ok && gguf_fread_val(file, &kv->value.{field},"
        if old_line not in content:
            # Try with different spacing
            old_line = f"case {typ}:  ok = ok && gguf_fread_el (file, &kv->value.{field},"
            new_line = f"case {typ}:  ok = ok && gguf_fread_val(file, &kv->value.{field},"
        if old_line in content:
            content = content.replace(old_line, new_line)
            count += 1
        else:
            print(f"  WARNING: Could not find KV read for {typ} / {field}")

    print(f"PATCH 5: Replaced {count}/{len(kv_scalar_types)} KV scalar reads with gguf_fread_val")

    # === PATCH 6: Array type and count reads ===
    old_arr_meta = """                        ok = ok && gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);"""

    new_arr_meta = """                        ok = ok && gguf_fread_val(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && gguf_fread_val(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);"""

    if old_arr_meta not in content:
        print("ERROR: Could not find array metadata reads to patch")
        sys.exit(1)
    content = content.replace(old_arr_meta, new_arr_meta)
    print("PATCH 6: Array type/count reads now use gguf_fread_val")

    # === PATCH 7: Array data bulk read - add element-wise swap ===
    old_arr_data = """                                    kv->value.arr.data = GGML_CALLOC(kv->value.arr.n, gguf_type_size(kv->value.arr.type));

                                    ok = ok && gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * gguf_type_size(kv->value.arr.type), &offset);
                                } break;"""

    new_arr_data = """                                    kv->value.arr.data = GGML_CALLOC(kv->value.arr.n, gguf_type_size(kv->value.arr.type));

                                    ok = ok && gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * gguf_type_size(kv->value.arr.type), &offset);
#if GGUF_IS_BIG_ENDIAN
                                    // Byte-swap each element in the array
                                    gguf_bswap_n(kv->value.arr.data, kv->value.arr.n, gguf_type_size(kv->value.arr.type));
#endif
                                } break;"""

    if old_arr_data not in content:
        print("ERROR: Could not find array data read to patch")
        sys.exit(1)
    content = content.replace(old_arr_data, new_arr_data)
    print("PATCH 7: Array data read now has element-wise byte-swap")

    # === PATCH 8: Tensor info reads ===
    old_tensor_ndims = """            ok = ok && gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);"""

    new_tensor_ndims = """            ok = ok && gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && gguf_fread_val(file, &info->n_dims, sizeof(info->n_dims),  &offset);"""

    if old_tensor_ndims not in content:
        print("ERROR: Could not find tensor n_dims read to patch")
        sys.exit(1)
    content = content.replace(old_tensor_ndims, new_tensor_ndims)
    print("PATCH 8a: Tensor n_dims now uses gguf_fread_val")

    # Tensor ne[j]
    old_tensor_ne = """                ok = ok && gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);"""
    new_tensor_ne = """                ok = ok && gguf_fread_val(file, &info->ne[j], sizeof(info->ne[j]), &offset);"""

    if old_tensor_ne not in content:
        print("ERROR: Could not find tensor ne read to patch")
        sys.exit(1)
    content = content.replace(old_tensor_ne, new_tensor_ne)
    print("PATCH 8b: Tensor ne[j] now uses gguf_fread_val")

    # Tensor type and offset
    old_tensor_type_off = """            ok = ok && gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);"""

    new_tensor_type_off = """            ok = ok && gguf_fread_val(file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && gguf_fread_val(file, &info->offset, sizeof(info->offset),  &offset);"""

    if old_tensor_type_off not in content:
        print("ERROR: Could not find tensor type/offset reads to patch")
        sys.exit(1)
    content = content.replace(old_tensor_type_off, new_tensor_type_off)
    print("PATCH 8c: Tensor type/offset now use gguf_fread_val")

    # Write patched file
    with open(path, 'w') as f:
        f.write(content)

    print(f"\nAll patches applied to {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-ggml.c>")
        sys.exit(1)
    patch_file(sys.argv[1])
