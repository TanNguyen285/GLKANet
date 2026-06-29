"""
Load và phân tích ONNX model:
  - Đọc graph metadata (inputs/outputs, opset, nodes)
  - Đếm params từ initializers
  - Tính FLOPs từ ONNX graph (onnx-opcounter → fallback tự đếm Conv MACs)
  - Benchmark với ONNX Runtime
"""
from __future__ import annotations
import os

from .deps import HAS_ONNX, HAS_ONNXRT, HAS_ONNX_OPCOUNTER
from .latency import benchmark_onnx


def load_onnx(
    path:       str,
    device_str: str        = "cpu",
    warmup:     int | None = None,
    runs:       int | None = None,
) -> dict:
    result: dict = {"format": "ONNX", "file_size_mb": os.path.getsize(path) / 1e6}

    if not HAS_ONNX:
        result["onnx_err"] = "pip install onnx"
    else:
        import onnx
        try:
            proto = onnx.load(path)
            onnx.checker.check_model(proto)
            result["onnx_valid"] = True
        except Exception as ex:
            result["onnx_valid"]     = False
            result["onnx_check_err"] = str(ex)

        try:
            proto = onnx.load(path)
            gi    = proto.graph

            result["graph_name"] = gi.name
            result["opset"]      = [
                f"{o.domain or 'ai.onnx'}:{o.version}"
                for o in proto.opset_import
            ]
            result["inputs"]  = _parse_io(gi.input)
            result["outputs"] = _parse_io(gi.output)
            result["ir_version"]       = proto.ir_version
            result["producer"]         = proto.producer_name
            result["producer_version"] = proto.producer_version
            result["meta"]             = {p.key: p.value for p in proto.metadata_props}

            # op summary
            op_count: dict[str, int] = {}
            for node in gi.node:
                op_count[node.op_type] = op_count.get(node.op_type, 0) + 1
            result["op_summary"]  = dict(sorted(op_count.items(), key=lambda x: -x[1]))
            result["total_nodes"] = len(gi.node)

            # params từ initializers
            total_params = 0
            total_bytes  = 0
            layers       = []
            init_shapes: dict[str, list[int]] = {}
            for init in gi.initializer:
                shape  = list(init.dims)
                numel  = 1
                for d in shape:
                    numel *= d
                total_params += numel
                total_bytes  += numel * _dtype_bytes(init.data_type)
                layers.append({"name": init.name, "shape": shape, "params": numel})
                init_shapes[init.name] = shape

            result["total_params"]  = total_params
            result["model_size_mb"] = total_bytes / 1e6
            result["layers"]        = sorted(layers, key=lambda x: -x["params"])

            # ── FLOPs ─────────────────────────────────────────────────────
            flops, flops_err = _calc_flops(proto, init_shapes)
            if flops:
                from ..utils import fmt_flops
                result["flops"]     = flops
                result["flops_str"] = fmt_flops(flops)
            if flops_err:
                result["flops_err"] = flops_err

        except Exception as ex:
            result["parse_error"] = str(ex)

    # ── ORT benchmark ─────────────────────────────────────────────────────
    if HAS_ONNXRT:
        import onnxruntime as ort
        import numpy as np

        try:
            providers = ort.get_available_providers()
            result["ort_providers"] = providers

            if device_str == "cuda" and "CUDAExecutionProvider" in providers:
                chosen = "CUDAExecutionProvider"
            elif "CoreMLExecutionProvider" in providers:
                chosen = "CoreMLExecutionProvider"
            else:
                chosen = "CPUExecutionProvider"
            result["ort_provider_used"] = chosen

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                path, sess_options=sess_opts, providers=[chosen]
            )

            dtype_map = {
                "tensor(float)":   np.float32,
                "tensor(float16)": np.float16,
                "tensor(int64)":   np.int64,
                "tensor(int32)":   np.int32,
                "tensor(uint8)":   np.uint8,
            }
            dummy_inputs = {}
            for inp in sess.get_inputs():
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                dtype = dtype_map.get(inp.type, np.float32)
                dummy_inputs[inp.name] = np.random.randn(*shape).astype(dtype)

            result.update(benchmark_onnx(
                sess, dummy_inputs,
                warmup=warmup,
                runs=runs,
                device_str=device_str,
            ))

        except Exception as ex:
            result["ort_error"] = str(ex)

    return result


# ── FLOPs ─────────────────────────────────────────────────────────────────────

def _calc_flops(proto, init_shapes: dict[str, list[int]]) -> tuple[int, str]:
    """
    Tính MACs từ ONNX graph.
    Ưu tiên onnx-opcounter; fallback shape_inference với batch=1 cố định.
    Trả về (macs, error_msg).
    """
    import onnx
    import copy

    # ── Bước 0: Fix dynamic batch → 1 để shape_inference hoạt động ──────
    # Clone proto, set tất cả dim_param (dynamic) thành dim_value=1
    proto_fixed = copy.deepcopy(proto)
    for inp in proto_fixed.graph.input:
        try:
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param or dim.dim_value == 0:
                    dim.ClearField("dim_param")
                    dim.dim_value = 1
        except Exception:
            pass

    # ── Bước 1: onnx-opcounter (chính xác nhất) ──────────────────────────
    if HAS_ONNX_OPCOUNTER:
        try:
            from onnx_opcounter import calculate_macs
            macs = calculate_macs(proto_fixed)
            if macs > 0:
                return int(macs), ""
        except Exception as ex:
            pass  # fallback xuống shape_inference

    # ── Bước 2: shape_inference + tự đếm Conv/Gemm ───────────────────────
    try:
        inferred  = onnx.shape_inference.infer_shapes(proto_fixed)
        shape_map: dict[str, list[int]] = dict(init_shapes)

        for vi in inferred.graph.value_info:
            try:
                s = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                if all(v > 0 for v in s):
                    shape_map[vi.name] = s
            except Exception:
                pass

        # Input tensors
        for inp in inferred.graph.input:
            try:
                s = [max(d.dim_value, 1) for d in inp.type.tensor_type.shape.dim]
                shape_map[inp.name] = s
            except Exception:
                pass

        total_macs = 0
        for node in inferred.graph.node:
            if node.op_type == "Conv":
                total_macs += _conv_macs(node, shape_map, init_shapes)
            elif node.op_type == "Gemm":
                total_macs += _gemm_macs(node, init_shapes)

        if total_macs > 0:
            return total_macs, ""
        return 0, "MACs = 0 sau shape_inference — kiểm tra lại ONNX graph"

    except Exception as ex:
        return 0, f"_calc_flops thất bại: {ex}"


def _conv_macs(node, shape_map: dict, init_shapes: dict) -> int:
    """MACs = C_out * C_in_per_group * kH * kW * oH * oW."""
    try:
        w_name  = node.input[1]
        w_shape = init_shapes.get(w_name, [])
        if len(w_shape) != 4:
            return 0
        c_out, c_in_g, kh, kw = w_shape

        out_name  = node.output[0]
        out_shape = shape_map.get(out_name, [])
        if len(out_shape) == 4 and all(v > 0 for v in out_shape):
            _, _, oh, ow = out_shape
        else:
            return 0   # không đoán — bỏ qua node này

        return int(c_out) * int(c_in_g) * int(kh) * int(kw) * int(oh) * int(ow)
    except Exception:
        return 0


def _gemm_macs(node, init_shapes: dict) -> int:
    """MACs = out_features * in_features."""
    try:
        w_shape = init_shapes.get(node.input[1], [])
        if len(w_shape) == 2:
            return w_shape[0] * w_shape[1]
        return 0
    except Exception:
        return 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_io(io_list) -> list[dict]:
    out = []
    for item in io_list:
        try:
            t     = item.type.tensor_type
            shape = [
                d.dim_value if d.HasField("dim_value") else f"?({d.dim_param})"
                for d in t.shape.dim
            ]
            out.append({"name": item.name, "shape": shape, "elem_type": t.elem_type})
        except Exception:
            out.append({"name": item.name})
    return out


def _dtype_bytes(dtype_int: int) -> int:
    return {1: 4, 2: 1, 3: 1, 4: 2, 5: 4, 6: 8, 7: 8,
            10: 2, 11: 8, 12: 4, 13: 8}.get(dtype_int, 4)