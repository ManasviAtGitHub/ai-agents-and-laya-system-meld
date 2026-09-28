"""Load Laya on a CPU backend behind one predict()/predict_batch() API.

    torch          PyTorch Agent (reference)
    onnx           ONNX Runtime, FP32 export
    openvino       OpenVINO, FP32 export (default: fastest backend that matches torch)

INT8 exports are deliberately not offered: on this model they changed 3 of 8 choice answers
against torch in our benchmark. See README.
"""
from unittest import mock

from .paths import MODELS_DIR

MODEL_ID = "convaiinnovations/laya"
BACKENDS = ("torch", "onnx", "openvino")


def onnx_path():
    path = MODELS_DIR / "laya.onnx"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run scripts/export_onnx.py "
            f"(or set LAYA_MODELS_DIR to a folder holding an existing export)"
        )
    return str(path)


class OpenVINOSession:
    """Stands in for an onnxruntime.InferenceSession; ONNXAgent only ever calls .run()."""

    def __init__(self, path, device="CPU", hint="LATENCY"):
        import openvino as ov

        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(path), device, {"PERFORMANCE_HINT": hint})
        self.request = self.compiled.create_infer_request()

    def run(self, output_names, inputs):
        results = self.request.infer(inputs)
        return [results[self.compiled.output(name)] for name in output_names]


def load(backend="openvino", openvino_device="CPU", openvino_hint="LATENCY"):
    if backend == "torch":
        import laya

        return laya.load(MODEL_ID, device="cpu")

    from laya.onnx_agent import ONNXAgent

    path = onnx_path()
    if backend == "onnx":
        return ONNXAgent(MODEL_ID, onnx_path=path)
    if backend == "openvino":
        # Hand ONNXAgent an OpenVINO session instead of letting it build an ONNX Runtime one,
        # so the weights are loaded once.
        session = OpenVINOSession(path, openvino_device, openvino_hint)
        with mock.patch("onnxruntime.InferenceSession", lambda *a, **k: session):
            return ONNXAgent(MODEL_ID, onnx_path=path)
    raise ValueError(f"unknown backend {backend!r}; pick one of {BACKENDS}")
