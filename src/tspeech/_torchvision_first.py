"""
Import torch + torchvision before lightning/torchmetrics.

Lightning pulls torchmetrics at import time; torchmetrics imports torchvision for
image metrics. If torchvision is not fully initialized first, you can see::

  AttributeError: partially initialized module 'torchvision' has no attribute 'extension'

**Also required:** ``torch``, ``torchvision``, and ``torchaudio`` must be built for the
same PyTorch release and CUDA variant (install all from the same PyTorch index), e.g.::

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

A mismatched torchvision (e.g. cu124 wheels with torch cu128) causes runtime errors
such as ``operator torchvision::nms does not exist``.
"""
import torch  # noqa: F401

try:
    import torchvision  # noqa: F401
except (ModuleNotFoundError, RuntimeError):
    # Inference-only environments may not have torchvision installed. It's only
    # needed for certain torchmetrics image metrics pulled in by Lightning.
    #
    # Some environments have an incompatible torchvision build (e.g. missing C++ ops),
    # which can raise RuntimeError like: "operator torchvision::nms does not exist".
    torchvision = None  # type: ignore[assignment]

