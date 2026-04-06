"""
SPECTER2 Encoder Wrapper
------------------------
Wraps the adapter-based allenai/specter2_base model behind the same
.encode(texts, normalize_embeddings, batch_size, show_progress_bar) interface
that sentence_transformers.SentenceTransformer exposes, so DenseIndexer and
HybridRetriever can use it as a drop-in replacement.

Multi-GPU note:
  When multiple CUDA devices are visible (e.g. via CUDA_VISIBLE_DEVICES=0,1,2),
  the encoder automatically wraps the model in torch.nn.DataParallel.  Each
  forward pass splits the batch evenly across all GPUs and gathers embeddings
  back to GPU 0, effectively multiplying encoding throughput by n_gpu.

SPECTER2 reference:
  Singh et al. (2022) "SciRepEval: A Multi-Format Benchmark for Scientific
  Document Representations". https://arxiv.org/abs/2211.13308
  Model card: https://huggingface.co/allenai/specter2_base
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("[Specter2Encoder] Running on device: %s%s", _DEVICE,
            f" ({torch.cuda.get_device_name(0)})" if _DEVICE == "cuda" else " (no GPU detected)")


class _ModelWrapper(torch.nn.Module):
    """
    Thin wrapper that calls the underlying model and returns the raw
    last_hidden_state tensor.  torch.nn.DataParallel can only scatter /
    gather plain tensors — it cannot handle HuggingFace ModelOutput
    namedtuple objects.  This adapter closes that gap.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(**kwargs).last_hidden_state


class Specter2Encoder:
    """
    Encodes text using allenai/specter2_base + the retrieval adapter.

    Output dimension: 768  (vs. 384 for bge-small)
    Max input tokens:  512

    The adapter library (`pip install adapters`) must be installed.
    On first use the model weights (~480 MB) and adapter (~40 MB) are
    downloaded from HuggingFace Hub and cached locally.
    """

    BASE_MODEL = "allenai/specter2_base"
    ADAPTER_NAME = "allenai/specter2"  # retrieval adapter on HF Hub
    DIMENSION = 768

    def __init__(self, device: str = _DEVICE, model_kwargs: dict = None):
        """
        Added `model_kwargs` to allow passing optimization flags (like torch.float16)
        down to the underlying HuggingFace adapter model.
        """
        if model_kwargs is None:
            model_kwargs = {}

        try:
            from adapters import AutoAdapterModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'adapters' package is required for SPECTER2. "
                "Install it with: pip install adapters"
            ) from e

        logger.info("Loading SPECTER2 tokenizer from %s ...", self.BASE_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)

        logger.info("Loading SPECTER2 base model with kwargs: %s ...", model_kwargs)
        # **model_kwargs unpacks the dictionary, passing torch_dtype=torch.float16
        # to HuggingFace, drastically reducing VRAM usage.
        self.model = AutoAdapterModel.from_pretrained(self.BASE_MODEL, **model_kwargs)

        logger.info("Loading SPECTER2 retrieval adapter (%s) ...", self.ADAPTER_NAME)
        self.model.load_adapter(
            self.ADAPTER_NAME,
            source="hf",
            load_as="specter2_retrieval",
            set_active=True,
        )
        # We explicitly activate the adapter to remove the warning and ensure
        # we are actually getting SPECTER2 retrieval embeddings, not base BERT.
        self.model.active_adapters = "specter2_retrieval"

        # We force any dynamically added adapter layers to match the base model's dtype.
        # If the base model is FP16, this converts the 32-bit adapter weights to 16-bit.
        if model_kwargs.get("torch_dtype") == torch.float16:
            self.model.half()

        self.device = device
        self.model.to(device)
        self.model.eval()

        # Multi-GPU: wrap in DataParallel so each forward pass splits the batch
        # evenly across all visible CUDA devices.  _ModelWrapper converts the
        # HuggingFace ModelOutput to a plain tensor so DataParallel can
        # scatter and gather correctly.
        # n_gpu: number of CUDA devices visible to this process.
        # 0 when no CUDA is available (CPU-only node) → DataParallel not used,
        # model stays on CPU which is set as self.device above.
        # 1 → single-GPU path (no DataParallel overhead).
        # 2+ → DataParallel splits each batch across all devices.
        self.n_gpu = torch.cuda.device_count() if device.startswith("cuda") else 0
        if self.n_gpu > 1:
            logger.info(
                "SPECTER2: enabling DataParallel across %d GPUs — effective "
                "batch size is multiplied by n_gpu for the same memory budget.",
                self.n_gpu,
            )
            self._dp_model = torch.nn.DataParallel(
                _ModelWrapper(self.model),
                device_ids=list(range(self.n_gpu)),
            )
            self._use_dp = True
        else:
            self._use_dp = False

        logger.info(
            "SPECTER2 encoder ready on device=%s  |  GPUs=%d",
            device, self.n_gpu,
        )

    def encode(
        self,
        texts: list,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into numpy float32 vectors (CLS pooling).

        Parameters
        ----------
        texts : list[str]
        normalize_embeddings : bool
            L2-normalise output vectors (required for cosine / inner-product search).
        batch_size : int
            Number of texts per forward pass.  When DataParallel is active the
            batch is split evenly across GPUs, so effective per-GPU batch size
            is batch_size // n_gpu.
        show_progress_bar : bool
            Show tqdm progress bar during batch encoding.

        Returns
        -------
        np.ndarray  shape (len(texts), 768), dtype float32
        """
        all_embeddings: list = []

        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding with SPECTER2")

        for start in iterator:
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                device_type = "cuda" if "cuda" in self.device else "cpu"
                with torch.autocast(device_type=device_type):
                    if self._use_dp:
                        # DataParallel scatters inputs across GPUs and gathers
                        # the last_hidden_state tensor back to self.device.
                        last_hidden_state = self._dp_model(**inputs)
                    else:
                        last_hidden_state = self.model(**inputs).last_hidden_state

            # CLS-token pooling — recommended by AllenAI for SPECTER2
            cls_embeddings = last_hidden_state[:, 0, :]

            if normalize_embeddings:
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)

            all_embeddings.append(cls_embeddings.cpu().numpy())

        # Crucial for FAISS: Even if the model computed these in float16,
        # we cast back to float32 here before returning.
        return np.vstack(all_embeddings).astype("float32")
