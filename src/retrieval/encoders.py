"""
SPECTER2 Encoder Wrapper
------------------------
Wraps the adapter-based allenai/specter2_base model behind the same
.encode(texts, normalize_embeddings, batch_size, show_progress_bar) interface
that sentence_transformers.SentenceTransformer exposes, so DenseIndexer and
HybridRetriever can use it as a drop-in replacement.

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

    def __init__(self, device: str = _DEVICE):
        try:
            from adapters import AutoAdapterModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'adapters' package is required for SPECTER2. "
                "Install it with: pip install adapters"
            ) from e

        logger.info("Loading SPECTER2 tokenizer from %s ...", self.BASE_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)

        logger.info("Loading SPECTER2 base model ...")
        self.model = AutoAdapterModel.from_pretrained(self.BASE_MODEL)

        logger.info("Loading SPECTER2 retrieval adapter (%s) ...", self.ADAPTER_NAME)
        self.model.load_adapter(
            self.ADAPTER_NAME,
            source="hf",
            load_as="specter2_retrieval",
            set_active=True,
        )

        self.device = device
        self.model.to(device)
        self.model.eval()
        logger.info("SPECTER2 encoder ready on device=%s", device)

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
            Number of texts per forward pass.
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
                outputs = self.model(**inputs)

            # CLS-token pooling — recommended by AllenAI for SPECTER2
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            if normalize_embeddings:
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)

            all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype("float32")
