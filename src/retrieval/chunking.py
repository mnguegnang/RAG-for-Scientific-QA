from transformers import AutoTokenizer
from typing import List, Dict

class QasperChunker:
    def __init__(self, model_name="allenai/specter2_base", max_tokens=500, overlap_pct=0.1):
        """
        Initializes the chunker.
        
        CRITICAL SOURCING: 
        1. model_name: MUST be "allenai/specter2_base" to match your DenseIndexer. 
           Chunking must be done using the bottleneck model's tokenizer.
        2. max_tokens: Set to 500 (leaving 12 tokens for [CLS] and [SEP] special tokens 
           required by SPECTER2's 512 absolute limit).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
<<<<<<< HEAD
        
        # Prevent HuggingFace from throwing console warnings when calculating length
        self.tokenizer.model_max_length = int(1e30)

=======
        
        # Prevent HuggingFace from throwing console warnings when calculating length
        self.tokenizer.model_max_length = int(1e30)
        
>>>>>>> cc6e01ad33bfbf2fa9000592545c986b7eeb4561
        self.max_tokens = max_tokens
        # Calculate overlap tokens (e.g., 500 * 0.1 = 50 tokens)
        self.overlap_tokens = int(max_tokens * overlap_pct)

    def split_large_text(self, text: str) -> List[str]:
        """
        Splits a single long text string into overlapping chunks based on token count.
        """
        # add_special_tokens=False because the embedding model will add [CLS]/[SEP] during the encode step
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Case 1: Text fits in one chunk (common for academic paragraphs)
        if len(tokens) <= self.max_tokens:
            return [text]
        
        # Case 2: Sliding Window for oversized paragraphs
        chunks = []
        stride = self.max_tokens - self.overlap_tokens
        
        if stride <= 0:
            raise ValueError("Overlap percentage is too high, stride must be > 0")
        
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    def process_paper(self, paper_data: Dict) -> List[Dict]:
        """
        Flattens a QASPER paper into a list of chunk dictionaries with metadata.
        """
        chunks = []
        paper_id = paper_data['id']
        title = paper_data['title']
        
        sections = paper_data['full_text']['section_name']
        paragraphs_list = paper_data['full_text']['paragraphs']

        for section_idx, (section_name, section_paras) in enumerate(zip(sections, paragraphs_list)):
            for para_idx, paragraph in enumerate(section_paras):
                
                # Base Metadata
                base_meta = {
                    "paper_id": paper_id,
                    "title": title,
                    "section_name": section_name,
                    "original_para_id": f"{paper_id}_{section_idx}_{para_idx}"
                }

                # Split paragraph using SPECTER2 constraints
                text_chunks = self.split_large_text(paragraph)
                
                for sub_idx, text in enumerate(text_chunks):
                    chunk = base_meta.copy()
                    chunk["chunk_id"] = f"{base_meta['original_para_id']}_{sub_idx}"
                    # Anthropic (2024) Contextual Retrieval — prepend document-level
                    # labels so SPECTER2 encodes *which* paper and section each passage
                    # belongs to. This anchors the chunk vector in a paper-specific
                    # region of the embedding space rather than a generic topic space,
                    # improving retrieval recall by up to 49% on scientific corpora.
                    chunk["text"] = (
                        f"Title: {title or 'Unknown'}. "
                        f"Section: {section_name or 'Unknown'}.\n"
                        + text
                    )
                    chunks.append(chunk)
                    
        return chunks