from transformers import AutoTokenizer
from typing import List, Dict

class QasperChunker:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", max_tokens=512, overlap_pct=0.1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        # Calculate overlap tokens (e.g., 512 * 0.1 = ~51 tokens)
        self.overlap_tokens = int(max_tokens * overlap_pct)

    def split_large_text(self, text: str) -> List[str]:
        """
        Splits a single long text string into overlapping chunks based on token count.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Case 1: Text fits in one chunk
        if len(tokens) <= self.max_tokens:
            return [text]
        
        # Case 2: Sliding Window
        chunks = []
        stride = self.max_tokens - self.overlap_tokens
        
        # range(start, stop, step) -> We step by 'stride', not 'max_tokens'
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    def process_paper(self, paper_data: Dict) -> List[Dict]:
        """
        Flattens a paper into a list of chunk dictionaries with metadata.
        """
        chunks = []
        paper_id = paper_data['id']
        title = paper_data['title']
        
        # Iterate over sections
        sections = paper_data['full_text']['section_name']
        paragraphs_list = paper_data['full_text']['paragraphs']

        for section_idx, (section_name, section_paras) in enumerate(zip(sections, paragraphs_list)):
            # Iterate over paragraphs in that section
            for para_idx, paragraph in enumerate(section_paras):
                
                # Base Metadata (Crucial for Citation)
                base_meta = {
                    "paper_id": paper_id,
                    "title": title,
                    "section_name": section_name,
                    "original_para_id": f"{paper_id}_{section_idx}_{para_idx}"
                }

                # Split paragraph if needed
                text_chunks = self.split_large_text(paragraph)
                
                for sub_idx, text in enumerate(text_chunks):
                    # Create the final chunk object
                    chunk = base_meta.copy()
                    chunk["chunk_id"] = f"{base_meta['original_para_id']}_{sub_idx}"
                    chunk["text"] = text
                    chunks.append(chunk)
                    
        return chunks