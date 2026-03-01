import requests
import logging
import json
import sys
from typing import List, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LocalLLMGenerator:
    def __init__(self, model_name: str = "llama3", api_url: str = "http://localhost:11434/api/generate"):
        """
        Initializes the connection to the local Ollama instance.
        
        Args:
            model_name (str): The name of the model pulled in Ollama (e.g., 'llama3' or 'mistral') our case is llama3.
            api_url (str): The REST API endpoint for Ollama generation.
        """
        self.model_name = model_name
        self.api_url = api_url
        logging.info(f"Initialized LocalLLMGenerator using model: {self.model_name}")

    def _build_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Constructs the Chain-of-Thought and Citation prompt.
        """
        # 1. Format the context
        context_str = ""
        for i, doc in enumerate(retrieved_docs, 1):
            # Extracting metadata gracefully
            meta = doc.get('metadata', {})
            source = meta.get('source', 'Unknown Source')
            page = meta.get('page', 'N/A')
            
            context_str += f"[Doc {i}]\nSource: {source} (Page: {page})\nContent: {doc.get('text', '')}\n\n"

        # 2. Construct the strict instructional prompt
        prompt = f"""You are a precise scientific AI research assistant. Answer the user's query based ONLY on the provided context.

<Context>
{context_str}
</Context>

<Instructions>
1. Comprehension: Read the context carefully. If the context does not contain the answer, reply exactly with: "The retrieved documents do not contain enough information to answer this." Do not guess.
2. Chain of Thought: Provide a brief <Reasoning> section where you outline your logic based on the documents.
3. Citations: Provide a <Final Answer> section. Every factual claim MUST end with the citation tag of the document it originated from, e.g., [Doc 1] or [Doc 2].
</Instructions>

<User Query>
{query}

<Output Format>
<Reasoning>
(your step-by-step thinking)
</Reasoning>
<Final Answer>
(your synthesized, cited answer)
</Final Answer>
"""
        return prompt

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Sends the constructed prompt to the local LLM and retrieves the answer.
        """
        if not retrieved_docs:
            return "No documents were retrieved. Cannot generate an answer."

        full_prompt = self._build_prompt(query, retrieved_docs)
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True, # We set this to True to get a streaming response token by token, which is crucial for long responses and better user experience
            "options": {
                "temperature": 0.1 # Low temperature to reduce hallucination and increase factual determinism
            }
        }


        try:
            logging.info("Sending prompt to local LLM. Awaiting stream...")
            
            response = requests.post(self.api_url, json=payload, stream=True, timeout=300)
            response.raise_for_status() 
            
            print("\n" + "="*40 + " LLM OUTPUT " + "="*40 + "\n")
            
            full_response = ""
            char_count = 0
            
            # The Safe JSONL Parsing Loop
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    body = json.loads(decoded_line)
                    token = body.get("response", "")
                    
                    full_response += token
                    
                    # Formatting / Text Wrapping Logic
                    if token == "\n":
                        char_count = 0
                        sys.stdout.write(token)
                    else:
                        char_count += len(token)
                        if char_count >= 80 and token.startswith(" "):
                            sys.stdout.write("\n" + token.lstrip())
                            char_count = len(token.lstrip())
                        else:
                            sys.stdout.write(token)
                            
                    sys.stdout.flush()
                    
                    if body.get("done"):
                        break
                        
            print("\n\n" + "="*92 + "\n")
            return full_response
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Ollama stream. Make sure stream=True is used correctly: {e}")
            return "System Error: JSON Parsing Failure."
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            return f"System Error: Connection failure. {e}"