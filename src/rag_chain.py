import time
import torch

from langchain_huggingface import HuggingFacePipeline
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from .retrieval.reranker import CrossEncoderReranker
from .retrieval.query_parser import QueryParser

class RAGChain:

    @staticmethod
    def _check_ollama_available(model_name):
        """
        Checking if Ollama is running and the model is available function.

        Args:
            model_name: Ollama model name to check for

        Returns:
            result: True if Ollama is running and model is available
        """

        try:
            import urllib.request
            import json
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read())
                models = [m["name"] for m in data.get("models", [])]
                # Checking for exact match or prefix match (e.g., "qwen2:1.5b" matches "qwen2:1.5b-instruct")
                return any(model_name in m or m.startswith(model_name.split(":")[0]) for m in models)
        except Exception:
            return False

    def __init__(self, embedding_generator, retriever, llm_model, llm_provider,
        ollama_model = "qwen2:1.5b", use_quantization = True, quantization_bits = 4,
        enable_prompt_cache = True):
        """
        Initialize RAG chain.

        Args:
            embedding_generator: Embedding generator instance
            retriever: Retriever instance (semantic, BM25, or hybrid)
            llm_model: LLM model name (HuggingFace format)
            llm_provider: LLM provider ("auto", "ollama", or "huggingface")
            ollama_model: Ollama model name (used when provider is "auto" or "ollama")
            use_quantization: Quantization for memory efficiency
            quantization_bits: Quantization bits (4 or 8)
            enable_prompt_cache: Enable prompt caching for faster generation (HuggingFace only)
        """

        # Auto-detecting provider: try Ollama first, fall back to HuggingFace
        if llm_provider == "auto":
            if self._check_ollama_available(ollama_model):
                llm_provider = "ollama"
                llm_model = ollama_model
                print(f"Auto-detected Ollama with model '{ollama_model}'")
            else:
                llm_provider = "huggingface"
                print(f"Ollama not available, falling back to HuggingFace with '{llm_model}'")

        self.retriever = retriever
        self.llm_provider = llm_provider
        self.embedding_generator = embedding_generator

        # Inject embedding_generator into retriever if it needs one but wasn't given one.
        # Supports callers that pass embedding_generator only to RAGChain (legacy pattern).
        if embedding_generator:
            if hasattr(retriever, 'embedding_generator') and retriever.embedding_generator is None:
                retriever.embedding_generator = embedding_generator
            # For HybridRetriever: also inject into the inner SemanticRetriever
            if hasattr(retriever, 'semantic_retriever'):
                sem = retriever.semantic_retriever
                if hasattr(sem, 'embedding_generator') and sem.embedding_generator is None:
                    sem.embedding_generator = embedding_generator

        self.query_parser = QueryParser()
        self.reranker = CrossEncoderReranker()

        if llm_provider == "ollama":
            self.llm = ChatOllama(model = llm_model, temperature = 0)

        elif llm_provider == "huggingface":
            if torch.backends.mps.is_available():
                device = "mps"
                device_map = "mps"

            elif torch.cuda.is_available():
                device = "cuda"
                device_map = "auto"

            else:
                device = "cpu"
                device_map = "cpu"
            
            model_kwargs = {}
            
            if device == "mps":
                model_kwargs = {
                    "device_map": "mps",
                    "dtype": torch.float16,
                }

                print(f"Using MPS (Metal) acceleration on Mac")
                pipeline_device = None

            elif device == "cuda":
                if use_quantization and quantization_bits in [4, 8]:
                    try:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit = (quantization_bits == 4),
                            load_in_8bit = (quantization_bits == 8),
                            bnb_4bit_compute_dtype=torch.float16
                        )

                        model_kwargs = {
                            "device_map": "auto",
                            "quantization_config": quantization_config,
                        }
                        print(f"Using {quantization_bits}-bit quantization on CUDA")
                        pipeline_device = None

                    except ImportError:
                        model_kwargs = {
                            "device_map": "auto",
                            "dtype": torch.float16,
                        }
                        pipeline_device = None

                else:
                    model_kwargs = {
                        "device_map": "auto",
                        "dtype": torch.float16,
                    }
                    pipeline_device = None

            else:
                model_kwargs = {
                    "dtype": torch.float16,
                }
                pipeline_device = -1

                print("Using CPU (consider using a smaller model or quantization)")
            
            model = AutoModelForCausalLM.from_pretrained(llm_model, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(llm_model)
            
            # Creating the pipeline
            pipe_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "max_new_tokens": 1024,
                "return_full_text": False,
            }

            if pipeline_device is not None:
                pipe_kwargs["device"] = pipeline_device
            
            pipe = pipeline("text-generation", **pipe_kwargs)
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.tokenizer = tokenizer
            self.llm_model = llm_model
            self.model = model
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'ollama' or 'huggingface'")
        
        # Defining the all important system prompt
        self.system_prompt = """You are a helpful research assistant. Answer questions using the provided context from research papers.
                        CITATION RULES:
                        - Each source is numbered [1], [2], [3], etc.
                        - ALWAYS include citation numbers when mentioning information from sources.
                        - Every sentence ending that uses source information must have citations.

                        ANSWER REQUIREMENTS:
                        - Provide a comprehensive, detailed answer.
                        - Elaborate on each approach/method mentioned.
                        - Explain how different approaches work and what they evaluate.
                        - Follow CITATION RULES when including specific details from the sources.

                        Example: "RUTEd evaluations [1] assess reliability. FACT-AUDIT [2] provides a framework. MIRAGE [3] explores complex scenarios."

                        If you cannot answer from the context, say so."""
        
        # Creating prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            ("human", "{user_message}")
        ])
        
        # Trying Prompt Caching :P
        if self.llm_provider == "huggingface" and enable_prompt_cache:
            self.prompt_cache_enabled = True
            self.cached_prompt_kv_cache = None
            self.cached_prompt_length = 0
            self._initialize_prompt_cache()
        else:
            self.prompt_cache_enabled = False
            self.cached_prompt_kv_cache = None
            self.cached_prompt_length = 0
        
        # Building the RAG chain using LangChain chains
        self._build_rag_chain()
    
    def _format_user_message(self, context, question):
        """
        Formatting user message function.
        
        Args:
            context: Retrieved context from papers
            question: User question
            
        Returns:
            Formatted user message string
        """
        
        return f"""Context from research papers:
                    {context}

                    Question: {question}

                    Answer (provide a detailed, comprehensive answer with citations):"""
    
    def _get_model_device(self):
        """
        Getting model device function.
        
        Returns:
            Device where the model is located
        """
        
        # Fetching the "device", very important for production-grade systems :)
        if hasattr(self.model, 'device'):
            return self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            return next(iter(self.model.hf_device_map.values()))
        else:
            return torch.device("cpu")
    
    def _build_rag_chain(self):
        """
        Building RAG chain using LangChain chains function.
        """
        
        # Block 1: Parse filters and prepare inputs
        def prepare_inputs(inputs: dict) -> dict:

            question = inputs.get("question", "")

            # Ensure question is a string (handle nested dict case)
            if isinstance(question, dict):
                question = question.get("question", "")
            if not isinstance(question, str):
                question = str(question)

            auto_parse_filters = inputs.get("auto_parse_filters", True)
            filters = inputs.get("filters")

            # Auto-parsing filters from query if enabled
            if auto_parse_filters and filters is None:
                _, parsed_filters = self.query_parser.parse(question)
                filters = parsed_filters

            return {
                "question": question,
                "filters": filters,
                "initial_retrieval_k": inputs.get("initial_retrieval_k", 20),
                "rerank_k": inputs.get("rerank_k", 3),
                "top_k": inputs.get("top_k", 5)
            }

        # Block 2: Retrieve chunks via unified LangChain retriever interface
        def retrieve(inputs: dict) -> dict:

            question = inputs["question"]
            filters = inputs.get("filters")
            initial_retrieval_k = inputs["initial_retrieval_k"]

            # All retrievers are LangChain-compatible via _get_relevant_documents;
            # embedding is handled internally by each retriever
            docs = self.retriever._get_relevant_documents(
                question, top_k=initial_retrieval_k, filters=filters
            )

            # Convert Documents to dict format expected by the reranker
            initial_chunks = [
                {
                    "text": doc.page_content,
                    "metadata": {k: v for k, v in doc.metadata.items()
                                 if k not in ("score", "rank", "retriever", "detected_sections")},
                    "score": doc.metadata.get("score", 0),
                    "rank": doc.metadata.get("rank", 0)
                }
                for doc in docs
            ]

            return {
                "question": question,
                "initial_chunks": initial_chunks,
                "filters": filters,
                "rerank_k": inputs["rerank_k"],
                "top_k": inputs["top_k"]
            }
        
        # Block 3: Re-ranking
        def rerank(inputs: dict) -> dict:
            
            question = inputs["question"]
            initial_chunks = inputs["initial_chunks"]
            rerank_k = inputs["rerank_k"]
            
            # Re-ranking chunks
            reranked_chunks = self.reranker.rerank(question, initial_chunks, top_k=rerank_k)
            
            return {
                "question": question,
                "reranked_chunks": reranked_chunks,
                "filters": inputs.get("filters"),
                "top_k": inputs["top_k"]
            }
        
        # Block 4: Formatting
        def format_context(inputs: dict) -> dict:
            
            reranked_chunks = inputs["reranked_chunks"]
            question = inputs["question"]
            top_k = inputs["top_k"]
            
            # Building context with numbered citations
            context_parts = []
            citation_map = {}
            
            for idx, chunk in enumerate(reranked_chunks, start=1):
                metadata = chunk["metadata"]
                citation_num = f"[{idx}]"
                
                # Creating citation mapping
                citation_map[idx] = {
                    "chunk_id": chunk.get("chunk_id"),
                    "title": metadata.get('title', 'Unknown'),
                    "conference": metadata.get('conference', 'Unknown'),
                    "year": metadata.get('year', 'Unknown'),
                    "text_preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                }
                
                # Formatting context with numbered citation
                title = metadata.get('title', 'Unknown')
                conference = metadata.get('conference', 'Unknown')
                year = metadata.get('year', 'Unknown')
                paper_info = f"Source {citation_num}: {title} ({conference} {year})"
                context_parts.append(f"{paper_info}\nContent: {chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Preparing sources for return
            sources = []
            for idx, chunk in enumerate(reranked_chunks[:top_k], start=1):
                sources.append({
                    "citation_number": idx,
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": chunk.get("score", 0),  # Reranked score
                    "original_score": chunk.get("original_score", None)  # Original retrieval score
                })
            
            return {
                "question": question,
                "context": context,
                "citation_map": citation_map,
                "sources": sources,
                "filters": inputs.get("filters")
            }
        
        # Composing the chain using LangChain runnables
        self.rag_chain = (
            RunnableLambda(prepare_inputs)
            | RunnableLambda(retrieve)
            | RunnableLambda(rerank)
            | RunnableLambda(format_context)
        )
    
    def _apply_chat_template(self, messages, add_generation_prompt = True):
        """
        Applying chat template function.
        
        Args:
            messages: List of message dictionaries with role and content
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Formatted prompt string
        """
        
        # Applying the Chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize = False,
                    add_generation_prompt = add_generation_prompt
                )
            except Exception:
                return "\n\n".join([msg.get("content", "") for msg in messages])
        else:
            return "\n\n".join([msg.get("content", "") for msg in messages])
    
    def _initialize_prompt_cache(self):
        """
        Initializing prompt cache function.
        
        """
        
        if not self.prompt_cache_enabled or self.llm_provider != "huggingface":
            return
        
        try:
            # Formatting the system prompt using chat template if available
            cached_prompt = self._apply_chat_template(
                [{"role": "system", "content": self.system_prompt}],
                add_generation_prompt = False
            )
            
            # Tokenizing the prompt
            inputs = self.tokenizer(cached_prompt, return_tensors="pt")
            
            # Moving inputs to the same device as the model
            device = self._get_model_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generating KV cache for the static prompt
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=True, output_attentions=False)
                self.cached_prompt_kv_cache = outputs.past_key_values
                self.cached_prompt_length = inputs['input_ids'].shape[1]
            
            print(f"Prompt cache initialized ({self.cached_prompt_length} tokens cached)")
        
        except Exception as e:

            self.prompt_cache_enabled = False
            self.cached_prompt_kv_cache = None
    
    def _stream_with_cache(self, user_message, max_new_tokens = 1024):
        """
        Streaming generation using cached KV cache function.
        
        Args:
            user_message: The variable part (context + question)
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Generated tokens one at a time
        """
        
        if not self.prompt_cache_enabled or self.cached_prompt_kv_cache is None:
            
            # Regular streaming
            full_prompt = f"{self.system_prompt}\n\n{user_message}"
            for chunk in self.llm.stream(full_prompt):
                yield chunk.text if hasattr(chunk, 'text') else str(chunk)

            return
        
        try:
            
            # Getting the device
            device = self._get_model_device()
            
            # Building the full prompt for tokenization
            full_prompt = self._apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                add_generation_prompt=True
            )
            
            # Tokenizing the full prompt
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            
            # Processing the new part with cached KV cache
            if input_ids.shape[1] > self.cached_prompt_length:
                new_ids = input_ids[:, self.cached_prompt_length:]
                past_key_values = self.cached_prompt_kv_cache
                
                # Processing new tokens using cached KV cache
                with torch.no_grad():
                    outputs = self.model(
                        input_ids = new_ids,
                        past_key_values = past_key_values,
                        use_cache = True
                    )

                    past_key_values = outputs.past_key_values
                    current_ids = input_ids[:, -1:]
            else:
                # Wild scenario where the new user message tokens < cached prompt tokens
                past_key_values = self.cached_prompt_kv_cache
                current_ids = input_ids
            
            # Streaming generation using cached KV cache
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids = current_ids,
                        past_key_values = past_key_values,
                        use_cache = True
                    )
                
                # Getting logits and sampling next token
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Decoding the token
                token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                yield token
                
                # Checking for end of generation
                if self.tokenizer.eos_token_id and next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                # Updating for next iteration
                current_ids = next_token_id
                past_key_values = outputs.past_key_values
                
        except Exception as e:
            
            # Regular streaming
            full_prompt = f"{self.system_prompt}\n\n{user_message}"
            for chunk in self.llm.stream(full_prompt):
                yield chunk.text if hasattr(chunk, 'text') else str(chunk)
    
    def query(self, question, top_k = 5, initial_retrieval_k = 20, rerank_k = 3, filters = None, 
        auto_parse_filters = True):
        """
        Query function using LangChain chains.
        
        Args:
            question: User query
            top_k: Number of chunks to return in sources (after re-ranking)
            initial_retrieval_k: Number of intial chunks
            rerank_k: Number of top chunks delivered to model
            filters: Dictionary of metadata filters
            auto_parse_filters: If True, automatically extract filters from the query
            
        Yields:
            Tuples of (token, metadata_dict) with timing metrics, answer, sources, and citations
        """
        
        # Using the LangChain chain to process the query
        chain_input = {
            "question": question,
            "filters": filters,
            "auto_parse_filters": auto_parse_filters,
            "initial_retrieval_k": initial_retrieval_k,
            "rerank_k": rerank_k,
            "top_k": top_k
        }
        
        # Invoking the chain :)
        chain_result = self.rag_chain.invoke(chain_input)
        
        # Results time :)
        context = chain_result["context"]
        citation_map = chain_result["citation_map"]
        sources = chain_result["sources"]
        question = chain_result["question"]  # Extract question from chain result
        filters = chain_result.get("filters")
        
        # Streaming generation
        start_time = time.time()
        first_token_time = None
        answer = ""
        
        if self.llm_provider == "ollama":

            user_message = self._format_user_message(context, question)
            messages = self.prompt_template.format_messages(user_message=user_message)
            
            for chunk in self.llm.stream(messages):
                if first_token_time is None:
                    first_token_time = time.time()
                    tfft = first_token_time - start_time
                else:
                    tfft = None
                
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                answer += token
                
                yield (token, {
                    "tfft": tfft,
                    "total_time": time.time() - start_time,
                    "complete": False,
                    "answer": answer,
                    "sources": sources,
                    "citation_map": citation_map,
                    "question": question,
                    "filters_applied": filters if filters else None
                })
        else:

            # HuggingFace streaming
            user_msg = self._format_user_message(context, question)
            
            # Using chat template if available
            prompt = self._apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                add_generation_prompt=True
            )
            
            # Using cached streaming if available, otherwise regular streaming
            if self.prompt_cache_enabled and self.cached_prompt_kv_cache is not None:
                token_stream = self._stream_with_cache(user_msg, max_new_tokens=1024)
            else:
                token_stream = (chunk.text if hasattr(chunk, 'text') else str(chunk) 
                              for chunk in self.llm.stream(prompt))
            
            for token in token_stream:
                if first_token_time is None:
                    first_token_time = time.time()
                    tfft = first_token_time - start_time
                else:
                    tfft = None
                
                answer += token
                
                yield (token, {
                    "tfft": tfft,
                    "total_time": time.time() - start_time,
                    "complete": False,
                    "answer": answer,
                    "sources": sources,
                    "citation_map": citation_map,
                    "question": question,
                    "filters_applied": filters if filters else None
                })
        
        # Final yield with complete flag
        total_time = time.time() - start_time
        yield ("", {
            "tfft": first_token_time - start_time if first_token_time else None,
            "total_time": total_time,
            "complete": True,
            "answer": answer,
            "sources": sources,
            "citation_map": citation_map,
            "question": question,
            "filters_applied": filters if filters else None
        })