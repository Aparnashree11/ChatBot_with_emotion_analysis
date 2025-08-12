import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Core dependencies
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

# Import embedding pipeline (from previous step)
from embeddings_gen import FAQEmbeddingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Structured response from the chatbot"""
    answer: str
    confidence_score: float
    sources_used: int
    response_time_seconds: float
    status: str  # "success", "no_context", "error"

class TinyLlamaLLM:
    """ULTRA-FAST TinyLlama model optimized for speed"""
    
    def __init__(self, device: str = "auto", ultra_fast_mode: bool = True):
        """
        Initialize TinyLlama model with ultra-fast optimizations
        
        Args:
            device: Device to run on ("auto", "cpu", "cuda")  
            ultra_fast_mode: Enable aggressive speed optimizations
        """
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = self._setup_device(device)
        self.ultra_fast_mode = ultra_fast_mode
        
        logger.info(f"Loading ULTRA-FAST TinyLlama on {self.device}")
        self._load_model()
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load TinyLlama with ULTRA-FAST settings"""
        try:
            logger.info("Loading tokenizer...")
            # Load tokenizer with minimal settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True,
                use_fast=True  # Use fast tokenizer
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model...")
            # Ultra-fast model loading - minimal settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,  # Use float32 for CPU speed
                "low_cpu_mem_usage": True,
                "device_map": None,  # Don't use device_map for speed
            }
            
            # Load model without quantization for speed
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Force to specific device
            self.model = self.model.to(self.device)
            
            # Put model in eval mode for speed
            self.model.eval()
            
            # Pre-compile for faster inference (if available)
            if hasattr(torch, 'compile') and self.ultra_fast_mode:
                logger.info("Compiling model for speed...")
                self.model = torch.compile(self.model, mode="max-autotune")
            
            logger.info("TinyLlama loaded with ULTRA-FAST optimizations")
            
        except Exception as e:
            logger.error(f"Failed to load TinyLlama: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(self, prompt: str) -> str:
        """Balanced generation - fast but maintains quality"""
        try:
            # Keep more context for better answers
            if len(prompt) > 600:
                prompt = prompt[-600:]  # Reasonable context length
            
            # Better tokenization for quality
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=400,  # Balanced length
                add_special_tokens=True  # Keep special tokens for quality
            )
            
            # Move to device
            inputs = inputs.to(self.device)
            
            # Balanced generation - fast but quality-aware
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=60,     # Longer responses for better quality
                    do_sample=False,       # Still greedy for speed
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    early_stopping=True,   # Stop at EOS for natural endings
                    output_scores=False,
                    return_dict_in_generate=False
                )
            
            # Better decode with cleanup
            new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Better cleanup for quality
            response = self._clean_response_quality(response)
            return response if response else "I can help you with that question."
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I'm sorry, I couldn't process that question properly."
    
    def _clean_response_quality(self, response: str) -> str:
        """Clean response while maintaining quality"""
        response = response.strip()
        
        # Remove common artifacts but keep content
        response = response.replace("A:", "").strip()
        response = response.replace("Answer:", "").strip()
        
        # Split into sentences and take the best ones
        sentences = response.split('.')
        good_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.lower().startswith(('q:', 'question')):
                good_sentences.append(sentence)
                if len(good_sentences) >= 2:  # Max 2 sentences for conciseness
                    break
        
        if good_sentences:
            response = '. '.join(good_sentences)
            if not response.endswith('.'):
                response += '.'
        
        # Reasonable length limit
        if len(response) > 200:
            response = response[:200].rsplit('.', 1)[0] + '.'
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove common artifacts
        response = response.replace("### Response:", "").strip()
        response = response.replace("Response:", "").strip()
        
        # Remove repeated lines
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines:
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines[:3])  # Limit to 3 lines max
        
        # Limit total length
        if len(response) > 300:
            response = response[:300].rsplit(' ', 1)[0] + "..."
        
        return response

class BalancedPromptTemplate:
    """Balanced prompt template - fast but maintains quality"""
    
    def format_context(self, retrieved_faqs: List[Dict]) -> str:
        """Better context formatting for quality"""
        if not retrieved_faqs:
            return "No relevant information available."
        
        # Use top 2 FAQs for better context
        context_parts = []
        for i, faq in enumerate(retrieved_faqs[:2], 1):
            meta = faq['metadata']
            score = faq.get('similarity_score', 0)
            
            # Include FAQs with decent confidence
            if score >= 0.4:
                # Format clearly
                context_part = f"FAQ {i}: {meta['question']}\nAnswer: {meta['answer']}"
                if meta.get('category'):
                    context_part += f" (Category: {meta['category']})"
                context_parts.append(context_part)
        
        return "\n\n".join(context_parts) if context_parts else "No relevant FAQs found."
    
    def create_prompt(self, question: str, retrieved_faqs: List[Dict]) -> str:
        """Create balanced prompt - concise but informative"""
        context = self.format_context(retrieved_faqs)
        
        # Better prompt structure for quality
        prompt = f"""Based on the FAQ information below, answer the customer's question helpfully and accurately.

{context}

Customer Question: {question}
Answer:"""

        return prompt

class LocalRAGChatbot:
    """Local RAG chatbot with TinyLlama"""
    
    def __init__(self, embeddings_path: str, similarity_threshold: float = 0.5, 
                 max_retrieved_faqs: int = 2, device: str = "cpu"):
        """
        Initialize BALANCED local RAG chatbot - speed + quality
        
        Args:
            embeddings_path: Path to FAISS embeddings
            similarity_threshold: Balanced threshold for good matches
            max_retrieved_faqs: Use 2 FAQs for context
            device: Force CPU as it's often faster for small models
        """
        self.similarity_threshold = similarity_threshold
        self.max_retrieved_faqs = max_retrieved_faqs
        
        # Initialize components
        logger.info("Initializing RAG chatbot...")
        self._load_embeddings(embeddings_path)
        self._load_llm(device)
        self._setup_prompt_template()
        
        logger.info("RAG chatbot ready!")
    
    def _load_embeddings(self, embeddings_path: str):
        """Load the embedding pipeline"""
        try:
            self.embedding_pipeline = FAQEmbeddingPipeline()
            self.embedding_pipeline.load_vector_store(embeddings_path)
            logger.info(f"Loaded embeddings from {embeddings_path}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise RuntimeError(f"Embedding loading failed: {e}")
    
    def _load_llm(self, device: str):
        """Load ULTRA-FAST TinyLlama model"""
        try:
            self.llm = TinyLlamaLLM(device=device, ultra_fast_mode=True)
            logger.info("ULTRA-FAST TinyLlama model loaded")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise RuntimeError(f"LLM loading failed: {e}")
    
    def _setup_prompt_template(self):
        """Setup balanced prompt template"""
        self.prompt_template = BalancedPromptTemplate()
        logger.info("Balanced prompt template configured")
    
    def _retrieve_faqs(self, query: str) -> Tuple[List[Dict], float]:
        """Retrieve relevant FAQs"""
        start_time = time.time()
        
        try:
            results = self.embedding_pipeline.search_faqs(
                query=query,
                k=self.max_retrieved_faqs,
                threshold=self.similarity_threshold
            )
            
            retrieval_time = time.time() - start_time
            return results, retrieval_time
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [], 0
    
    def _generate_response(self, query: str, retrieved_faqs: List[Dict]) -> Tuple[str, float]:
        """Generate response using TinyLlama"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self.prompt_template.create_prompt(query, retrieved_faqs)
            
            # Generate response
            response = self.llm.generate(prompt)
            
            generation_time = time.time() - start_time
            return response, generation_time
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            fallback_response = "I apologize, but I'm experiencing technical difficulties. Please contact customer support for assistance."
            return fallback_response, 0
    
    def ask(self, question: str) -> ChatResponse:
        """
        Ask a question and get a response
        
        Args:
            question: User's question
            
        Returns:
            ChatResponse with answer and metadata
        """
        if not question or not question.strip():
            return ChatResponse(
                answer="Please ask a specific question and I'll help you find the answer.",
                confidence_score=0.0,
                sources_used=0,
                response_time_seconds=0.0,
                status="error"
            )
        
        total_start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant FAQs
            retrieved_faqs, retrieval_time = self._retrieve_faqs(question)
            
            if not retrieved_faqs:
                return ChatResponse(
                    answer="I don't have information about that topic in my knowledge base. Please contact our customer support team for assistance.",
                    confidence_score=0.0,
                    sources_used=0,
                    response_time_seconds=retrieval_time,
                    status="no_context"
                )
            
            # Step 2: Generate response
            answer, generation_time = self._generate_response(question, retrieved_faqs)
            
            # Calculate confidence (average similarity score)
            confidence = sum(faq.get('similarity_score', 0) for faq in retrieved_faqs) / len(retrieved_faqs)
            
            total_time = time.time() - total_start_time
            
            return ChatResponse(
                answer=answer,
                confidence_score=round(confidence, 3),
                sources_used=len(retrieved_faqs),
                response_time_seconds=round(total_time, 2),
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return ChatResponse(
                answer="I'm sorry, but I encountered an error processing your request. Please try again.",
                confidence_score=0.0,
                sources_used=0,
                response_time_seconds=time.time() - total_start_time,
                status="error"
            )
    
    def chat_interface(self):
        """Interactive chat interface for local usage"""
        print("=" * 60)
        print("FAQ CHATBOT - TinyLlama")
        print("=" * 60)
        print("Ask questions about our products and services.")
        print("Type 'quit', 'exit', or 'q' to exit.")
        print("Type 'help' for usage tips.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using our FAQ chatbot. Goodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    print("\nUsage Tips:")
                    print("- Ask specific questions: 'How do I return a product?'")
                    continue
                
                if not user_input:
                    continue
                
                print("Bot: Thinking...")
                start_time = time.time()
                response = self.ask(user_input)
                end_time = time.time()
                
                print(f"Bot: {response.answer}")
                
                # Show metadata with actual timing
                actual_time = end_time - start_time
                print(f"\n[Confidence: {response.confidence_score:.2f} | "
                      f"Sources: {response.sources_used} | "
                      f"Time: {actual_time:.1f}s]")
                
                # Show status-specific messages
                if response.status == "no_context":
                    print("Tip: Try rephrasing your question or contact customer support.")
                elif response.confidence_score < 0.5:
                    print("Low confidence - this answer might not be accurate.")
                
            except KeyboardInterrupt:
                print("\n\nThank you for using our FAQ chatbot. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or contact technical support.")

def create_balanced_chatbot(embeddings_path: str, device: str = "cpu") -> LocalRAGChatbot:
    
    # Balanced settings - good speed with maintained quality
    chatbot = LocalRAGChatbot(
        embeddings_path=embeddings_path,
        similarity_threshold=0.5,    # Balanced threshold
        max_retrieved_faqs=2,        # 2 FAQs for better context
        device=device
    )
    
    return chatbot

def main():
    
    embeddings_path = "faq_vector_store"
    if not Path(f"{embeddings_path}.faiss").exists():
        print(f"Embeddings not found at {embeddings_path}")
        return
    
    try:
        print("\nLoading balanced chatbot...")
        
        # Use balanced version for good speed + quality
        chatbot = create_balanced_chatbot(embeddings_path, device="cpu")
        
        print("Chatbot loaded!")
        
        # Quality-focused speed test
        print("\nSpeed & Quality test...")
        test_questions = [
            "How do I return a product?",
            "What are your shipping options?",
            "Do you accept credit cards?"
        ]
        
        total_time = 0
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Testing: '{question}'")
            
            import time
            start = time.time()
            response = chatbot.ask(question)
            elapsed = time.time() - start
            total_time += elapsed
            
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Answer: {response.answer}")
            print(f"   Confidence: {response.confidence_score:.2f}")
            print(f"   Sources: {response.sources_used}")
            
            # Check quality
            if len(response.answer) < 20:
                print("Answer seems too short")
            elif response.confidence_score < 0.4:
                print("Low confidence answer")
            else:
                print("Good quality answer")
        
        avg_time = total_time / len(test_questions)
        print(f"\nAverage response time: {avg_time:.1f} seconds")
        
        if avg_time > 10:
            print("Still slow - try quality-focused version for better results")
        elif avg_time < 3:
            print("Great speed!")
        else:
            print("Good balance of speed and quality")
        
        chatbot.chat_interface()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
