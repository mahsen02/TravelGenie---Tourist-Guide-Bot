import gradio as gr
import os
import time
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
from deep_translator import GoogleTranslator
# Core libraries
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

# Translation and TTS
from googletrans import Translator
import gtts
import io
import tempfile

# Web scraping and API calls
import requests
from bs4 import BeautifulSoup
import wikipedia
import urllib.parse

# Speech recognition
import speech_recognition as sr
from pydub import AudioSegment

# LLM API clients
import groq
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TouristGuideBot:
    def __init__(self):
        """Initialize the Tourist Guide Bot with all necessary components."""
        
        # Initialize components
        self.translator = Translator()
        self.embedding_model = None
        self.faiss_index = None
        self.knowledge_base = []
        self.groq_client = None
        self.gemini_model = None
        
        # Supported languages
        self.languages = {
            "English": "en",
            "Urdu": "ur", 
            "Arabic": "ar",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
            "Italian": "it",
            "Chinese": "zh",
            "Japanese": "ja",
            "Hindi": "hi"
        }
        
        # Initialize models and data
        self.setup_models()
        self.load_or_create_knowledge_base()
        
    def setup_models(self):
        """Setup embedding model and LLM APIs."""
        try:
            # Load multilingual sentence transformer
            self.embedding_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
            logger.info("✅ Embedding model loaded successfully")
            
            # Setup API clients (add your API keys)
            groq_api_key = os.getenv('GROQ_API_KEY')
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            if groq_api_key:
                self.groq_client = groq.Groq(api_key=groq_api_key)
                logger.info("✅ Groq API client initialized")
            
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = GenerativeModel('gemini-pro')
                logger.info("✅ Gemini API client initialized")
                
        except Exception as e:
            logger.error(f"❌ Error setting up models: {e}")
    
    def load_or_create_knowledge_base(self):
        """Load existing knowledge base or create a new one."""
        kb_path = "travel_knowledge_base.pkl"
        index_path = "faiss_index.idx"
        
        try:
            # Try to load existing knowledge base
            if os.path.exists(kb_path) and os.path.exists(index_path):
                with open(kb_path, 'rb') as f:
                    self.knowledge_base = pickle.load(f)
                self.faiss_index = faiss.read_index(index_path)
                logger.info(f"✅ Loaded existing knowledge base with {len(self.knowledge_base)} chunks")
            else:
                # Create new knowledge base
                self.create_knowledge_base()
                
        except Exception as e:
            logger.error(f"❌ Error loading knowledge base: {e}")
            self.create_knowledge_base()
    
    def create_knowledge_base(self):
        """Create knowledge base from sample travel data."""
        logger.info("🔄 Creating new knowledge base...")
        
        # Sample travel knowledge (in practice, load from files/Wikipedia)
        sample_data = [
            {
                "content": "Paris is the capital city of France, known for its iconic Eiffel Tower, Louvre Museum, and romantic atmosphere. Best visited in spring or fall.",
                "location": "Paris, France",
                "category": "destination"
            },
            {
                "content": "Dubai offers luxury shopping, ultramodern architecture, and desert safaris. The Burj Khalifa is the world's tallest building.",
                "location": "Dubai, UAE", 
                "category": "destination"
            },
            {
                "content": "Tokyo combines traditional Japanese culture with cutting-edge technology. Visit temples, enjoy sushi, and experience the bustling city life.",
                "location": "Tokyo, Japan",
                "category": "destination"
            },
            {
                "content": "New York City offers Broadway shows, world-class museums, Central Park, and diverse neighborhoods like Times Square and Brooklyn.",
                "location": "New York, USA",
                "category": "destination"
            },
            {
                "content": "Always carry a universal adapter, pack light, research local customs, and keep copies of important documents when traveling internationally.",
                "location": "General",
                "category": "travel_tips"
            },
            {
                "content": "Book flights 2-3 months in advance for best prices. Use flight comparison websites and be flexible with dates.",
                "location": "General", 
                "category": "travel_tips"
            },
            {
                "content": "Istanbul bridges Europe and Asia, featuring the Blue Mosque, Hagia Sophia, and Grand Bazaar. Turkish cuisine is exceptional.",
                "location": "Istanbul, Turkey",
                "category": "destination"
            },
            {
                "content": "Rome offers ancient history with the Colosseum, Vatican City, Trevi Fountain, and delicious Italian food.",
                "location": "Rome, Italy",
                "category": "destination"
            }
        ]
        
        # Create embeddings for all content
        texts = [item["content"] for item in sample_data]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store knowledge base
        self.knowledge_base = sample_data
        
        # Save to disk
        with open("travel_knowledge_base.pkl", 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        faiss.write_index(self.faiss_index, "faiss_index.idx")
        
        logger.info(f"✅ Created knowledge base with {len(self.knowledge_base)} chunks")
    

    def translate_text(self, text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
        try:
            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[str]:
        """Search Wikipedia for relevant travel information."""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=max_results)
            summaries = []
            
            for title in search_results[:max_results]:
                try:
                    summary = wikipedia.summary(title, sentences=3)
                    summaries.append(f"Wikipedia - {title}: {summary}")
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=3)
                        summaries.append(f"Wikipedia - {e.options[0]}: {summary}")
                    except:
                        continue
                except:
                    continue
            
            return summaries
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def search_duckduckgo(self, query: str, max_results: int = 3) -> List[str]:
        """Search DuckDuckGo for travel information."""
        try:
            # DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/"
            params = {
                'q': query + " travel guide",
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            
            # Extract abstract
            if data.get('Abstract'):
                results.append(f"DuckDuckGo: {data['Abstract']}")
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:2]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(f"DuckDuckGo: {topic['Text']}")
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def retrieve_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve similar chunks from knowledge base using FAISS."""
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            # Get relevant chunks
            similar_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.knowledge_base):
                    chunk = self.knowledge_base[idx].copy()
                    chunk['similarity_score'] = float(distances[0][i])
                    similar_chunks.append(chunk)
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def query_llm(self, prompt: str) -> str:
        """Query LLM using available APIs."""
        try:
            # Try Groq first
            if self.groq_client:
                try:
                    completion = self.groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful and knowledgeable tourist guide assistant. Provide accurate, helpful, and engaging travel advice."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        model="llama3-8b-8192",
                        temperature=0.7,
                        max_tokens=1024
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    logger.error(f"Groq API error: {e}")
            
            # Try Gemini as fallback
            if self.gemini_model:
                try:
                    response = self.gemini_model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
            
            # Fallback response if no API available
            return "I'm sorry, but I'm currently unable to access the AI models. Please check your API keys and try again."
            
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return "An error occurred while processing your request. Please try again."
    
    def process_audio_input(self, audio_file) -> str:
        """Convert speech to text."""
        if audio_file is None:
            return ""
            
        try:
            # Initialize recognizer
            r = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)
            
            # Convert speech to text
            text = r.recognize_google(audio)
            return text
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return "Could not understand audio. Please try again."
    
    def generate_speech(self, text: str, lang_code: str = "en") -> str:
        """Generate speech from text using gTTS."""
        try:
            tts = gtts.gTTS(text=text, lang=lang_code, slow=False)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def answer_question(self, question: str, language: str, enable_tts: bool = True) -> Tuple[str, Optional[str]]:
        """Main function to answer user questions."""
        try:
            lang_code = self.languages.get(language, "en")
            
            # Step 1: Translate question to English if needed
            if lang_code != "en":
                english_question = self.translate_text(question, target_lang="en")
            else:
                english_question = question
            
            # Step 2: Retrieve similar chunks from knowledge base
            similar_chunks = self.retrieve_similar_chunks(english_question, k=3)
            
            # Step 3: Get live web data
            wikipedia_results = self.search_wikipedia(english_question)
            duckduckgo_results = self.search_duckduckgo(english_question)
            
            # Step 4: Build context from all sources
            context_parts = []
            
            # Add knowledge base chunks
            if similar_chunks:
                context_parts.append("From Knowledge Base:")
                for chunk in similar_chunks:
                    context_parts.append(f"- {chunk['content']}")
            
            # Add Wikipedia results
            if wikipedia_results:
                context_parts.append("\nFrom Wikipedia:")
                for result in wikipedia_results:
                    context_parts.append(f"- {result}")
            
            # Add DuckDuckGo results
            if duckduckgo_results:
                context_parts.append("\nFrom DuckDuckGo:")
                for result in duckduckgo_results:
                    context_parts.append(f"- {result}")
            
            context = "\n".join(context_parts)
            
            # Step 5: Create prompt for LLM
            prompt = f"""Based on the following context, please answer the user's travel question comprehensively and helpfully.
Context:
{context}
User Question: {english_question}
Please provide a detailed, helpful answer that combines information from the context. Focus on practical travel advice, recommendations, and useful tips. Keep the response informative but conversational."""
            
            # Step 6: Query LLM
            english_answer = self.query_llm(prompt)
            
            # Step 7: Translate answer back to user's language
            if lang_code != "en":
                final_answer = self.translate_text(english_answer, target_lang=lang_code)
            else:
                final_answer = english_answer
            
            # Step 8: Generate speech if enabled
            audio_file = None
            if enable_tts and final_answer:
                audio_file = self.generate_speech(final_answer, lang_code)
            
            return final_answer, audio_file
            
        except Exception as e:
            logger.error(f"Answer processing error: {e}")
            error_msg = "Sorry, I encountered an error while processing your question. Please try again."
            if lang_code != "en":
                error_msg = self.translate_text(error_msg, target_lang=lang_code)
            return error_msg, None

# Initialize bot
bot = TouristGuideBot()

def process_text_input(question, language, enable_tts):
    """Process text input from user."""
    if not question.strip():
        return "Please enter a question.", None
    
    answer, audio = bot.answer_question(question, language, enable_tts)
    return answer, audio

def process_audio_input(audio, language, enable_tts):
    """Process audio input from user."""
    if audio is None:
        return "Please record an audio message.", None
    
    # Convert speech to text
    question = bot.process_audio_input(audio)
    if not question or question.startswith("Could not understand"):
        return question, None
    
    # Process the transcribed question
    answer, audio_response = bot.answer_question(question, language, enable_tts)
    return f"Your question: {question}\n\nAnswer: {answer}", audio_response

# Custom CSS for enhanced styling
custom_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Global styles */
.gradio-container {
    font-family: 'Poppins', sans-serif !important;
    background: white !important;
    min-height: 100vh;
}

/* Main container styling */
.main-container {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
    margin: 20px !important;
    padding: 30px !important;
}

/* Header styling */
.header-title {
    text-align: center !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    margin-bottom: 20px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
}

.subtitle {
    text-align: center !important;
    color: #555 !important;
    font-size: 1.2rem !important;
    margin-bottom: 30px !important;
    line-height: 1.6 !important;
}

/* Features Section - Modern Card Design */
.features-container {
    background: white !important;
    border-radius: 16px !important;
    padding: 25px !important;
    margin: 20px 0 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    border: 1px solid #f0f0f0 !important;
}

.features-title {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    margin-bottom: 20px !important;
    color: #333 !important;
    text-align: center !important;
}

.features-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
    gap: 15px !important;
}

.feature-card {
    background: #f9f9f9 !important;
    border-radius: 12px !important;
    padding: 20px 15px !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
    border: 1px solid #eee !important;
}

.feature-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1) !important;
    background: #f5f5f5 !important;
}

.feature-icon {
    font-size: 2rem !important;
    margin-bottom: 10px !important;
}

.feature-text {
    font-weight: 500 !important;
    color: #555 !important;
    font-size: 0.95rem !important;
}

/* Option 1: Light blue accent */
.feature-card {
    background: #f8fafc !important;
    border: 1px solid #e0e7ff !important;
}


/* Feature list styling */
.features-container {
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 20px 0 !important;
    color: white !important;
    text-align: center !important;
}

.features-title {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    margin-bottom: 15px !important;
    color: white !important;
}

.feature-item {
    display: inline-block !important;
    margin: 5px 15px !important;
    padding: 8px 16px !important;
    background: rgba(255, 255, 255, 0.2) !important;
    border-radius: 25px !important;
    backdrop-filter: blur(5px) !important;
    font-weight: 500 !important;
}

/* Modern Control Panel - Fixed Checkbox Version */
.control-panel {
    background: white !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin: 20px 0 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    gap: 20px !important;
}

.control-section {
    padding: 20px !important;
    border-radius: 12px !important;
}

.language-section {
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%) !important;
}

.voice-section {
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%) !important;
    display: flex !important;
    flex-direction: column !important;
}

.section-header {
    color: white !important;
    margin-bottom: 15px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* Fixed Checkbox Styling */
.dark-checkbox {
    --size: 18px;
    margin: 0 !important;
    align-items: center !important;
}

.dark-checkbox .wrap {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    color: white !important;
}

.dark-checkbox input[type="checkbox"] {
    width: var(--size) !important;
    height: var(--size) !important;
    min-width: var(--size) !important;
    min-height: var(--size) !important;
}

.dark-checkbox label {
    color: white !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Question Prompt Container */
.question-prompt-container {
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%);
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 20px;
    text-align: center;
}

.question-prompt {
    color: white;
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
}

/* Custom Tabs */
.custom-tabs {
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}

.custom-tabs .tab-nav {
    margin-bottom: 20px;
}

.custom-tabs .tab-nav button {
    color: white !important;
    background: rgba(255,255,255,0.1) !important;
    border: none !important;
    border-radius: 8px !important;
    margin-right: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}

.custom-tabs .tab-nav button.selected {
    background: rgba(255,255,255,0.2) !important;
    font-weight: 600 !important;
}

/* Question Input Group */
.question-group {
    background: #1a202c;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
}

.question-label {
    color: white !important;
    font-size: 1.1rem !important;
    margin-bottom: 10px !important;
}

.question-textbox {
    background: #2d3748 !important;
    color: white !important;
    border: 1px solid #4a5568 !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

.question-textbox::placeholder {
    color: #a0aec0 !important;
}

.question-audio {
    width: 100% !important;
    border-radius: 10px !important;
}

/* Modern Button */
.modern-btn {
    background: white !important;
    color: #2d3748 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}

.modern-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
}

/* Tab styling */
.tab-nav button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    margin-right: 5px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
}

/* Input styling */
.gr-textbox, .gr-dropdown {
    border: 2px solid #e1e8ed !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    background: rgba(255, 255, 255, 0.9) !important;
}

.gr-textbox:focus, .gr-dropdown:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    transform: translateY(-2px) !important;
}

/* Button styling */
.gr-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}


/* Output textbox styling */
.output-textbox {
    border: 2px solid #e1e8ed !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.95) !important;
    min-height: 200px !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding: 20px !important;
}

/* Audio component styling */
.gr-audio {
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

/* Output Section */
.output-section {
    background: white;
    border-radius: 16px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.output-header {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 20px;
}

.output-textbox {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: white !important;
    padding: 20px !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}

.gr-audio {
    border-radius: 12px !important;
    background: white !important;
    border: 2px solid #e2e8f0 !important;
    padding: 15px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    margin-top: 15px !important;
}

/* EXAMPLES SECTION STYLING */
.examples-container {
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%);
    border-radius: 16px;
    padding: 25px;
    margin: 20px 0;
}

.examples-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    color: white;
    justify-content: center;
}

.examples-header h3 {
    margin: 0;
    font-size: 1.3rem;
}

.examples-icon {
    font-size: 1.5rem;
}

.examples-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 12px;
}

.example-question {
    background: white;
    color: black;
    border-radius: 10px;
    padding: 12px 15px;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.2);
    text-align: center;
}

.example-question:hover {
    background: rgba(255,255,255,0.2);
    transform: translateY(-3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Modern Header Styles */
.header-container {
    text-align: center;
    padding: 30px 20px;
    margin-bottom: 20px;
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.08);
}

.header-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 15px;
    letter-spacing: -1px;

}


.title-accent {
    color: #000000;  /* Pure black */
    font-weight: 800;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
}
.title-accent {
    color: #2d3748;  /* Pure black */
    font-weight: 800;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
}

.title-accent {
    color: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%); /* Matches your features section */
}

.header-subtitle {
    font-size: 1.5rem;
    color: #7f8c8d;
    margin-bottom: 15px;
    font-weight: 500;
}

.highlight {
    background: linear-gradient(145deg, #2c3e50 0%, #1e272e 100%) !important;
    background-repeat: no-repeat;
    background-size: 100% 30%;
    background-position: 0 85%;
    padding: 0 4px;
}

.header-divider {
    width: 100px;
    height: 4px;
    background: linear-gradient(to right, #3498db, #2ecc71);
    margin: 0 auto 20px;
    border-radius: 2px;
}

.header-tagline {
    display: flex;
    justify-content: center;
    gap: 15px;
    flex-wrap: wrap;
    color: #2c3e50;
    font-weight: 500;
    font-size: 1.1rem;
}

.header-tagline span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

/* Emoji styling */
.emoji {
    font-size: 1.2em;
    vertical-align: middle;
}

/* Responsive design */
@media (max-width: 768px) {
    .header-title {
        font-size: 2rem !important;
    }
    
    .question-prompt {
        font-size: 1.4rem !important;
    }
    
    .feature-item {
        display: block !important;
        margin: 5px 0 !important;
    }
}

/* Animation keyframes */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradio-container > div {
    animation: fadeIn 0.6s ease-out !important;
}
"""

# Create enhanced Gradio interface
def create_interface():
    """Create the enhanced Gradio interface."""
    
    with gr.Blocks(title="🌍 Multilingual Tourist Guide Bot", css=custom_css, theme=gr.themes.Base()) as demo:
        
        # Header Section
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="header-container">
                    <div class="header-title">
                        ✈️ <span class="title-accent">TravelGenie</span>
                    </div>
                    <div class="header-subtitle">
                        Your <span class="highlight">AI-Powered</span> Travel Companion
                    </div>
                    <div class="header-divider"></div>
                    <div class="header-tagline">
                        Discover destinations • Get instant advice • Traveling tips
                    </div>
                </div>
                """)
        
        # Features Section
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="features-container">
                    <div class="features-title">✨ Key Features</div>
                    <div class="features-grid">
                        <div class="feature-card">
                            <div class="feature-icon">🗣️</div>
                            <div class="feature-text">Voice Input & Output</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">🌐</div>
                            <div class="feature-text">10+ Languages</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">🤖</div>
                            <div class="feature-text">AI Recommendations</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">⚡</div>
                            <div class="feature-text">Fast Responses</div>
                        </div>
                    </div>
                </div>
                """)
        
        # Control Panel
        with gr.Row(elem_classes="control-panel"):
            # Language Section
            with gr.Column(scale=1, min_width=300, elem_classes="control-section language-section"):
                gr.Markdown("### 🌐 LANGUAGE", elem_classes="section-header")
                language_dropdown = gr.Dropdown(
                    choices=list(bot.languages.keys()),
                    value="English",
                    label="Select your preferred language",
                    elem_classes="dark-dropdown"
                )
            
            # Voice Section (Fixed)
            with gr.Column(scale=1, min_width=300, elem_classes="control-section voice-section"):
                gr.Markdown("### 🔊 VOICE", elem_classes="section-header")
                enable_tts_checkbox = gr.Checkbox(
                    value=True,
                    label="Enable voice responses",
                    elem_classes="dark-checkbox",
                    interactive=True
                )

                    
        
        # Question Prompt Section
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="question-prompt-container">
                    <div class="question-prompt">
                        🎤 Ask your travel question in text or voice
                    </div>
                </div>
                """)
        
        # Input Tabs
        with gr.Tabs(elem_classes="custom-tabs"):
            with gr.TabItem("💬 Text Input", elem_classes="tab-item"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="question-group"):
                            gr.Markdown("✍️ Your Travel Question", elem_classes="question-label")
                            text_input = gr.Textbox(
                                placeholder="Ask me anything about destinations, travel tips, local customs...",
                                lines=4,
                                elem_classes="question-textbox"
                            )
                        text_submit_btn = gr.Button("Get Travel Advice", elem_classes="modern-btn")
            
            with gr.TabItem("🎤 Voice Input", elem_classes="tab-item", visible=enable_tts_checkbox.value) as voice_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="question-group"):
                            gr.Markdown("🎤 Record Your Travel Question", elem_classes="question-label")
                            audio_input = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                elem_classes="question-audio"
                            )
                        audio_submit_btn = gr.Button("Process Voice Question", elem_classes="modern-btn")

        # This makes voice tab visibility depend on the checkbox
        enable_tts_checkbox.change(
            lambda x: gr.update(visible=x),
            inputs=[enable_tts_checkbox],
            outputs=[voice_tab]
        )
        
        # Output Section
        with gr.Row(elem_classes="output-section"):
            with gr.Column():
                gr.HTML("""
                <div class="output-header">
                    📋 Your Travel Guide Response
                </div>
                """)
                
                # Single text output box
                text_output = gr.Textbox(
                    label="💡 Travel Advice & Information",
                    lines=12,
                    max_lines=20,
                    show_copy_button=True,
                    elem_classes="output-textbox"
                )
                
                # Audio output (conditionally visible)
                audio_output = gr.Audio(
                    label="🔊 Audio Response",
                    elem_classes="gr-audio",
                    visible=False
                )
        
        # Connect TTS checkbox to audio output visibility
        enable_tts_checkbox.change(
            lambda x: gr.update(visible=x),
            inputs=[enable_tts_checkbox],
            outputs=[audio_output]
        )

        # ========== EXAMPLE QUESTIONS SECTION ==========
        with gr.Row(elem_classes="examples-container"):
            with gr.Column():
                gr.HTML("""
                <div class="examples-header">
                    <span class="examples-icon">💡</span>
                    <h3>Try These Example Questions</h3>
                </div>
                <div class="examples-grid">
                    <div class="example-question">"What are the best places to visit in Paris?"</div>
                    <div class="example-question">"Give me travel tips for first-time international travelers"</div>
                    <div class="example-question">"What's the best time to visit Japan?"</div>
                    <div class="example-question">"How can I save money while traveling in Europe?"</div>
                    <div class="example-question">"Tell me about local customs in Dubai"</div>
                    <div class="example-question">"Best budget-friendly destinations in Southeast Asia"</div>
                    <div class="example-question">"What documents do I need for international travel?"</div>
                </div>
                """)
        
        
        
        # Event handlers for text input
        def handle_text_input(question, language, enable_tts):
            """Handle text input and show/hide appropriate outputs."""
            if not question.strip():
                return "Please enter a question.", None, gr.update(visible=True), gr.update(visible=False)
            
            answer, audio = process_text_input(question, language, enable_tts)
            return answer, audio, gr.update(visible=True), gr.update(visible=False)
        
        def handle_audio_input(audio, language, enable_tts):
            """Handle audio input and show/hide appropriate outputs."""
            if audio is None:
                return "Please record an audio message.", None, gr.update(visible=False), gr.update(visible=True)
            
            answer, audio_response = process_audio_input(audio, language, enable_tts)
            return answer, audio_response, gr.update(visible=False), gr.update(visible=True)
        
        # Connect event handlers
        text_submit_btn.click(
            fn=process_text_input,
            inputs=[text_input, language_dropdown, enable_tts_checkbox],
            outputs=[text_output, audio_output]
        )
        
        audio_submit_btn.click(
            fn=process_audio_input,
            inputs=[audio_input, language_dropdown, enable_tts_checkbox],
            outputs=[text_output, audio_output]
        )
        
        # Auto-submit on Enter for text input
        text_input.submit(
            fn=handle_text_input,
            inputs=[text_input, language_dropdown, enable_tts_checkbox],
            outputs=[text_output, audio_output]
        )
        
        # Example question click handlers
        def set_example_question(question):
            return question
        
        # Footer with additional information
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="text-align: center; margin-top: 40px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                    <h3 style="margin-bottom: 15px;">🔑 Setup Instructions</h3>
                    <p style="margin: 10px 0;"><strong>GROQ_API_KEY</strong> - For fast LLM responses via Groq API</p>
                    <p style="margin: 10px 0;"><strong>GEMINI_API_KEY</strong> - For Google Gemini API access</p>
                    <p style="margin: 10px 0; font-size: 0.9rem; opacity: 0.8;">Add these as environment variables for full functionality</p>
                </div>
                """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    
    print("🚀 Starting Enhanced Tourist Guide Bot...")
    print("📝 Note: Add your API keys as environment variables:")
    print("   - GROQ_API_KEY for Groq API")
    print("   - GEMINI_API_KEY for Google Gemini API")
    
    demo = create_interface()
    demo.launch(
        share=True,
        show_error=True,

    )
