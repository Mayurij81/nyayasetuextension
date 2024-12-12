import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env",override=True)

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="NYAYSETU"
from langchain_community.embeddings import OllamaEmbeddings  
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

llm=Ollama(model="llama3.2")
output_parser=StrOutputParser()

import webbrowser
import time
import logging
import streamlit as st
from gtts import gTTS
import tempfile
import pygame  
import threading  
import speech_recognition as sr  
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from googletrans import Translator
from deep_translator import GoogleTranslator
from faq import FAQ_QUESTIONS
from datetime import datetime

from htmlTemplates import css, bot_template, user_template

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
PDF_DIRECTORY = 'pdfs'
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

logging.basicConfig(level=logging.INFO)

# Language mapping
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi', 
    'Marathi': 'mr', 
    'Kannada': 'kn', 
    'Punjabi': 'pa', 
    'Tamil': 'ta', 
    'Telugu': 'te', 
    'Arabic': 'ar', 
    'Urdu': 'ur', 
    'Japanese': 'ja', 
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh-CN',
    'Portuguese': 'pt'
}

# Supported TTS languages
TTS_SUPPORTED_LANGUAGES = ['en', 'hi', 'es', 'fr', 'de', 'zh-CN', 'pt', 'ja']

SPEECH_RECOGNITION_LANGUAGES = {
    'English': 'en-US',
    'Hindi': 'hi-IN',
    'Spanish': 'es-ES',
    'French': 'fr-FR',
    'German': 'de-DE',
    'Chinese': 'zh-CN',
    'Portuguese': 'pt-BR',
    'Japanese': 'ja-JP'
}

is_playing = False

translator = Translator()

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                page_text = page_text.replace('\n\n', ' ').replace('\n', ' ').strip()
                text += page_text + "\n\n" 
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf}: {e}")

    logging.info(f"Total extracted text length: {len(text)} characters")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,  
        chunk_overlap=300,  
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Log chunks for debugging
    logging.info(f"Created {len(chunks)} text chunks")
    for i, chunk in enumerate(chunks[:5], 1):
        logging.info(f"Chunk {i} preview (first 200 chars): {chunk[:200]}...")
    
    return chunks

def load_vector_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        try:
            vectorstore = Chroma(
                collection_name=VECTOR_STORE_NAME,
                embedding_function=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            
            document_count = vectorstore._collection.count()
            logging.info(f"Loaded existing Chroma vector database with {document_count} documents.")
            
            return vectorstore
        except Exception as e:
            logging.error(f"Error loading existing vector database: {e}")
            
    pdf_docs = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    
    if not pdf_docs:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIRECTORY}.")

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)

    try:
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()
        
        document_count = vectorstore._collection.count()
        logging.info(f"Created and persisted Chroma vector database with {document_count} documents.")
        
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating vector database: {e}")
        raise

def debug_vector_store(vectorstore, query, top_k=5):
    results = vectorstore.similarity_search(query, k=top_k)
    logging.info(f"Retrieved {len(results)} documents for query: {query}")
    for i, doc in enumerate(results, 1):
        logging.info(f"Document {i} preview:\n{doc.page_content[:500]}...")

def get_conversation_chain(vectorstore):
    llm = ChatOllama(
        model=MODEL_NAME, 
        max_tokens=300,
        temperature=0.5
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        max_token_limit=1000
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 5
            }
        ),
        memory=memory,
        verbose=True
    )
    logging.info("Conversation chain created with enhanced context memory.")
    return conversation_chain

def translate_text(text, target_lang):
    try:
        if target_lang == 'en':
            return text
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def speak_text(text, selected_language):
    global is_playing
    
    if is_playing:
        st.warning("Audio is already playing. Please wait.")
        return
    
    def play_audio_thread():
        global is_playing
        is_playing = True
        
        try:
            language_code = LANGUAGES.get(selected_language, 'en')
            
            if language_code not in TTS_SUPPORTED_LANGUAGES:
                language_code = 'en'
                st.warning(f"Language {selected_language} not supported. Falling back to English.")
            
            temp_dir = tempfile.gettempdir()
            temp_filename = os.path.join(temp_dir, f"tts_audio_{time.time()}.mp3")
            
            tts = gTTS(text=text, lang=language_code)
            tts.save(temp_filename)
            
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            
            try:
                os.unlink(temp_filename)
            except Exception as cleanup_error:
                logging.warning(f"Could not delete temporary audio file: {cleanup_error}")
        
        except Exception as e:
            st.error(f"Text-to-Speech error: {e}")
            logging.error(f"TTS Error: {e}")
        
        finally:
            is_playing = False
    
    threading.Thread(target=play_audio_thread, daemon=True).start()

def recognize_speech(language):
    
    recognizer = sr.Recognizer()
    
    try:
        lang_code = SPEECH_RECOGNITION_LANGUAGES.get(language, 'en-US')
        
        with sr.Microphone() as source:
            st.info(f"Listening... (Speak in {language})")
            
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
        
        try:
            text = recognizer.recognize_google(audio, language=lang_code)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    
    except Exception as e:
        st.error(f"An error occurred during speech recognition: {e}")
    
    return None

def handle_userinput(user_question, is_faq=False):
    if not user_question.strip():
        st.error("Please enter a valid question.")
        return

    try:
        selected_language = st.session_state.selected_language
        target_lang_code = LANGUAGES.get(selected_language, 'en')
        
        with st.spinner("Processing your question..."):
            if is_faq and user_question in FAQ_QUESTIONS:
                response_text = FAQ_QUESTIONS[user_question]
            else:
                if st.session_state.conversation is None:
                    st.error("The documents need to be processed first.")
                    return
                
                if target_lang_code != 'en':
                    try:
                        translated_question = translator.translate(user_question, dest='en')
                        translated_question = translated_question.text if translated_question else user_question
                    except Exception as e:
                        logging.error(f"Translation error: {e}")
                        translated_question = user_question 
                else:
                    translated_question = user_question
                
                response = st.session_state.conversation({
                    'question': translated_question,
                    'chat_history': [
                        (msg['user'], msg['bot']) for msg in st.session_state.chat_history[-3:]
                    ]
                })
                if not response or 'answer' not in response:
                    st.error("No valid response could be retrieved.")
                    logging.error("LangChain response is None or missing 'answer'.")
                    return
                
                response_text = response['answer']
            
            translated_response = translate_text(response_text, target_lang_code)
            
            st.session_state.chat_history.append({"user": user_question, "bot": translated_response})

            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")
        logging.error(f"Error in handle_userinput: {e}")
        return

    
st.set_page_config(page_title="Nyaysetu", page_icon="https://i.ibb.co/1XLGTs1/Whats-App-Image-2024-11-21-at-20-26-32-5dec28a6.jpg")
st.write(css, unsafe_allow_html=True)

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

current_hour=datetime.now().hour
def get_greeting():
    if current_hour < 12:
        return "Good Morning"
    elif current_hour >=12 and current_hour<18:
        return "Good Afternon"
    else:
        return "Good Evening"

# Main app logic
def main():

    

    st.markdown(
    """
    <style>
    /* Add gradient background to the main app */
    .stApp {
        background: linear-gradient(to right, #f2d49d, #FBAC1B);
        color: #000000; /* Ensure text is visible */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.header("NYAYASETU")
    st.write(get_greeting())
    st.write("I am your virtual-assistant, how can I help you?")
    
    # Language selection and FAQ buttons in sidebar
    with st.sidebar:
        st.subheader("Language Selection")
        st.session_state.selected_language = st.selectbox(
            "Choose your preferred language:", 
            list(LANGUAGES.keys()), 
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )

        try:
            vectorstore = load_vector_db() # Load or create the Chroma vector store
            
            st.session_state.conversation = get_conversation_chain(vectorstore) # Create a conversation chain
            print("Documents have been processed successfully!")
        except Exception as e:
            st.error(f"An error occurred while processing the PDFs: {e}")
            logging.error(f"Error in processing PDFs: {e}")


        # FAQ Buttons
        st.subheader("Frequently Asked Questions")
        faq_questions = list(FAQ_QUESTIONS.keys())
        
        # Create two columns for FAQ buttons
        col1= st.columns(1)[0]
        
        with col1:
            for i in range(0, len(faq_questions), 1):
                if st.button(faq_questions[i]):
                    handle_userinput(faq_questions[i], is_faq=True)
        
        st.divider()  
        st.subheader("Quick Access")
        default_url = "https://services.ecourts.gov.in/ecourtindia_v6/"
        popup_url = default_url

        if st.button("know your case status"):
            try:
                webbrowser.open(popup_url, new=2)
                st.success(f"Opened {popup_url} in a new window")
            except Exception as e:
                st.error(f"Error opening URL: {e}")

        
        default_url = "https://doj.gov.in/live-streaming-of-court-cases/"
        pop_url = default_url

        if st.button("live stream of cases"):
            try:
                webbrowser.open(pop_url, new=2)
                st.success(f"Opened {pop_url} in a new window")
            except Exception as e:
                st.error(f"Error opening URL: {e}")

        default_url = "https://filing.ecourts.gov.in/pdedev/"
        pop_url = default_url

        if st.button("e-filing "):
            try:
                webbrowser.open(pop_url, new=2)
                st.success(f"Opened {pop_url} in a new window")
            except Exception as e:
                st.error(f"Error opening URL: {e}")

        default_url = "https://pay.ecourts.gov.in/epay/"
        pop_url = default_url

        if st.button("e-payment"):
            try:
                webbrowser.open(pop_url, new=2)
                st.success(f"Opened {pop_url} in a new window")
            except Exception as e:
                st.error(f"Error opening URL: {e}")


        default_url = "https://vcourts.gov.in/virtualcourt/"
        pop_url = default_url

        if st.button("settle your traffic violation"):
            try:
                webbrowser.open(pop_url, new=2)
                st.success(f"Opened {pop_url} in a new window")
            except Exception as e:
                st.error(f"Error opening URL: {e}")


    if 'reply_context' not in st.session_state:
        st.session_state.reply_context = None

    if st.session_state.reply_context:
        st.info(f"Replying to: {st.session_state.reply_context}")
  
    col1, col2 = st.columns([12,1])

    with col1:
        user_question = st.chat_input(
            placeholder="Ask a question:",
            key="primary_chat_input"
        )
    with col2:
        if st.button("🎙️"):
            voice_input = recognize_speech(st.session_state.selected_language)
            
            if voice_input:
                handle_userinput(voice_input)

    container = st.container()

    if user_question and st.session_state.reply_context:
        full_question = f"Regarding previous context - {st.session_state.reply_context}: {user_question}"
        st.session_state.reply_context = None
    else:
        full_question = user_question

    if full_question:
        handle_userinput(full_question)

    if st.session_state.chat_history:
        for idx, message in enumerate(reversed(st.session_state.chat_history[-5:])):
            st.write(user_template.replace("{{MSG}}", message["user"]), unsafe_allow_html=True)
            
            bot_msg_div = bot_template.replace("{{MSG}}", message["bot"])
            st.write(bot_msg_div, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 12])
            
            with col1:
                tts_button = st.button(f"🔊", key=f"tts_{idx}")
                if tts_button:
                    speak_text(message["bot"], st.session_state.selected_language)
            
            with col2:
                reply_button = st.button(f"↩️", key=f"reply_{idx}")
                if reply_button:
                    st.session_state.reply_context = message["user"]
                    st.rerun()


if __name__ == '__main__':
    main()