import os
import time
import logging
import streamlit as st
from gtts import gTTS
import tempfile
import pygame  # For playing audio
import threading  # To handle audio playback in a separate thread
import speech_recognition as sr  # New import for voice recognition
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from googletrans import Translator
from deep_translator import GoogleTranslator
from faq import COMPREHENSIVE_FAQS, FAQ_QUESTIONS

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

# Global variable to track audio playback
is_playing = False

# Initialize translator
translator = Translator()

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def load_vector_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            collection_name=VECTOR_STORE_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        logging.info("Loaded existing Chroma vector database.")
    else:
        pdf_docs = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
        if not pdf_docs:
            raise FileNotFoundError(f"No PDF files found in {PDF_DIRECTORY}.")

        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)

        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist()
        logging.info("Chroma vector database created and persisted.")

    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOllama(
        model=MODEL_NAME, 
        max_tokens=200, 
        temperature=0.5
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )
    logging.info("Conversation chain created successfully.")
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

# Text-to-Speech Function with Comprehensive Error Handling
def speak_text(text, selected_language):
    """
    Convert text to speech using gTTS and play audio with improved error handling
    
    Args:
        text (str): Text to be converted to speech
        selected_language (str): Selected language name
    """
    global is_playing
    
    # Prevent multiple simultaneous playbacks
    if is_playing:
        st.warning("Audio is already playing. Please wait.")
        return
    
    def play_audio_thread():
        global is_playing
        is_playing = True
        
        try:
            # Determine language code
            language_code = LANGUAGES.get(selected_language, 'en')
            
            # Fallback to English if language not supported by gTTS
            if language_code not in TTS_SUPPORTED_LANGUAGES:
                language_code = 'en'
                st.warning(f"Language {selected_language} not supported. Falling back to English.")
            
            # Use a unique temporary filename with a specific extension
            temp_dir = tempfile.gettempdir()
            temp_filename = os.path.join(temp_dir, f"tts_audio_{time.time()}.mp3")
            
            # Generate speech
            tts = gTTS(text=text, lang=language_code)
            tts.save(temp_filename)
            
            # Play the audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            
            # Safely remove the temporary file
            try:
                os.unlink(temp_filename)
            except Exception as cleanup_error:
                logging.warning(f"Could not delete temporary audio file: {cleanup_error}")
        
        except Exception as e:
            st.error(f"Text-to-Speech error: {e}")
            logging.error(f"TTS Error: {e}")
        
        finally:
            is_playing = False
    
    # Start audio playback in a separate thread
    threading.Thread(target=play_audio_thread, daemon=True).start()

# New function for voice recognition
def recognize_speech(language):
    """
    Recognize speech input based on selected language
    
    Args:
        language (str): Selected language name
    
    Returns:
        str: Recognized speech text or None if recognition fails
    """
    recognizer = sr.Recognizer()
    
    try:
        # Get the language code for speech recognition
        lang_code = SPEECH_RECOGNITION_LANGUAGES.get(language, 'en-US')
        
        # Use microphone as source
        with sr.Microphone() as source:
            st.info(f"Listening... (Speak in {language})")
            
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Listen for speech with timeout
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        
        # Recognize speech
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
    """Process user questions and retrieve answers."""
    if not user_question.strip():
        st.error("Please enter a valid question.")
        return

    try:
        selected_language = st.session_state.selected_language
        target_lang_code = LANGUAGES.get(selected_language, 'en')
        
        # Check if it's a predefined FAQ
        if is_faq and user_question in FAQ_QUESTIONS:
            response_text = FAQ_QUESTIONS[user_question]
        else:
            # Ensure conversation is initialized
            if st.session_state.conversation is None:
                st.error("The documents need to be processed first.")
                return
            
            # Translate question to English
            if target_lang_code != 'en':
                try:
                    translated_question = translator.translate(user_question, dest='en')
                    translated_question = translated_question.text if translated_question else user_question
                except Exception as e:
                    logging.error(f"Translation error: {e}")
                    translated_question = user_question  # Fallback to original text
            else:
                translated_question = user_question
            
            # Get response from LangChain
            response = st.session_state.conversation({'question': translated_question})
            if not response or 'answer' not in response:
                st.error("No valid response could be retrieved.")
                logging.error("LangChain response is None or missing 'answer'.")
                return
            
            response_text = response['answer']
        
        # Translate response back to user's selected language
        translated_response = translate_text(response_text, target_lang_code)
        
        # Append to chat history
        st.session_state.chat_history.append({"user": user_question, "bot": translated_response})
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")
        logging.error(f"Error in handle_userinput: {e}")
        return
    
# Streamlit app configuration
st.set_page_config(page_title="Nyaysetu", page_icon="https://i.ibb.co/1XLGTs1/Whats-App-Image-2024-11-21-at-20-26-32-5dec28a6.jpg")
st.write(css, unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

# Main app logic
def main():
    st.header("NYAYASETU")
    
    # Language selection and FAQ buttons in sidebar
    with st.sidebar:
        st.subheader("Language Selection")
        st.session_state.selected_language = st.selectbox(
            "Choose your preferred language:", 
            list(LANGUAGES.keys()), 
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )

        #if st.button("Process Documents"):
        try:
            vectorstore = load_vector_db() # Load or create the Chroma vector store
            
            st.session_state.conversation = get_conversation_chain(vectorstore) # Create a conversation chain
            print("Documents have been processed successfully!")
        except Exception as e:
            st.error(f"An error occurred while processing the PDFs: {e}")
            logging.error(f"Error in processing PDFs: {e}")

        # Voice Input Button
        if st.button("🎤 Voice Input"):
            # Call voice recognition function
            voice_input = recognize_speech(st.session_state.selected_language)
            
            if voice_input:
                # Process the voice input
                handle_userinput(voice_input)

         # Iterate through FAQ topics and create dropdowns
        for topic, questions in COMPREHENSIVE_FAQS.items():
            selected_question = st.selectbox(
                f"{topic}:",
                ["Select a Question"] + list(questions.keys()),
                key=f"faq_dropdown_{topic}"
            )
            
            if selected_question != "Select a Question":
                # Display the answer for the selected question
                st.session_state.chat_history.append({"user": selected_question, "bot": questions[selected_question]})

    
    if st.session_state.chat_history:
        for idx, message in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 messages
                # User message
                st.write(user_template.replace("{{MSG}}", message["user"]), unsafe_allow_html=True)
                
                # Bot message with TTS button
                bot_msg_div = bot_template.replace("{{MSG}}", message["bot"])
                st.write(bot_msg_div, unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Ask a question:")

    if user_question:
        handle_userinput(user_question)

    # Display chat history
    if st.session_state.chat_history and user_question:
        for idx, message in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 messages
            # User message
            st.write(user_template.replace("{{MSG}}", message["user"]), unsafe_allow_html=True)
            
            # Bot message with TTS button
            bot_msg_div = bot_template.replace("{{MSG}}", message["bot"])
            st.write(bot_msg_div, unsafe_allow_html=True)
            
            
            tts_button = st.button(f"🔊 Listen", key=f"tts_1_{idx}")
            if tts_button:
                speak_text(message["bot"], st.session_state.selected_language)


if __name__ == '__main__':
    main()