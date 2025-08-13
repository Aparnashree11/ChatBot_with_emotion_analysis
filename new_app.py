"""
Streamlit Chat App with MySQL Feedback Storage
Complete chat sessions are saved to MySQL when ended

Requirements:
pip install streamlit mysql-connector-python

Run with: streamlit run streamlit_mysql_chat.py
"""

import streamlit as st
import mysql.connector
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid
from dataclasses import dataclass, asdict

# Import your chatbot
from model_testing import LocalRAGChatbot
from emotion_classification_training import EmotionPredictor

import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChatMessage:
    """Individual chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    confidence_score: Optional[float] = None
    sources_used: Optional[int] = None
    response_time: Optional[float] = None
    detected_emotion: Optional[str] = None  # Added for emotion tracking
    emotion_score: Optional[float] = None   # Added for emotion confidence

@dataclass
class ChatSession:
    """Complete chat session with feedback"""
    session_id: str
    start_time: str
    end_time: Optional[str]
    messages: List[ChatMessage]
    overall_rating: Optional[int]  # 1-5 stars
    feedback_comment: Optional[str]
    user_email: Optional[str]
    total_interactions: int

class MySQLChatStorage:
    """MySQL storage for chat sessions and feedback"""
    
    def __init__(self):
        self.config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'database': os.getenv('MYSQL_DB', 'chatbot_feedback'),
            'port': 3306,
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
        self._init_database()
    
    def _get_connection(self):
        """Get database connection"""
        return mysql.connector.connect(**self.config)
    
    def _init_database(self):
        """Initialize MySQL tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create chat_sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(50) UNIQUE NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                total_interactions INT DEFAULT 0,
                overall_rating INT,
                feedback_comment TEXT,
                user_email VARCHAR(255),
                messages_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_session_id (session_id),
                INDEX idx_start_time (start_time),
                INDEX idx_rating (overall_rating)
            )
            ''')
            
            # Create individual_messages table for detailed analytics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS individual_messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(50) NOT NULL,
                message_order INT NOT NULL,
                role ENUM('user', 'assistant') NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                confidence_score FLOAT,
                sources_used INT,
                response_time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_session_id (session_id),
                INDEX idx_timestamp (timestamp),
                INDEX idx_confidence (confidence_score),
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
            ''')
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("✅ MySQL tables initialized successfully")
            
        except mysql.connector.Error as e:
            print(f"❌ MySQL initialization error: {e}")
            raise
    
    def save_chat_session(self, session: ChatSession):
        """Save complete chat session to MySQL"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare messages for JSON storage
            messages_data = []
            for msg in session.messages:
                msg_dict = asdict(msg)
                messages_data.append(msg_dict)
            
            # Insert or update chat session
            cursor.execute('''
            INSERT INTO chat_sessions (
                session_id, start_time, end_time, total_interactions,
                overall_rating, feedback_comment, user_email, messages_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                end_time = VALUES(end_time),
                total_interactions = VALUES(total_interactions),
                overall_rating = VALUES(overall_rating),
                feedback_comment = VALUES(feedback_comment),
                user_email = VALUES(user_email),
                messages_json = VALUES(messages_json)
            ''', (
                session.session_id,
                session.start_time,
                session.end_time,
                session.total_interactions,
                session.overall_rating,
                session.feedback_comment,
                session.user_email,
                json.dumps(messages_data)
            ))
            
            # Save individual messages for analytics with emotion data
            cursor.execute('DELETE FROM individual_messages WHERE session_id = %s', (session.session_id,))
            
            for i, message in enumerate(session.messages):
                cursor.execute('''
                INSERT INTO individual_messages (
                    session_id, message_order, role, content, timestamp,
                    confidence_score, sources_used, response_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    session.session_id,
                    i + 1,
                    message.role,
                    message.content,
                    message.timestamp,
                    message.confidence_score,
                    message.sources_used,
                    message.response_time
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except mysql.connector.Error as e:
            print(f"❌ Error saving chat session: {e}")
            return False

class EmotionAwareChatbot:
    """Emotion-aware chatbot that adapts responses based on user emotions"""
    
    def __init__(self, rag_chatbot: LocalRAGChatbot, emotion_predictor: EmotionPredictor):
        self.rag_chatbot = rag_chatbot
        self.emotion_predictor = emotion_predictor
        
        # Emotion categories
        self.positive_emotions = {'joy', 'love', 'excitement', 'gratitude', 'admiration', 'amusement', 'optimism', 'pride', 'relief'}
        self.negative_emotions = {'anger', 'sadness', 'fear', 'disgust', 'disappointment', 'annoyance', 'embarrassment', 'grief', 'remorse'}
        self.extreme_negative = {'anger', 'disgust', 'grief', 'remorse'}  # Escalation triggers
        self.neutral_emotions = {'neutral', 'realization', 'approval', 'disapproval'}
        self.complex_emotions = {'confusion', 'curiosity', 'desire', 'nervousness', 'surprise', 'caring'}
    
    def detect_emotion(self, user_message: str) -> Tuple[str, float, str]:
        """Detect emotion in user message"""
        try:
            top_emotions = self.emotion_predictor.predict_top_k(user_message, k=1)
            primary_emotion = top_emotions[0][0] if top_emotions else 'neutral'
            emotion_score = top_emotions[0][1] if top_emotions else 0.0
            
            # Categorize emotion
            if primary_emotion in self.positive_emotions:
                emotion_category = 'positive'
            elif primary_emotion in self.negative_emotions:
                emotion_category = 'negative'
            elif primary_emotion in self.complex_emotions:
                emotion_category = 'complex'
            else:
                emotion_category = 'neutral'
            
            return primary_emotion, emotion_score, emotion_category
        except Exception as e:
            return 'neutral', 0.0, 'neutral'
    
    def should_escalate_to_support(self, emotion: str, emotion_score: float) -> bool:
        """Determine if chat should be escalated to human support"""
        # Escalate for extreme negative emotions with high confidence
        if emotion in self.extreme_negative and emotion_score > 0.7:
            return True
        
        # Escalate for very strong negative emotions
        if emotion in self.negative_emotions and emotion_score > 0.9:
            return True
        
        return False
    
    def adapt_response_tone(self, base_response: str, emotion: str, emotion_category: str, emotion_score: float) -> str:
        """Adapt response tone based on detected emotion"""
        
        # Check for escalation first
        if self.should_escalate_to_support(emotion, emotion_score):
            escalation_msg = "\n\n🔔 **I've noticed you might be experiencing some frustration. Let me connect you with our human support team who can provide more personalized assistance.**\n\n📞 **Contact Support:**\n- Live Chat: Available 24/7\n- Email: support@company.com\n- Phone: 1-800-SUPPORT"
            return base_response + escalation_msg
        
        # Adapt tone based on emotion category
        if emotion_category == 'positive':
            if emotion in ['joy', 'excitement', 'love']:
                return f"😊 I'm so glad to help! {base_response}\n\nIs there anything else I can assist you with today?"
            elif emotion in ['gratitude']:
                return f"You're very welcome! ✨ {base_response}\n\nI'm happy I could help you!"
            elif emotion in ['admiration', 'pride']:
                return f"That's wonderful! 🌟 {base_response}\n\nFeel free to ask if you need any more information!"
            else:
                return f"😊 {base_response}\n\nI'm here if you need any other assistance!"
        
        elif emotion_category == 'negative':
            if emotion in ['anger', 'annoyance']:
                return f"I understand your frustration, and I'm here to help resolve this. 🤝 {base_response}\n\nIf you need immediate assistance, please don't hesitate to contact our support team."
            elif emotion in ['sadness', 'disappointment']:
                return f"I'm sorry to hear you're having difficulties. 💙 Let me do my best to help: {base_response}\n\nIs there anything else I can clarify for you?"
            elif emotion in ['fear', 'nervousness']:
                return f"I understand your concerns, and I'm here to help put your mind at ease. 🤗 {base_response}\n\nPlease let me know if you have any other questions!"
            else:
                return f"I apologize for any inconvenience. 🙏 {base_response}\n\nI'm committed to helping you resolve this."
        
        elif emotion_category == 'complex':
            if emotion in ['confusion']:
                return f"I can see this might be confusing - let me break it down clearly: 💡 {base_response}\n\nDoes this help clarify things? Feel free to ask for more details!"
            elif emotion in ['curiosity']:
                return f"Great question! I love your curiosity! 🔍 {base_response}\n\nIs there anything specific about this you'd like me to explain further?"
            elif emotion in ['surprise']:
                return f"I hope this information is helpful! ✨ {base_response}\n\nLet me know if you'd like to learn more about anything!"
            else:
                return f"📝 {base_response}\n\nI hope this addresses your question!"
        
        else:  # neutral
            return f"{base_response}\n\nIs there anything else I can help you with?"
    
class EmotionAwareChatbot:
    """Emotion-aware chatbot that adapts responses based on user emotions"""
    
    def __init__(self, rag_chatbot: LocalRAGChatbot, emotion_predictor: EmotionPredictor):
        self.rag_chatbot = rag_chatbot
        self.emotion_predictor = emotion_predictor
        
        # Emotion categories
        self.positive_emotions = {'joy', 'love', 'excitement', 'gratitude', 'admiration', 'amusement', 'optimism', 'pride', 'relief'}
        self.negative_emotions = {'anger', 'sadness', 'fear', 'disgust', 'disappointment', 'annoyance', 'embarrassment', 'grief', 'remorse'}
        self.extreme_negative = {'anger', 'disgust', 'grief', 'remorse'}  # Escalation triggers
        self.neutral_emotions = {'neutral', 'realization', 'approval', 'disapproval'}
        self.complex_emotions = {'confusion', 'curiosity', 'desire', 'nervousness', 'surprise', 'caring'}
    
    def detect_emotion(self, user_message: str) -> tuple:
        """Detect emotion in user message"""
        try:
            top_emotions = self.emotion_predictor.predict_top_k(user_message, k=1)
            primary_emotion = top_emotions[0][0] if top_emotions else 'neutral'
            emotion_score = top_emotions[0][1] if top_emotions else 0.0
            
            # Categorize emotion
            if primary_emotion in self.positive_emotions:
                emotion_category = 'positive'
            elif primary_emotion in self.negative_emotions:
                emotion_category = 'negative'
            elif primary_emotion in self.complex_emotions:
                emotion_category = 'complex'
            else:
                emotion_category = 'neutral'
            
            return primary_emotion, emotion_score, emotion_category
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return 'neutral', 0.0, 'neutral'
    
    def should_escalate_to_support(self, emotion: str, emotion_score: float) -> bool:
        """Determine if chat should be escalated to human support"""
        # Escalate for extreme negative emotions with high confidence
        if emotion in self.extreme_negative and emotion_score > 0.7:
            return True
        
        # Escalate for very strong negative emotions
        if emotion in self.negative_emotions and emotion_score > 0.9:
            return True
        
        return False
    
    def adapt_response_tone(self, base_response: str, emotion: str, emotion_category: str, emotion_score: float) -> str:
        """Adapt response tone based on detected emotion"""
        
        # Check for escalation first
        if self.should_escalate_to_support(emotion, emotion_score):
            escalation_msg = "\n\n🔔 **I've noticed you might be experiencing some frustration. Let me connect you with our human support team who can provide more personalized assistance.**\n\n📞 **Contact Support:**\n- Live Chat: Available 24/7\n- Email: support@company.com\n- Phone: 1-800-SUPPORT"
            return base_response + escalation_msg
        
        # Adapt tone based on emotion category
        if emotion_category == 'positive':
            if emotion in ['joy', 'excitement', 'love']:
                return f"😊 I'm so glad to help! {base_response}\n\nIs there anything else I can assist you with today?"
            elif emotion in ['gratitude']:
                return f"You're very welcome! ✨ {base_response}\n\nI'm happy I could help you!"
            elif emotion in ['admiration', 'pride']:
                return f"That's wonderful! 🌟 {base_response}\n\nFeel free to ask if you need any more information!"
            else:
                return f"😊 {base_response}\n\nI'm here if you need any other assistance!"
        
        elif emotion_category == 'negative':
            if emotion in ['anger', 'annoyance']:
                return f"I understand your frustration, and I'm here to help resolve this. 🤝 {base_response}\n\nIf you need immediate assistance, please don't hesitate to contact our support team."
            elif emotion in ['sadness', 'disappointment']:
                return f"I'm sorry to hear you're having difficulties. 💙 Let me do my best to help: {base_response}\n\nIs there anything else I can clarify for you?"
            elif emotion in ['fear', 'nervousness']:
                return f"I understand your concerns, and I'm here to help put your mind at ease. 🤗 {base_response}\n\nPlease let me know if you have any other questions!"
            else:
                return f"I apologize for any inconvenience. 🙏 {base_response}\n\nI'm committed to helping you resolve this."
        
        elif emotion_category == 'complex':
            if emotion in ['confusion']:
                return f"I can see this might be confusing - let me break it down clearly: 💡 {base_response}\n\nDoes this help clarify things? Feel free to ask for more details!"
            elif emotion in ['curiosity']:
                return f"Great question! I love your curiosity! 🔍 {base_response}\n\nIs there anything specific about this you'd like me to explain further?"
            elif emotion in ['surprise']:
                return f"I hope this information is helpful! ✨ {base_response}\n\nLet me know if you'd like to learn more about anything!"
            else:
                return f"📝 {base_response}\n\nI hope this addresses your question!"
        
        else:  # neutral
            return f"{base_response}\n\nIs there anything else I can help you with?"
    
    def ask_with_emotion_awareness(self, user_message: str) -> Dict:
        """Get response with emotion awareness - THIS IS THE MISSING METHOD"""
        # Detect emotion in user message
        emotion, emotion_score, emotion_category = self.detect_emotion(user_message)
        
        # Get base response from RAG chatbot
        base_response = self.rag_chatbot.ask(user_message)
        
        # Adapt response based on emotion
        adapted_answer = self.adapt_response_tone(
            base_response.answer, 
            emotion, 
            emotion_category, 
            emotion_score
        )
        
        # Check for escalation
        needs_escalation = self.should_escalate_to_support(emotion, emotion_score)
        
        return {
            'answer': adapted_answer,
            'confidence_score': base_response.confidence_score,
            'sources_used': base_response.sources_used,
            'response_time': base_response.response_time_seconds,
            'detected_emotion': emotion,
            'emotion_score': emotion_score,
            'emotion_category': emotion_category,
            'needs_escalation': needs_escalation,
            'status': base_response.status
        }

class StreamlitChatApp:
    """Streamlit chat application"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            self.load_chatbot()
        
        # Initialize MySQL storage with default configuration
        if 'mysql_storage' not in st.session_state:
            self.setup_mysql()
    
    def handle_small_talk(self, question: str) -> Optional[str]:
        """Handle common greetings and small talk without using the LLM"""
        question_lower = question.lower().strip()
        
        # Greetings
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(pattern in question_lower for pattern in greeting_patterns):
            return "Hello! 👋 I'm here to help you with questions about our products and services. What can I help you with today?"
        
        # How are you
        if any(phrase in question_lower for phrase in ['how are you', 'how are things', 'how\'s it going']):
            return "I'm doing great, thank you for asking! 🤖 I'm ready to help you with any questions about our services. What would you like to know?"
        
        # Thank you
        if any(phrase in question_lower for phrase in ['thank you', 'thanks', 'thx']):
            return "You're very welcome! 😊 I'm happy I could help. Is there anything else you'd like to know?"
        
        # Goodbye
        if any(phrase in question_lower for phrase in ['bye', 'goodbye', 'see you', 'farewell']):
            return "Goodbye! 👋 Thank you for using our FAQ chatbot. Have a wonderful day!"
        
        # Help requests
        if question_lower in ['help', '?', 'what can you do']:
            return """I can help you with questions about our products and services! Here are some common questions:

• How do I return a product?
• What are your shipping options?
• What payment methods do you accept?
• How can I track my order?
• Do you offer customer support?

Just ask me anything!"""
        
        return None  # Not small talk, proceed with normal processing
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="FAQ Chatbot",
            page_icon="🤖",
            layout="centered"
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'chat_started' not in st.session_state:
            st.session_state.chat_started = False
        
        if 'start_time' not in st.session_state:
            st.session_state.start_time = None
        
        if 'feedback_given' not in st.session_state:
            st.session_state.feedback_given = False
        
        if 'show_feedback' not in st.session_state:
            st.session_state.show_feedback = False
    
    def load_chatbot(self):
        """Load the chatbot"""
        try:
            with st.spinner("Loading chatbot..."):
                embeddings_path = "faq_vector_store"
                st.session_state.chatbot = LocalRAGChatbot(embeddings_path)
            st.success("✅ Chatbot loaded successfully!", icon="🤖")
        except Exception as e:
            st.error(f"❌ Failed to load chatbot: {e}")
            st.stop()
    
    def setup_mysql(self):
        """Setup MySQL connection with environment variables"""
        try:
            # Initialize MySQL storage using environment variables from .env file
            st.session_state.mysql_storage = MySQLChatStorage()
        except Exception as e:
            st.error(f"❌ MySQL connection failed: {e}")
            st.info("Please ensure MySQL is running and check your .env file with:")
            st.code("""
MYSQL_HOST=localhost
MYSQL_USER=root  
MYSQL_PASSWORD=your_password
MYSQL_DB=chatbot_feedback
            """)
            st.stop()
    
    def start_new_chat(self):
        """Start a new chat session"""
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.chat_started = True
        st.session_state.start_time = datetime.now().isoformat()
        st.session_state.feedback_given = False
        st.session_state.show_feedback = False
        st.rerun()
    
    def end_chat_session(self):
        """End current chat and save to MySQL"""
        if len(st.session_state.messages) == 0:
            st.warning("No messages to save.")
            return
        
        # Show feedback form
        st.subheader("📝 Chat Feedback")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Rating
            rating = st.select_slider(
                "How would you rate this chat session?",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x,
                key="rating_slider"
            )
            
            # Comment
            feedback_comment = st.text_area(
                "Any additional feedback? (optional)",
                placeholder="Tell us how we can improve...",
                key="feedback_text"
            )
            
            # Optional email
            user_email = st.text_input(
                "Email (optional)",
                placeholder="your.email@example.com",
                key="user_email"
            )
        
        with col2:
            st.metric("Session ID", st.session_state.session_id)
            st.metric("Messages", len(st.session_state.messages))
            st.metric("Interactions", f"{len(st.session_state.messages) // 2}")
        
        # Buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("💾 Save Chat & Feedback", type="primary", use_container_width=True):
                self.save_chat_with_feedback(rating, feedback_comment, user_email)
        
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_feedback = False
                st.rerun()
    
    def save_chat_with_feedback(self, rating: int, feedback_comment: str, user_email: str):
        """Save the complete chat session with feedback and emotion data"""
        try:
            # Convert messages to ChatMessage objects
            chat_messages = []
            for msg in st.session_state.messages:
                chat_msg = ChatMessage(
                    role=msg["role"],
                    content=msg["content"], 
                    timestamp=msg.get("timestamp", datetime.now().isoformat()),
                    confidence_score=msg.get("confidence_score"),
                    sources_used=msg.get("sources_used"),
                    response_time=msg.get("response_time"),
                    detected_emotion=msg.get("detected_emotion"),
                    emotion_score=msg.get("emotion_score")
                )
                chat_messages.append(chat_msg)
            
            # Create chat session
            session = ChatSession(
                session_id=st.session_state.session_id,
                start_time=st.session_state.start_time,
                end_time=datetime.now().isoformat(),
                messages=chat_messages,
                overall_rating=rating,
                feedback_comment=feedback_comment.strip() if feedback_comment.strip() else None,
                user_email=user_email.strip() if user_email.strip() else None,
                total_interactions=len([m for m in st.session_state.messages if m["role"] == "user"])
            )
            
            # Save to MySQL
            success = st.session_state.mysql_storage.save_chat_session(session)
            
            if success:
                st.success("✅ Chat saved successfully! Thank you for your feedback.")
                st.session_state.feedback_given = True
                st.session_state.show_feedback = False
                
                # Show emotion summary
                user_emotions = [m.get("detected_emotion") for m in st.session_state.messages if m["role"] == "user" and m.get("detected_emotion")]
                if user_emotions:
                    dominant_emotion = max(set(user_emotions), key=user_emotions.count)
                    st.info(f"🎭 Session Summary: Dominant emotion was '{dominant_emotion}'")
                
                # Show new chat button
                st.balloons()
                if st.button("🆕 Start New Chat", type="primary", use_container_width=True):
                    self.start_new_chat()
            else:
                st.error("❌ Failed to save chat. Please try again.")
                
        except Exception as e:
            st.error(f"❌ Error saving chat: {e}")
    
    def display_chat_interface(self):
        """Main chat interface"""
        st.title("🤖 FAQ Chatbot")
        
        # Show initial greeting if no messages yet
        if len(st.session_state.messages) == 0:
            with st.chat_message("assistant"):
                st.write("Hello! 👋 Welcome to our FAQ chatbot.")
                st.write("I'm here to help you with questions about our products and services.")
                st.write("You can ask me about returns, shipping, payments, and more!")
                st.write("What can I help you with today?")
        
        # Chat controls at the top
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.caption(f"💬 Session: {st.session_state.session_id}")
        
        with col2:
            if st.session_state.chat_started and len(st.session_state.messages) > 0:
                if st.button("🔚 End Chat", use_container_width=True):
                    st.session_state.show_feedback = True
                    st.rerun()
        
        with col3:
            if st.button("🆕 New Chat", use_container_width=True):
                self.start_new_chat()
        
        st.divider()
        
        # Chat container
        chat_container = st.container(height=500)
        
        with chat_container:
            # Display chat messages with emotion indicators
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    # Show emotion info for user messages
                    if message["role"] == "user" and message.get("detected_emotion"):
                        emotion_color = {
                            'positive': '🟢',
                            'negative': '🔴', 
                            'neutral': '⚪',
                            'complex': '🟡'
                        }
                        emotion_category = message.get("emotion_category", "neutral")
                        emotion_icon = emotion_color.get(emotion_category, '⚪')
                        
                        st.caption(f"🎭 Emotion: {emotion_icon} {message['detected_emotion']} ({message.get('emotion_score', 0):.2f})")
                    
                    # Show confidence and emotion info for assistant messages
                    elif message["role"] == "assistant" and message.get("confidence_score"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.caption(f"🎯 Confidence: {message['confidence_score']:.2f}")
                        with col2:
                            st.caption(f"📚 Sources: {message.get('sources_used', 0)}")
                        with col3:
                            st.caption(f"⏱️ Time: {message.get('response_time', 0):.1f}s")
                        with col4:
                            # Show emotion score from the corresponding user message
                            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
                            if user_msgs:
                                last_user_msg = user_msgs[-1]
                                if last_user_msg.get("emotion_score"):
                                    st.caption(f"🎭 User Emotion: {last_user_msg['emotion_score']:.2f}")
                                else:
                                    st.caption(f"🎭 User Emotion: N/A")
                        
                        # Show escalation notice if applicable
                        if message.get("needs_escalation"):
                            st.warning("🔔 This conversation was flagged for human support due to detected frustration")
        
        # Chat input
        if not st.session_state.get('feedback_given', False):
            if prompt := st.chat_input("Ask a question..."):
                self.handle_user_message(prompt)
        else:
            st.info("💬 Chat session ended. Click 'New Chat' to continue.")
    
    def handle_user_message(self, prompt: str):
        """Handle user message and get bot response"""
        # Start chat if not started
        if not st.session_state.chat_started:
            st.session_state.chat_started = True
            st.session_state.start_time = datetime.now().isoformat()
        
        # Add user message
        user_msg = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_msg)
        
        # Check if it's small talk first
        small_talk_response = self.handle_small_talk(prompt)
        
        if small_talk_response:
            # Handle small talk without using the LLM
            assistant_msg = {
                "role": "assistant", 
                "content": small_talk_response,
                "timestamp": datetime.now().isoformat(),
                "confidence_score": 1.0,  # High confidence for predefined responses
                "sources_used": 0,
                "response_time": 0.1  # Very fast
            }
        else:
            # Get bot response for FAQ questions
            with st.spinner("🤖 Thinking..."):
                response = st.session_state.chatbot.ask(prompt)
            
            # Add assistant message
            assistant_msg = {
                "role": "assistant", 
                "content": response.answer,
                "timestamp": datetime.now().isoformat(),
                "confidence_score": response.confidence_score,
                "sources_used": response.sources_used,
                "response_time": response.response_time_seconds
            }
        
        st.session_state.messages.append(assistant_msg)
        st.rerun()
    
    def run(self):
        """Run the Streamlit app"""
        # Show feedback form if requested
        if st.session_state.get('show_feedback', False) and not st.session_state.get('feedback_given', False):
            self.end_chat_session()
        else:
            self.display_chat_interface()

def main():
    """Main function"""
    # Check requirements
    try:
        import streamlit as st
        import mysql.connector
    except ImportError as e:
        st.error(f"Missing requirements: {e}")
        st.info("Install with: pip install streamlit mysql-connector-python")
        st.stop()
    
    # Check for embeddings
    from pathlib import Path
    embeddings_path = "faq_vector_store"
    if not Path(f"{embeddings_path}.faiss").exists():
        st.error("❌ FAQ embeddings not found. Please run the embedding pipeline first.")
        st.info("Run the FAQ embedding pipeline to create 'faq_vector_store' files.")
        st.stop()
    
    # Run app
    app = StreamlitChatApp()
    app.run()

if __name__ == "__main__":
    main()