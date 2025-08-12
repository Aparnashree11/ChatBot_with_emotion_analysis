import os
# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import emotion model
from emotion_classification_training import EmotionPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatEmotionDashboard:
    def __init__(self):
        self.setup_page_config()
        self.initialize_components()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Chat Emotion Dashboard",
            page_icon="🎭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_components(self):
        # Initialize session state
        if 'selected_session' not in st.session_state:
            st.session_state.selected_session = None
        
        if 'emotion_data' not in st.session_state:
            st.session_state.emotion_data = None
        
        # Load emotion model
        self.load_emotion_model()
        
        # Setup MySQL connection
        self.setup_mysql()
    
    def load_emotion_model(self):
        if 'emotion_predictor' not in st.session_state:
            model_path = "./distilbert-goemotions-10k"  # Update path as needed
            
            try:
                with st.spinner("Loading emotion model..."):
                    st.session_state.emotion_predictor = EmotionPredictor(model_path)
                st.success("Emotion model loaded!")
            except Exception as e:
                st.error(f"Failed to load emotion model: {e}")
                st.info("Please ensure the trained model exists at the specified path")
                st.stop()
    
    def setup_mysql(self):
        """Setup MySQL connection"""
        if 'mysql_config' not in st.session_state:
            st.session_state.mysql_config = {
                'host': os.getenv('MYSQL_HOST', 'localhost'),
                'user': os.getenv('MYSQL_USER', 'root'),
                'password': os.getenv('MYSQL_PASSWORD', ''),
                'database': os.getenv('MYSQL_DB', 'chatbot_feedback'),
                'port': 3306,
                'charset': 'utf8mb4'
            }
    
    def get_chat_data(self, days_back: int = 30) -> pd.DataFrame:
        """Retrieve chat data using correct table structure with proper JOIN"""
        try:
            conn = mysql.connector.connect(**st.session_state.mysql_config)
            
            # Correct query using actual table columns
            query = '''
            SELECT 
                cs.session_id,
                cs.start_time,
                cs.end_time,
                cs.total_interactions,
                cs.overall_rating,
                cs.feedback_comment,
                cs.user_email,
                im.message_order,
                im.role,
                im.content as message_content,
                im.timestamp as message_timestamp,
                im.confidence_score,
                im.sources_used,
                im.response_time
            FROM chat_sessions cs
            JOIN individual_messages im ON cs.session_id = im.session_id
            WHERE cs.start_time >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY cs.start_time DESC, im.message_order ASC
            '''
            
            df = pd.read_sql(query, conn, params=(days_back,))
            conn.close()
            
            logger.info(f"Retrieved {len(df)} messages from {df['session_id'].nunique()} sessions")
            return df
            
        except Exception as e:
            st.error(f"❌ Error retrieving chat data: {e}")
            st.error(f"SQL Error details: {e}")
            return pd.DataFrame()
    
    def categorize_emotion(self, emotion: str) -> str:
        """Categorize emotion into broad categories"""
        positive = {'joy', 'love', 'excitement', 'gratitude', 'admiration', 'amusement', 'optimism', 'pride', 'relief'}
        negative = {'anger', 'sadness', 'fear', 'disgust', 'disappointment', 'annoyance', 'embarrassment', 'grief', 'remorse'}
        neutral = {'neutral', 'realization', 'approval', 'disapproval'}
        complex = {'confusion', 'curiosity', 'desire', 'nervousness', 'surprise', 'caring'}
        
        if emotion in positive:
            return 'positive'
        elif emotion in negative:
            return 'negative'
        elif emotion in neutral:
            return 'neutral'
        elif emotion in complex:
            return 'complex'
        else:
            return 'other'
    
    def create_overall_dashboard(self, df: pd.DataFrame):
        st.header("Overall Chatbot Analytics")
        
        if df.empty:
            st.warning("No data available for analysis")
            return
        
        # Get session-level data (deduplicated by session_id)
        sessions = df.groupby('session_id').agg({
            'start_time': 'first',
            'overall_rating': 'first',
            'total_interactions': 'first',
            'feedback_comment': 'first'
        }).reset_index()
        
        # Overall metrics
        total_sessions = len(sessions)
        avg_rating = sessions['overall_rating'].dropna().mean() if not sessions['overall_rating'].dropna().empty else 0
        avg_interactions = sessions['total_interactions'].dropna().mean() if not sessions['total_interactions'].dropna().empty else 0
        total_user_messages = len(df[df['role'] == 'user'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", total_sessions)
        
        with col2:
            if avg_rating > 0:
                st.metric("Avg Rating", f"{avg_rating:.2f}/5.0")
            else:
                st.metric("Avg Rating", "No ratings")
        
        with col3:
            if avg_interactions > 0:
                st.metric("Avg Interactions", f"{avg_interactions:.1f}")
            else:
                st.metric("Avg Interactions", "N/A")
        
        with col4:
            st.metric("User Messages", total_user_messages)
        
        st.divider()
        
        # Analyze emotions for all user messages
        user_messages = df[df['role'] == 'user']
        if not user_messages.empty:
            with st.spinner("Analyzing emotions across all chats..."):
                self.display_overall_emotion_analysis(user_messages, sessions)
        else:
            st.warning("No user messages found to analyze")
    
    def display_overall_emotion_analysis(self, user_messages: pd.DataFrame, sessions: pd.DataFrame):
        """Display overall emotion analysis with error handling"""
        
        # Analyze emotions for all messages
        all_emotions = []
        session_emotions = {}
        
        progress_bar = st.progress(0)
        total_messages = len(user_messages)
        
        if total_messages == 0:
            st.warning("No user messages found to analyze")
            return
        
        for idx, (_, message_row) in enumerate(user_messages.iterrows()):
            try:
                emotions = st.session_state.emotion_predictor.predict_top_k(
                    message_row['message_content'], k=1
                )
                primary_emotion = emotions[0][0] if emotions else 'neutral'
                emotion_category = self.categorize_emotion(primary_emotion)
                
                all_emotions.append({
                    'session_id': message_row['session_id'],
                    'emotion': primary_emotion,
                    'category': emotion_category,
                    'timestamp': message_row['message_timestamp']
                })
                
                # Track emotions per session
                if message_row['session_id'] not in session_emotions:
                    session_emotions[message_row['session_id']] = []
                session_emotions[message_row['session_id']].append(emotion_category)
                
            except Exception as e:
                # Skip problematic messages
                continue
            
            # Update progress
            progress_bar.progress((idx + 1) / total_messages)
        
        progress_bar.empty()
        
        if not all_emotions:
            st.warning("No emotions could be analyzed")
            return
        
        emotions_df = pd.DataFrame(all_emotions)
        
        # Overall emotion trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Emotion Distribution")
            
            emotion_counts = emotions_df['emotion'].value_counts()
            if not emotion_counts.empty:
                fig_emotions = px.bar(
                    x=emotion_counts.index,
                    y=emotion_counts.values,
                    title="Individual Emotions",
                    labels={'x': 'Emotion', 'y': 'Count'},
                    color=emotion_counts.values,
                    color_continuous_scale='viridis'
                )
                fig_emotions.update_layout(xaxis_tickangle=45, height=400)
                st.plotly_chart(fig_emotions, use_container_width=True)
        
        with col2:
            st.subheader("Emotion Categories")
            
            category_counts = emotions_df['category'].value_counts()
            if not category_counts.empty:
                colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6', 'complex': '#f39c12'}
                
                fig_categories = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Emotion Category Breakdown",
                    color=category_counts.index,
                    color_discrete_map=colors
                )
                st.plotly_chart(fig_categories, use_container_width=True)
        
        # Additional analysis only if we have valid data
        self.display_additional_analysis(emotions_df, sessions, session_emotions)
    
    def display_additional_analysis(self, emotions_df: pd.DataFrame, sessions: pd.DataFrame, session_emotions: Dict):
       # Emotions over time
        if 'timestamp' in emotions_df.columns and not emotions_df.empty:
            st.subheader("Emotion Trends Over Time")
            
            try:
                emotions_df['date'] = pd.to_datetime(emotions_df['timestamp']).dt.date
                daily_emotions = emotions_df.groupby(['date', 'category']).size().reset_index(name='count')
                
                if not daily_emotions.empty:
                    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6', 'complex': '#f39c12'}
                    fig_trends = px.line(
                        daily_emotions,
                        x='date',
                        y='count',
                        color='category',
                        title="Daily Emotion Trends",
                        color_discrete_map=colors
                    )
                    st.plotly_chart(fig_trends, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display time trends: {e}")
        
        # Emotion vs Rating correlation
        if 'overall_rating' in sessions.columns and sessions['overall_rating'].notna().any():
            st.subheader("Emotion-Rating Correlation")
            
            try:
                # Calculate dominant emotion per session
                session_dominant_emotions = {}
                for session_id, emotions in session_emotions.items():
                    if emotions:
                        dominant = max(set(emotions), key=emotions.count)
                        session_dominant_emotions[session_id] = dominant
                
                # Merge with ratings
                rating_emotion_data = []
                for _, session in sessions.iterrows():
                    session_id = session['session_id']
                    rating = session['overall_rating']
                    
                    if pd.notna(rating) and session_id in session_dominant_emotions:
                        rating_emotion_data.append({
                            'session_id': session_id,
                            'rating': rating,
                            'dominant_emotion': session_dominant_emotions[session_id]
                        })
                
                if rating_emotion_data:
                    rating_df = pd.DataFrame(rating_emotion_data)
                    
                    fig_correlation = px.box(
                        rating_df,
                        x='dominant_emotion',
                        y='rating',
                        title="Chat Rating by Dominant Emotion",
                        labels={'rating': 'Chat Rating (1-5)', 'dominant_emotion': 'Dominant Emotion'}
                    )
                    fig_correlation.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_correlation, use_container_width=True)
                else:
                    st.info("No rating data available for correlation analysis")
                    
            except Exception as e:
                st.warning(f"Could not display rating correlation: {e}")
    
    def create_session_sidebar(self, df: pd.DataFrame) -> Optional[str]:
        """Create sidebar with session selection using correct data structure"""
        st.sidebar.header("Chat Sessions")
        
        if df.empty:
            st.sidebar.warning("No sessions available")
            return None
        
        # Get unique sessions with metadata (aggregate from message data)
        sessions = df.groupby('session_id').agg({
            'start_time': 'first',
            'overall_rating': 'first',
            'total_interactions': 'first',
            'feedback_comment': 'first'
        }).reset_index()
        
        sessions = sessions.sort_values('start_time', ascending=False)
        
        # Create session display names with better error handling
        session_options = []
        for _, session in sessions.iterrows():
            try:
                # Safe datetime conversion
                if pd.notna(session['start_time']):
                    start_time = pd.to_datetime(session['start_time']).strftime('%m-%d %H:%M')
                else:
                    start_time = "Unknown"
                
                # Safe rating display
                if pd.notna(session['overall_rating']):
                    rating = f"{int(session['overall_rating'])}"
                else:
                    rating = "No rating"
                
                # Safe interaction count
                if pd.notna(session['total_interactions']):
                    interactions = f"{int(session['total_interactions'])}msgs"
                else:
                    interactions = "0msgs"
                
                display_name = f"{session['session_id']} | {start_time} | {rating} | {interactions}"
                session_options.append((display_name, session['session_id']))
                
            except Exception as e:
                # Skip problematic sessions
                continue
        
        # Session selection
        st.sidebar.subheader("Select Session")
        
        if session_options:
            selected_display = st.sidebar.selectbox(
                "Choose a chat session:",
                options=[opt[0] for opt in session_options],
                key="session_selector"
            )
            
            # Get actual session ID
            selected_session_id = None
            for display, session_id in session_options:
                if display == selected_display:
                    selected_session_id = session_id
                    break
            
            return selected_session_id
        
        st.sidebar.warning("No valid sessions found")
        return None
    
    def display_session_analysis(self, df: pd.DataFrame, session_id: str):
        """Display detailed analysis for a specific session"""
        st.header(f"Session Analysis: {session_id}")
        
        # Get session data using the correct column structure
        session_data = df[df['session_id'] == session_id].sort_values('message_order')
        
        if session_data.empty:
            st.error("No data found for this session")
            return
        
        # Session metadata from first row
        session_info = session_data.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if pd.notna(session_info['start_time']):
                start_time_str = pd.to_datetime(session_info['start_time']).strftime('%Y-%m-%d %H:%M')
                st.metric("Start Time", start_time_str)
            else:
                st.metric("Start Time", "Unknown")
        
        with col2:
            rating = session_info['overall_rating']
            if pd.notna(rating):
                st.metric("Rating", f"{int(rating)}/5")
            else:
                st.metric("Rating", "Not rated")
        
        with col3:
            st.metric("Messages", len(session_data))
        
        with col4:
            interactions = session_info['total_interactions']
            if pd.notna(interactions):
                st.metric("Interactions", f"{int(interactions)}")
            else:
                st.metric("Interactions", "0")
        
        st.divider()
        
        # Analyze emotions in this session
        with st.spinner("Analyzing emotions in this session..."):
            emotion_analysis = self.analyze_single_session(session_data)
        
        if not emotion_analysis or not emotion_analysis.get('emotions_data'):
            st.warning("No user messages found in this session to analyze")
            return
        
        # Display emotion analysis
        self.display_session_emotion_details(emotion_analysis, session_data)
        
        # Display feedback at the end
        st.subheader("Session Feedback")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if pd.notna(session_info['feedback_comment']) and session_info['feedback_comment'].strip():
                st.info(f"**User Feedback:** {session_info['feedback_comment']}")
            else:
                st.info("**User Feedback:** No feedback provided")
        
        with col2:
            if pd.notna(session_info['overall_rating']):
                rating_stars = "⭐" * int(session_info['overall_rating'])
                st.info(f"**Final Rating:** {rating_stars} ({int(session_info['overall_rating'])}/5)")
            else:
                st.info("**Final Rating:** Not rated")
    
    def analyze_single_session(self, session_data: pd.DataFrame) -> Dict:
        """Analyze emotions for a single session using correct data structure"""
        # Filter user messages using correct column name
        user_messages = session_data[session_data['role'] == 'user']
        
        if user_messages.empty:
            return {}
        
        emotions_data = []
        emotion_sequence = []
        
        for _, row in user_messages.iterrows():
            # Use correct column name
            message = row['message_content']
            
            # Skip empty messages
            if not message or not message.strip():
                continue
            
            try:
                # Get emotion predictions - only use predict_top_k (which exists)
                top_emotions = st.session_state.emotion_predictor.predict_top_k(message, k=3)
                
                # Get all emotions using predict method (now added)
                all_emotions = st.session_state.emotion_predictor.predict_top_k(message, k=3)
                
                primary_emotion = top_emotions[0][0] if top_emotions else 'neutral'
                primary_score = top_emotions[0][1] if top_emotions else 0.0
                emotion_category = self.categorize_emotion(primary_emotion)
                
                emotions_data.append({
                    'message_order': row['message_order'],
                    'message': message,
                    'primary_emotion': primary_emotion,
                    'primary_score': primary_score,
                    'emotion_category': emotion_category,
                    'top_emotions': top_emotions,
                    'all_emotions': all_emotions,
                    'timestamp': row['message_timestamp']
                })
                
                emotion_sequence.append(emotion_category)
                
            except Exception as e:
                # For debugging - show which message failed
                st.error(f"Error analyzing message '{message[:50]}...': {e}")
                continue
        
        # Safe calculations
        emotion_distribution = {}
        if emotion_sequence:
            emotion_counts = pd.Series(emotion_sequence).value_counts()
            emotion_distribution = emotion_counts.to_dict()
        
        avg_confidence = 0
        if emotions_data:
            confidence_scores = [e['primary_score'] for e in emotions_data if e['primary_score'] > 0]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'emotions_data': emotions_data,
            'emotion_sequence': emotion_sequence,
            'session_emotion_summary': {
                'total_messages': len(emotions_data),
                'dominant_emotion': max(set(emotion_sequence), key=emotion_sequence.count) if emotion_sequence else 'neutral',
                'emotion_distribution': emotion_distribution,
                'avg_confidence': avg_confidence
            }
        }
    
    def display_session_emotion_details(self, emotion_analysis: Dict, session_data: pd.DataFrame):
        """Display detailed emotion analysis for a session with safe calculations"""
        
        emotions_data = emotion_analysis['emotions_data']
        emotion_sequence = emotion_analysis['emotion_sequence']
        summary = emotion_analysis['session_emotion_summary']
        
        # Session emotion summary with safe calculations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Session Emotion Summary")
            
            # Safe emotion distribution pie chart
            emotion_dist = summary.get('emotion_distribution', {})
            if emotion_dist:
                fig_session_pie = px.pie(
                    values=list(emotion_dist.values()),
                    names=list(emotion_dist.keys()),
                    title="Emotion Categories in This Session",
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'negative': '#e74c3c',
                        'neutral': '#95a5a6',
                        'complex': '#f39c12'
                    }
                )
                st.plotly_chart(fig_session_pie, use_container_width=True)
            else:
                st.info("No emotion distribution data available")
        
        with col2:
            st.subheader("Emotion Journey")
            
            # Emotion sequence over messages with safe handling
            if emotion_sequence and len(emotion_sequence) > 0:
                emotion_nums = list(range(1, len(emotion_sequence) + 1))
                colors = [{'positive': 1, 'negative': -1, 'neutral': 0, 'complex': 0.5}.get(cat, 0) for cat in emotion_sequence]
                
                fig_journey = go.Figure()
                fig_journey.add_trace(go.Scatter(
                    x=emotion_nums,
                    y=colors,
                    mode='lines+markers',
                    marker=dict(size=10, color=colors, colorscale='RdYlGn', cmin=-1, cmax=1),
                    line=dict(width=3),
                    text=emotion_sequence,
                    hovertemplate='Message %{x}<br>Emotion: %{text}<extra></extra>'
                ))
                
                fig_journey.update_layout(
                    title="Emotional Journey Throughout Chat",
                    xaxis_title="Message Number",
                    yaxis_title="Emotion Valence",
                    yaxis=dict(tickvals=[-1, -0.5, 0, 0.5, 1], 
                              ticktext=['Negative', 'Mixed-', 'Neutral', 'Mixed+', 'Positive']),
                    height=300
                )
                
                st.plotly_chart(fig_journey, use_container_width=True)
            else:
                st.info("No emotion journey data available")
        
        # Session metrics with safe calculations
        st.subheader("Session Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dominant_emotion = summary.get('dominant_emotion', 'unknown')
            st.metric("Dominant Emotion", dominant_emotion.title())
        
        with col2:
            avg_confidence = summary.get('avg_confidence', 0)
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            # Safe positive ratio calculation
            total_messages = summary.get('total_messages', 0)
            positive_count = summary.get('emotion_distribution', {}).get('positive', 0)
            
            if total_messages > 0:
                positive_ratio = positive_count / total_messages
                st.metric("Positive Ratio", f"{positive_ratio:.1%}")
            else:
                st.metric("Positive Ratio", "N/A")
        
        # Detailed message analysis
        st.subheader("Message-by-Message Analysis")
        
        if emotions_data:
            for emotion_data in emotions_data:
                with st.expander(f"Message {emotion_data['message_order']}: {emotion_data['primary_emotion'].title()} (confidence: {emotion_data['primary_score']:.3f})"):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**User Message:** {emotion_data['message']}")
                        
                        # Show top 3 emotions
                        st.write("**Top Emotions:**")
                        for emotion, score in emotion_data.get('top_emotions', []):
                            st.write(f"  • {emotion}: {score:.3f}")
                    
                    with col2:
                        # Emotion category badge
                        category = emotion_data['emotion_category']
                        category_colors = {
                            'positive': '🟢',
                            'negative': '🔴', 
                            'neutral': '⚪',
                            'complex': '🟡'
                        }
                        
                        st.write(f"**Category:** {category_colors.get(category, '⚪')} {category.title()}")
                        st.write(f"**Confidence:** {emotion_data['primary_score']:.3f}")
                        
                        if emotion_data.get('timestamp'):
                            timestamp_str = pd.to_datetime(emotion_data['timestamp']).strftime('%H:%M:%S')
                            st.write(f"**Time:** {timestamp_str}")
        else:
            st.info("No emotion data available for this session")
    
    def run(self):
        """Main app runner"""
        # Controls
        st.sidebar.header("Dashboard Controls")
        days_back = st.sidebar.slider("Days to analyze", 1, 90, 30, help="Number of days of chat history to analyze")
        
        if st.sidebar.button("Refresh Data", type="primary"):
            if 'chat_data' in st.session_state:
                del st.session_state.chat_data
        
        # Load data
        if 'chat_data' not in st.session_state:
            with st.spinner("Loading chat data..."):
                st.session_state.chat_data = self.get_chat_data(days_back)
        
        chat_df = st.session_state.chat_data
        
        if chat_df.empty:
            st.error("No chat data found")
            st.info("Make sure MySQL is running and contains chat data")
            return
        
        # Create two-column layout
        col_sessions, col_main = st.columns([1, 10])
        
        with col_sessions:
            # Session sidebar
            selected_session = self.create_session_sidebar(chat_df)
        
        with col_main:
            st.title("Chat Emotion Analysis Dashboard")
            st.markdown("Analyze emotions and trends in chatbot conversations")
        
            # Main content
            if selected_session:
                # Show individual session analysis
                self.display_session_analysis(chat_df, selected_session)
            else:
                # Show overall dashboard
                self.create_overall_dashboard(chat_df)
                
                # Instructions
                st.info("Select a session from the left sidebar to view detailed emotion analysis")

def main():
    """Main entry point"""
    
    # Check if model exists
    from pathlib import Path
    model_path = "./distilbert-goemotions-10k"
    
    if not Path(model_path).exists():
        st.error(f"Emotion model not found at {model_path}")
        st.info("Please train the DistilBERT emotion model first")
        st.stop()
    
    # Run dashboard
    dashboard = ChatEmotionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()