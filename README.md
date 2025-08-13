# Intelligent E-Commerce FAQ Chatbot with Emotion Analysis

A complete end-to-end chatbot system that combines **Retrieval-Augmented Generation (RAG)**, **emotion analysis**, and **feedback collection** for intelligent customer service automation.

---

## Project Overview

This project implements a sophisticated chatbot system that:
- **Answers FAQ questions** using semantic search and local LLM generation
- **Analyzes user emotions** in real-time using fine-tuned DistilBERT
- **Collects user feedback** and ratings for continuous improvement
- **Provides analytics dashboards** for monitoring and optimization

---

## Key Features

### RAG (Retrieval-Augmented Generation)
- **Semantic Search**: FAISS vector database for FAQ retrieval
- **Local LLM**: TinyLlama-1.1B-Chat for response generation
- **Smart Context**: Combines relevant FAQs for accurate answers
- **Fast Performance**: 3-6 second response times

### Emotion Analysis
- **Real-time Detection**: 28 different emotions using DistilBERT
- **GoEmotions Dataset**: Trained on 40,000 samples for accuracy
- **Emotion Categories**: Positive, Negative, Neutral, Complex emotions
- **Business Intelligence**: Correlate emotions with satisfaction ratings

### Interactive Chat Interface
- **Streamlit Web App**: Clean, modern chat interface
- **Smart Small Talk**: Instant responses for greetings and casual conversation
- **Session Management**: Unique session tracking with timestamps
- **MySQL Storage**: Complete conversation logging

### Analytics & Feedback
- **Feedback Collection**: Star ratings and text comments
- **Emotion Dashboard**: Visualize emotion trends and patterns
- **Session Analysis**: Detailed per-conversation emotion journeys
- **Continuous Improvement**: Data-driven optimization suggestions

---

## System Architecture

```
User Input → Small Talk Handler → RAG Pipeline → Response Generation
     ↓              ↓                   ↓              ↓
Emotion Analysis → MySQL Storage → Analytics Dashboard → Insights
```

### Core Components:
1. **FAQ Embeddings**: Sentence Transformers + FAISS
2. **Response Generation**: TinyLlama-1.1B local inference
3. **Emotion Analysis**: DistilBERT + GoEmotions dataset
4. **Data Storage**: MySQL with session and message tracking
5. **User Interface**: Streamlit with real-time chat
6. **Analytics**: Plotly visualizations and trend analysis

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for local LLM)
- MySQL server
- 4GB+ disk space

### 1. Clone and Setup Environment
```bash
git clone https://github.com/Aparnashree11/ChatBot_with_emotion_analysis.git
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup MySQL Database
```sql
-- Create database
CREATE DATABASE chatbot_feedback CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create user (optional)
CREATE USER 'chatbot_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON chatbot_feedback.* TO 'chatbot_user'@'localhost';
FLUSH PRIVILEGES;
```

### 4. Environment Configuration
Create `.env` file:
```env
MYSQL_HOST=localhost
MYSQL_USER=chatbot_user
MYSQL_PASSWORD=your_password
MYSQL_DB=chatbot_feedback
```

---

## Quick Start Guide

### Step 1: Generate FAQ Embeddings
```bash
python embeddings_gen.py
```
- Processes your FAQ data (CSV format)
- Creates FAISS vector store for semantic search
- **Output**: `faq_vector_store.faiss` + metadata

### Step 2: Train Emotion Model
```bash
python emotion_classification_training.py
```
- Fine-tunes DistilBERT on GoEmotions dataset
- Training time: ~2-3 hours for 10,000 samples
- **Output**: `distilbert-goemotions-10k/` model directory

### Step 3: Launch Chat Interface
```bash
streamlit run new_app.py
```
- Opens chat interface at `http://localhost:8501`
- Users can chat and provide feedback
- **Features**: Real-time chat, session management, feedback collection

### Step 4: View Analytics Dashboard
```bash
streamlit run main.py
```
- Analytics dashboard at `http://localhost:8501`
- **Features**: Emotion trends, session analysis, rating correlations

---

## Usage Examples

### Basic Chat Interface
```python
from model_testing import create_balanced_chatbot

# Initialize chatbot
chatbot = create_balanced_chatbot("faq_vector_store")

# Ask questions
response = chatbot.ask("How do I return a product?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score}")
```

### Emotion Analysis
```python
from emotion_classification_training import EmotionPredictor

# Load emotion model
predictor = EmotionPredictor("./distilbert-goemotions-10k")

# Analyze emotions
emotions = predictor.predict_top_k("I'm so frustrated with this!", k=3)
print(emotions)  # [('anger', 0.87), ('annoyance', 0.72), ('disappointment', 0.54)]
```

### Streamlit Apps
```bash
# Chat interface
streamlit run new_app.py

# Analytics dashboard  
streamlit run main.py
```

---

## Performance Benchmarks

### Response Times
- **Small Talk**: 0.1 seconds (instant)
- **FAQ Questions**: 5-10 seconds average
- **Cold Start**: 10-15 seconds (model loading)

### Accuracy Metrics
- **FAQ Retrieval**: Confidence-scored semantic matching
- **Emotion Detection**: F1-macro ~0.45-0.50 (28 emotions)
- **User Satisfaction**: Tracked via ratings and feedback

### Resource Usage
- **RAM**: 2-3GB with model quantization
- **Storage**: 3-4GB total (models + embeddings)
- **GPU**: Optional but recommended for training

## Configuration Options

### RAG Chatbot Settings
```python
chatbot = LocalRAGChatbot(
    embeddings_path="faq_vector_store",
    similarity_threshold=0.5,    # Higher = more selective
    max_retrieved_faqs=2,        # Number of FAQs to use
    device="cpu"                 # "cpu" or "cuda"
)
```

### Emotion Model Settings
```python
config = TrainingConfig(
    max_train_samples=10000,     # Training data size
    num_epochs=2,                # Training epochs
    batch_size=32,               # Batch size
    learning_rate=5e-5           # Learning rate
)
```

---

## Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check if model files exist
ls -la distilbert-goemotions-10k/
# Should contain: config.json, pytorch_model.bin, tokenizer files
```

**Memory Issues**
```python
# Reduce batch size or use CPU
config.batch_size = 8
config.device = "cpu"
```

**MySQL Connection**
```bash
# Test MySQL connection
mysql -u root -p chatbot_feedback
# Verify tables exist: SHOW TABLES;
```

**Slow Performance**
- Use GPU if available
- Reduce FAQ database size
- Lower similarity threshold
- Use smaller models

---

## Analytics & Insights

### Emotion Analytics Dashboard
- **Overall Trends**: Emotion distribution across all chats
- **Session Analysis**: Click any session for detailed emotion journey
- **Rating Correlation**: How emotions affect user satisfaction
- **Time Patterns**: Emotion trends by day/hour

### Business Metrics
- **Customer Satisfaction**: Average ratings and positive emotion rates
- **Content Gaps**: Identify FAQ areas needing improvement
- **Performance Monitoring**: Response times and confidence scores
- **User Behavior**: Conversation patterns and emotional journeys

---

## Continuous Improvement

### Feedback Loop
1. **Users chat** → Provide ratings and feedback
2. **System analyzes** → Emotion patterns and satisfaction
3. **Identifies gaps** → FAQ content and response quality
4. **Generates insights** → Optimization recommendations
5. **Updates system** → Better responses and user experience

### Optimization Areas
- **FAQ Content**: Add missing questions, improve answers
- **Model Parameters**: Adjust similarity thresholds and context size
- **Response Quality**: Fine-tune prompts and generation settings
- **User Experience**: Enhance based on emotional feedback patterns

---

## Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test locally
4. Commit changes: `git commit -m "Add new feature"`
5. Push to branch: `git push origin feature/new-feature`
6. Create Pull Request

### Code Standards
- **PEP 8**: Python style guide
- **Type Hints**: Use typing annotations
- **Documentation**: Docstrings for all functions
- **Testing**: Add tests for new features

---

## Support

For questions, issues, or contributions:
- **Create an Issue**: Use GitHub Issues for bugs and feature requests
- **Documentation**: Check inline code documentation
- **Community**: Join discussions in project Issues

---
