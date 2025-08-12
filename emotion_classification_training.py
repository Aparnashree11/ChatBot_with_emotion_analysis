import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
from datetime import datetime
import json
import logging
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    set_seed
)
from datasets import load_dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_seed(42)

@dataclass
class TrainingConfig:
    """Fast training configuration"""
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "./distilbert-goemotions-10k"
    max_length: int = 64
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 2
    max_train_samples: int = 40000
    max_eval_samples: int = 5000

class GoEmotionsProcessor:
    """Processes GoEmotions dataset"""
    
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.num_labels = len(self.EMOTION_LABELS)
        self.id2label = {i: label for i, label in enumerate(self.EMOTION_LABELS)}
        self.label2id = {label: i for i, label in enumerate(self.EMOTION_LABELS)}
    
    def load_data(self) -> DatasetDict:
        """Load exactly 10K samples"""
        logger.info("Loading GoEmotions dataset...")
        
        dataset = load_dataset("go_emotions", "simplified")
        
        # Limit to exactly what we need
        limited_dataset = DatasetDict({
            'train': dataset['train'].select(range(self.config.max_train_samples)),
            'validation': dataset['validation'].select(range(self.config.max_eval_samples)),
            'test': dataset['test'].select(range(500))
        })
        
        logger.info(f"Loaded {len(limited_dataset['train'])} train samples")
        return limited_dataset

class FastDataCollator:
    """Fast data collator for proper tensor types"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        batch = {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.float)
        }
        return batch

class FastDistilBERTTrainer:
    """Streamlined DistilBERT trainer"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.processor = GoEmotionsProcessor(config)
        
    def train(self):
        """Fast training pipeline"""
        logger.info("Starting fast DistilBERT training...")
        
        # Load data
        dataset = self.processor.load_data()
        
        # Initialize model and tokenizer
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.processor.num_labels,
            id2label=self.processor.id2label,
            label2id=self.processor.label2id,
            problem_type="multi_label_classification"
        )
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_dataset = self.preprocess_data(dataset, tokenizer)
        
        # Setup training
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            eval_steps=150,
            save_steps=300,
            evaluation_strategy="steps",
            save_strategy="steps",
            metric_for_best_model="eval_f1_macro",
            load_best_model_at_end=False,  # Skip for speed
            report_to=None,
            seed=42,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            save_total_limit=1,
            remove_unused_columns=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=processed_dataset['validation'],
            tokenizer=tokenizer,
            data_collator=FastDataCollator(tokenizer),
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        logger.info("Training started...")
        start_time = datetime.now()
        
        train_result = trainer.train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        # Quick evaluation
        test_results = trainer.evaluate(processed_dataset['test'])
        
        logger.info(f"Training completed in: {duration}")
        logger.info(f"F1-macro: {test_results.get('eval_f1_macro', 0):.4f}")
        
        return self.config.output_dir
    
    def preprocess_data(self, dataset, tokenizer):
        """Fast preprocessing"""
        def tokenize_function(examples):
            encoding = tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
            )
            
            # Convert labels to float vectors
            labels = []
            for label_list in examples['labels']:
                label_vector = np.zeros(self.processor.num_labels, dtype=np.float32)
                for label_idx in label_list:
                    if 0 <= label_idx < self.processor.num_labels:
                        label_vector[label_idx] = 1.0
                labels.append(label_vector.tolist())
            
            encoding['labels'] = labels
            return encoding
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
    
    def compute_metrics(self, eval_pred):
        """Fast metrics computation"""
        predictions, labels = eval_pred
        
        # Convert to binary predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = (probs > 0.5).int().numpy()
        y_true = np.array(labels).astype(int)
        
        # Calculate main metrics
        _, _, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
        }

class EmotionPredictor:
    """Fast inference for trained model"""
    
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def predict_top_k(self, text: str, k: int = 3):
        """Predict top-k emotions"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).squeeze()
        
        top_k_indices = torch.topk(probabilities, k).indices
        top_k_scores = torch.topk(probabilities, k).values
        
        return [(self.emotions[idx], round(score.item(), 3)) for idx, score in zip(top_k_indices, top_k_scores)]

def main():
    """Train DistilBERT on 10K samples in <3 hours"""
    
    print("Fast DistilBERT GoEmotions Training")
    
    # Fixed configuration for 10K samples, <3 hours
    config = TrainingConfig()
    
    # Auto-optimize for hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Optimize batch size for GPU
        if gpu_memory >= 8:
            config.batch_size = 64
        elif gpu_memory >= 4:
            config.batch_size = 32
        else:
            config.batch_size = 16
    else:
        print("CPU training - reducing settings")
        config.batch_size = 8
        config.max_train_samples = 40000  # Less data for CPU
    
    print(f"Training config:")
    print(f"  - Samples: {config.max_train_samples}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max length: {config.max_length}")
    
    # Estimate time
    if torch.cuda.is_available():
        steps_per_epoch = config.max_train_samples // config.batch_size
        estimated_minutes = (steps_per_epoch * config.num_epochs) * 0.5  # 0.5 min per step estimate
    else:
        steps_per_epoch = config.max_train_samples // config.batch_size
        estimated_minutes = (steps_per_epoch * config.num_epochs) * 2    # 2 min per step for CPU
    
    print(f"⏱️  Estimated time: {estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")
    
    # Start training
    try:
        start_time = datetime.now()
        print(f"\nTraining started at {start_time.strftime('%H:%M:%S')}")
        
        trainer = FastDistilBERTTrainer(config)
        model_path = trainer.train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nTraining completed!")
        print(f"Duration: {duration}")
        print(f"Model saved to: {model_path}")
        
        # Quick test
        print("\nTesting model...")
        predictor = EmotionPredictor(model_path)
        
        test_texts = [
            "I'm so excited!",
            "This is terrible.",
            "Thank you so much!",
            "I'm confused."
        ]
        
        for text in test_texts:
            emotions = predictor.predict_top_k(text, k=2)
            print(f"'{text}' -> {emotions}")
        
        print(f"\nReady to use! Load with: EmotionPredictor('{model_path}')")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()