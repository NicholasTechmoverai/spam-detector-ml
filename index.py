import torch
import torch.nn as nn
import pandas as pd
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from functools import wraps

def clear_console(func):
    @wraps(func) 
    def wrapper(*args, **kwargs):
        os.system('cls' if os.name == 'nt' else 'clear')  
        return func(*args, **kwargs) 
    return wrapper

class SpamDetectModel(nn.Module):
    """Model architecture must match the trained model"""
    def __init__(self, input_dim):
        super().__init__()
        # Match the architecture from training
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def load_model_and_resources():
    """Load model, vectorizer, and configuration"""
    try:
        # Load configuration
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        print("📋 Model Configuration:")
        print(f"   Model Class: {config.get('model_class', 'SpamDetectModel')}")
        print(f"   Vectorizer: {config.get('vectorizer_type', 'CountVectorizer')}")
        print(f"   Optimal Threshold: {config.get('optimal_threshold', 0.5):.3f}")
        
        # Load vectorizer
        vectorizer = joblib.load("vectorizer.pkl")
        input_dim = vectorizer.get_feature_names_out().shape[0]
        
        # Initialize and load model
        model = SpamDetectModel(input_dim=input_dim)
        model.load_state_dict(torch.load("spam-model.pt", map_location=torch.device('cpu')))
        model.eval()
        
        print(f"✅ Resources loaded successfully!")
        print(f"📊 Input dimension: {input_dim}")
        
        return model, vectorizer, config
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have these files in your directory:")
        print("  1. spam-model.pt - trained model weights")
        print("  2. vectorizer.pkl - text vectorizer")
        print("  3. model_config.json - model configuration")
        return None, None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None, None

def analyze_text_features(text, vectorizer):
    """Debug function to show what features are being extracted"""
    text_vectorized = vectorizer.transform([text]).toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    # Get non-zero features
    non_zero_indices = np.where(text_vectorized[0] > 0)[0]
    
    print("\n🔍 Text Analysis:")
    print(f"   Total vocabulary size: {len(feature_names)}")
    print(f"   Words found in vocabulary: {len(non_zero_indices)}")
    
    if len(non_zero_indices) > 0:
        print("\n   Extracted features:")
        for idx in non_zero_indices[:10]:  # Show first 10
            print(f"   - '{feature_names[idx]}': {text_vectorized[0][idx]:.4f}")
        if len(non_zero_indices) > 10:
            print(f"   ... and {len(non_zero_indices) - 10} more")
    
    return text_vectorized

def predict_single_text(model, vectorizer, text, threshold=0.5, debug=False):
    """Predict if a single text is spam"""
    if debug:
        text_vectorized = analyze_text_features(text, vectorizer)
    else:
        text_vectorized = vectorizer.transform([text]).toarray()
    
    text_tensor = torch.tensor(text_vectorized, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        probability = model(text_tensor).item()
        prediction = 1 if probability > threshold else 0
    
    # Calculate confidence level
    confidence = abs(probability - 0.5) * 2  # How far from decision boundary
    
    return {
        "text": text,
        "probability": probability,
        "prediction": "SPAM" if prediction == 1 else "HAM",
        "is_spam": prediction == 1,
        "confidence": confidence,
        "threshold_used": threshold
    }

def predict_multiple_texts(model, vectorizer, texts, threshold=0.5, batch_size=32):
    """Predict multiple texts efficiently"""
    all_results = []
    
    # Process in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Vectorize batch
        texts_vectorized = vectorizer.transform(batch_texts).toarray()
        texts_tensor = torch.tensor(texts_vectorized, dtype=torch.float32)
        
        # Predict batch
        with torch.no_grad():
            probabilities = model(texts_tensor).squeeze().numpy()
            predictions = (probabilities > threshold).astype(int)
        
        # Create results for batch
        for text, prob, pred in zip(batch_texts, probabilities, predictions):
            confidence = abs(prob - 0.5) * 2
            all_results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "probability": float(prob),
                "prediction": "SPAM" if pred == 1 else "HAM",
                "is_spam": pred == 1,
                "confidence": confidence
            })
    
    return all_results

def print_prediction(result, show_confidence=True):
    """Print prediction result in a formatted way"""
    if result['is_spam']:
        color = "\033[91m"  # Red for spam
        label = "🚫 SPAM"
    else:
        color = "\033[92m"  # Green for ham
        label = "✅ HAM"
    
    reset = "\033[0m"
    
    # Visual indicator of probability
    bar_length = 20
    filled = int(result['probability'] * bar_length)
    probability_bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\n{color}══════════════════════════════════════════════════════════════════{reset}")
    print(f"{color}📝 Text:{reset}")
    print(f"   {result['text']}")
    print(f"{color}📊 Analysis:{reset}")
    print(f"   Probability: {result['probability']:.4f}")
    print(f"   [{probability_bar}]")
    print(f"   Prediction: {label}")
    # print(f"   Threshold: >{result['threshold_used']:.3f}")
    
    if show_confidence:
        if result['confidence'] > 0.7:
            confidence_text = "High"
        elif result['confidence'] > 0.4:
            confidence_text = "Medium"
        else:
            confidence_text = "Low"
        print(f"   Confidence: {confidence_text} ({result['confidence']:.2f})")
    
    # Warning for borderline cases
    if 0.45 <= result['probability'] <= 0.55:
        print(f"{color}⚠️  Note:{reset} This is a borderline case")
    
    print(f"{color}══════════════════════════════════════════════════════════════════{reset}")

def interactive_mode(model, vectorizer, config):
    """Interactive mode for typing texts"""
    threshold = config.get('optimal_threshold', 0.5)
    
    print(f"\n🔍 Interactive Spam Detector (Threshold: {threshold:.3f})")
    print("Commands: 'debug' to show analysis, 'threshold X' to change, 'quit' to exit")
    print("=" * 60)
    
    current_threshold = threshold
    debug_mode = False
    
    while True:
        user_input = input("\n📝 Enter text to check: ").strip()
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'debug':
            debug_mode = not debug_mode
            status = "ON" if debug_mode else "OFF"
            print(f"🔧 Debug mode: {status}")
            continue
        elif user_input.lower().startswith('threshold'):
            try:
                new_threshold = float(user_input.split()[1])
                if 0 <= new_threshold <= 1:
                    current_threshold = new_threshold
                    print(f"⚖️  Threshold changed to: {current_threshold:.3f}")
                else:
                    print("❌ Threshold must be between 0 and 1")
            except:
                print("❌ Use: threshold 0.4 (or any number between 0-1)")
            continue
        elif not user_input:
            print("⚠️  Please enter some text")
            continue
        
        # Make prediction
        result = predict_single_text(
            model, vectorizer, user_input, 
            threshold=current_threshold,
            debug=debug_mode
        )
        print_prediction(result, show_confidence=True)

def batch_example_test(model, vectorizer, config):
    """Test with example texts"""
    threshold = config.get('optimal_threshold', 0.5)
    
    example_texts = [
        "WINNER!! You have won a $1,000 Walmart gift card. Call now 555-1234",
        "Hey, are we still meeting for lunch tomorrow at 1 PM?",
        "URGENT: Your bank account has been compromised. Click link to secure: http://bank-secure.com",
        "Mom, can you pick up some milk on your way home?",
        "Congratulations! You've been selected for a free iPhone 15. Claim now: http://free-iphone.com",
        "Meeting rescheduled to 3 PM. Please confirm attendance.",
        "You have 1 new voicemail. Call 555-9876 to listen.",
        "FREE MONEY! No fees, just call 800-123-4567 to claim",
        "Can you grab coffee this afternoon?",
        "Your package delivery failed. Update address: http://track-package.com"
    ]
    
    print(f"\n🧪 Testing {len(example_texts)} example texts...")
    print(f"   Using threshold: {threshold:.3f}")
    
    results = predict_multiple_texts(model, vectorizer, example_texts, threshold=threshold)
    
    # Summary statistics
    spam_count = sum(1 for r in results if r['is_spam'])
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"\n📊 Batch Results Summary:")
    print(f"   Spam: {spam_count} texts")
    print(f"   Ham: {len(results) - spam_count} texts")
    print(f"   Average confidence: {avg_confidence:.2f}")
    
    # Display each result
    for i, result in enumerate(results, 1):
        print(f"\n[{i}/{len(results)}]", end="")
        print_prediction(result, show_confidence=False)

def predict_from_csv(model, vectorizer, config):
    """Predict spam for texts in a CSV file"""
    csv_path = input("Enter CSV file path: ").strip()
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"\n📄 CSV Loaded:")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Select text column
        text_column = input("Enter text column name (default: 'text'): ").strip()
        if not text_column:
            text_column = "text" if "text" in df.columns else df.columns[0]
        
        if text_column not in df.columns:
            print(f"❌ Column '{text_column}' not found")
            print(f"   Available: {list(df.columns)}")
            return
        
        # Get threshold
        default_threshold = config.get('optimal_threshold', 0.5)
        threshold_input = input(f"Enter spam threshold (default: {default_threshold:.3f}): ").strip()
        threshold = float(threshold_input) if threshold_input else default_threshold
        
        # Get texts
        texts = df[text_column].astype(str).tolist()
        
        print(f"\n🔍 Processing {len(texts)} texts...")
        results = predict_multiple_texts(model, vectorizer, texts, threshold=threshold)
        
        # Add predictions to dataframe
        results_df = pd.DataFrame([{
            'prediction': r['prediction'],
            'probability': r['probability'],
            'is_spam': r['is_spam'],
            'confidence': r['confidence']
        } for r in results])
        
        df_with_predictions = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Save results
        output_path = csv_path.replace('.csv', '_predictions.csv')
        df_with_predictions.to_csv(output_path, index=False)
        
        print(f"\n✅ Predictions saved to: {output_path}")
        
        # Detailed summary
        spam_count = sum(1 for r in results if r['is_spam'])
        spam_indices = [i for i, r in enumerate(results) if r['is_spam']]
        
        print(f"\n📊 Detailed Summary:")
        print(f"   Total texts: {len(texts)}")
        print(f"   Spam detected: {spam_count} ({spam_count/len(texts)*100:.1f}%)")
        print(f"   Ham detected: {len(texts) - spam_count} ({(len(texts)-spam_count)/len(texts)*100:.1f}%)")
        
        if spam_count > 0:
            print(f"\n🚫 Spam examples (first 3):")
            for i, idx in enumerate(spam_indices[:3]):
                print(f"   {i+1}. {texts[idx][:80]}...")
        
        # Show some example predictions
        print(f"\n🔍 Sample predictions:")
        for i in range(min(3, len(results))):
            print(f"   {i+1}. {results[i]['prediction']}: {texts[i][:60]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

@clear_console
def main():
    """Main function"""
    print("=" * 60)
    print("🤖 ENHANCED SMS SPAM DETECTOR")
    print("=" * 60)
    
    # Load model and resources
    model, vectorizer, config = load_model_and_resources()
    if model is None:
        return
    
    print("\n📋 Available modes:")
    print("  1. 🔍 Interactive mode (type and analyze texts)")
    print("  2. 🧪 Batch test (pre-defined examples)")
    print("  3. 📄 CSV batch prediction")
    print("  4. ⚙️  View model details")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == "1":
        interactive_mode(model, vectorizer, config)
    
    elif choice == "2":
        batch_example_test(model, vectorizer, config)
    
    elif choice == "3":
        predict_from_csv(model, vectorizer, config)
    
    elif choice == "4":
        print("\n📊 Model Details:")
        print(f"   Input dimension: {vectorizer.get_feature_names_out().shape[0]}")
        print(f"   Optimal threshold: {config.get('optimal_threshold', 0.5):.3f}")
        print(f"   Vocabulary sample (first 10):")
        features = vectorizer.get_feature_names_out()[:10]
        for feat in features:
            print(f"     - {feat}")
        if vectorizer.get_feature_names_out().shape[0] > 10:
            print(f"     ... and {vectorizer.get_feature_names_out().shape[0] - 10} more")
    
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    main()