import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Force CPU usage to avoid CUDA errors
torch.cuda.is_available = lambda: False
device = torch.device("cpu")
print(f"Using device: {device}")

# Step 1: Load dataset
df = pd.read_csv(r"Final_Heteronym_Dataset.csv")

# Step 2: Prepare embeddings
try:
    embeddings = np.load('sentence_embeddings.npy')
    if len(embeddings) == len(df):
        X_all = embeddings
        print("Loaded pre-computed embeddings")
    else:
        raise FileNotFoundError("Embedding count doesn't match dataset size")
except FileNotFoundError:
    from sentence_transformers import SentenceTransformer
    print("Computing sentence embeddings...")
    model_bert = SentenceTransformer("all-MiniLM-L6-v2")
    X_all = model_bert.encode(df["Sentence"].tolist())
    np.save('sentence_embeddings.npy', X_all)
    print(f"Sentence embeddings shape: {X_all.shape}")

# Step 3: Encode labels
print("Encoding labels...")
# Check for problematic values in WordType
print(f"Unique values in WordType column: {df['WordType'].unique()}")
if 'WordType' in df['WordType'].unique():
    print("Found 'WordType' as a value in the WordType column. Fixing...")
    df.loc[df['WordType'] == 'WordType', 'WordType'] = 'non-heteronym'
    print(f"After fixing: {df['WordType'].unique()}")

# Encode labels
meaning_encoder = LabelEncoder()
y_meaning_all = meaning_encoder.fit_transform(df["Meaning"])

wordtype_encoder = LabelEncoder()
y_wordtype_all = wordtype_encoder.fit_transform(df["WordType"])

# Create pronunciation encodings
pronunciation_encoder = LabelEncoder()
pronunciations = df["Pronunciation"].fillna("unknown").astype(str)
y_pron_all = pronunciation_encoder.fit_transform(pronunciations)

print(f"Found {len(meaning_encoder.classes_)} meaning classes")
print(f"Found {len(wordtype_encoder.classes_)} word type classes: {wordtype_encoder.classes_}")
print(f"Found {len(pronunciation_encoder.classes_)} pronunciation classes")

# Step 4: Filter out rare classes
print("Filtering rare classes...")
meaning_counts = Counter(y_meaning_all)
valid_meanings = [label for label, count in meaning_counts.items() if count >= 2]
mask = np.isin(y_meaning_all, valid_meanings)
X = np.array(X_all)[mask]
y_meaning = np.array(y_meaning_all)[mask]
y_wordtype = np.array(y_wordtype_all)[mask]
y_pron = np.array(y_pron_all)[mask]
df_filtered = df[mask].reset_index(drop=True)

print(f"After filtering: {len(np.unique(y_meaning))} meaning classes")
print(f"Dataset size: {len(X)} samples")

# Step 5: Create word vectors for additional input
# Extract unique words for embedding
words = df_filtered["Word"].unique()
word_encoder = LabelEncoder()
word_indices = word_encoder.fit_transform(df_filtered["Word"])

# Step 6: Create data augmentation function
def augment_data(X, y_meaning, y_wordtype, y_pron, word_indices, augment_ratio=0.2):
    """
    Augment data by adding noise to embeddings
    """
    n_samples = int(len(X) * augment_ratio)

    # Select samples for augmentation, focusing on heteronyms
    heteronym_mask = (y_wordtype == wordtype_encoder.transform(['heteronym'])[0])
    heteronym_indices = np.where(heteronym_mask)[0]

    # If we have heteronyms, ensure they're well represented in augmentation
    if len(heteronym_indices) > 0:
        augment_indices = np.random.choice(heteronym_indices, size=min(n_samples, len(heteronym_indices)), replace=True)
        if len(augment_indices) < n_samples:
            # Add some non-heteronyms if needed
            nonheteronym_indices = np.where(~heteronym_mask)[0]
            additional_indices = np.random.choice(nonheteronym_indices, size=n_samples-len(augment_indices), replace=False)
            augment_indices = np.concatenate([augment_indices, additional_indices])
    else:
        # Fallback if no heteronyms
        augment_indices = np.random.choice(len(X), size=n_samples, replace=False)

    # Create augmented samples
    X_aug = X[augment_indices].copy()

    # Add noise to embeddings (5% random noise)
    noise = np.random.normal(0, 0.05, X_aug.shape)
    X_aug += noise

    # Keep the same labels
    y_meaning_aug = y_meaning[augment_indices]
    y_wordtype_aug = y_wordtype[augment_indices]
    y_pron_aug = y_pron[augment_indices]
    word_indices_aug = word_indices[augment_indices]

    # Combine with original data
    X_combined = np.vstack([X, X_aug])
    y_meaning_combined = np.concatenate([y_meaning, y_meaning_aug])
    y_wordtype_combined = np.concatenate([y_wordtype, y_wordtype_aug])
    y_pron_combined = np.concatenate([y_pron, y_pron_aug])
    word_indices_combined = np.concatenate([word_indices, word_indices_aug])

    return X_combined, y_meaning_combined, y_wordtype_combined, y_pron_combined, word_indices_combined

# Step 7: Define a simpler model with stronger regularization
class ImprovedHeteronymProcessor(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, dropout_rate=0.5,
                 meaning_output_dim=92, wordtype_output_dim=2, pron_output_dim=50,
                 word_embedding_dim=32, num_words=100):
        super(ImprovedHeteronymProcessor, self).__init__()

        # Word embedding layer (smaller dimension)
        self.word_embedding = nn.Embedding(num_words, word_embedding_dim)

        # Simpler semantic pathway with higher dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Combined features
        self.combined_layer = nn.Linear(hidden_dim + word_embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Task-specific heads (simplified)
        self.meaning_classifier = nn.Linear(hidden_dim, meaning_output_dim)
        self.wordtype_classifier = nn.Linear(hidden_dim, wordtype_output_dim)
        self.pronunciation_classifier = nn.Linear(hidden_dim, pron_output_dim)

        # Processing time simulation
        self.activation_time = nn.Linear(hidden_dim, 1)

    def forward(self, x, word_idx):
        # Get word embeddings
        word_emb = self.word_embedding(word_idx)

        # Process sentence context
        encoded = self.encoder(x)

        # Combine word and context features
        combined = torch.cat((encoded, word_emb), dim=1)
        hidden = self.dropout(torch.relu(self.combined_layer(combined)))

        # Generate outputs
        meaning_logits = self.meaning_classifier(hidden)
        wordtype_logits = self.wordtype_classifier(hidden)
        pron_logits = self.pronunciation_classifier(hidden)

        # Simulated processing time
        act_time = torch.sigmoid(self.activation_time(hidden))

        return meaning_logits, wordtype_logits, pron_logits, act_time

# Step 8: Cross-validation setup
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Step 9: Training and evaluation with cross-validation
cv_meaning_acc = []
cv_wordtype_acc = []
cv_pron_acc = []
cv_heteronym_acc = []
cv_nonheteronym_acc = []

final_test_idx = None
final_preds = None
final_act_times = None

print(f"Starting {n_folds}-fold cross-validation...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_wordtype)):
    print(f"\nFold {fold+1}/{n_folds}")

    # Split data for this fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_meaning_train, y_meaning_test = y_meaning[train_idx], y_meaning[test_idx]
    y_wordtype_train, y_wordtype_test = y_wordtype[train_idx], y_wordtype[test_idx]
    y_pron_train, y_pron_test = y_pron[train_idx], y_pron[test_idx]
    word_indices_train, word_indices_test = word_indices[train_idx], word_indices[test_idx]

    # Track the last fold's indices for final analysis
    if fold == n_folds - 1:
        final_test_idx = test_idx

    # Further split training data for validation
    X_train, X_val, y_meaning_train, y_meaning_val, y_wordtype_train, y_wordtype_val, \
    y_pron_train, y_pron_val, word_indices_train, word_indices_val = train_test_split(
        X_train, y_meaning_train, y_wordtype_train, y_pron_train, word_indices_train,
        test_size=0.15, stratify=y_wordtype_train, random_state=42
    )

    # Data augmentation on training set only
    X_train, y_meaning_train, y_wordtype_train, y_pron_train, word_indices_train = augment_data(
        X_train, y_meaning_train, y_wordtype_train, y_pron_train, word_indices_train, augment_ratio=0.3
    )

    # Remap labels to ensure contiguous indices
    train_meaning_classes = np.unique(y_meaning_train)
    meaning_mapping = {old_label: new_label for new_label, old_label in enumerate(train_meaning_classes)}

    y_meaning_train_remapped = np.array([meaning_mapping[label] for label in y_meaning_train])

    # Apply mapping to validation set
    val_valid_mask = np.isin(y_meaning_val, train_meaning_classes)
    X_val = X_val[val_valid_mask]
    y_meaning_val = y_meaning_val[val_valid_mask]
    y_wordtype_val = y_wordtype_val[val_valid_mask]
    y_pron_val = y_pron_val[val_valid_mask]
    word_indices_val = word_indices_val[val_valid_mask]
    y_meaning_val_remapped = np.array([meaning_mapping[label] for label in y_meaning_val])

    # Apply mapping to test set
    test_valid_mask = np.isin(y_meaning_test, train_meaning_classes)
    X_test = X_test[test_valid_mask]
    y_meaning_test = y_meaning_test[test_valid_mask]
    y_wordtype_test = y_wordtype_test[test_valid_mask]
    y_pron_test = y_pron_test[test_valid_mask]
    word_indices_test = word_indices_test[test_valid_mask]
    y_meaning_test_remapped = np.array([meaning_mapping[label] for label in y_meaning_test])

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_meaning_train_tensor = torch.tensor(y_meaning_train_remapped, dtype=torch.long)
    y_wordtype_train_tensor = torch.tensor(y_wordtype_train, dtype=torch.long)
    y_pron_train_tensor = torch.tensor(y_pron_train, dtype=torch.long)
    word_indices_train_tensor = torch.tensor(word_indices_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_meaning_val_tensor = torch.tensor(y_meaning_val_remapped, dtype=torch.long)
    y_wordtype_val_tensor = torch.tensor(y_wordtype_val, dtype=torch.long)
    y_pron_val_tensor = torch.tensor(y_pron_val, dtype=torch.long)
    word_indices_val_tensor = torch.tensor(word_indices_val, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_meaning_test_tensor = torch.tensor(y_meaning_test_remapped, dtype=torch.long)
    y_wordtype_test_tensor = torch.tensor(y_wordtype_test, dtype=torch.long)
    y_pron_test_tensor = torch.tensor(y_pron_test, dtype=torch.long)
    word_indices_test_tensor = torch.tensor(word_indices_test, dtype=torch.long)

    # Model parameters
    n_meaning_classes = len(train_meaning_classes)
    n_wordtype_classes = len(np.unique(y_wordtype_train))
    n_pron_classes = len(np.unique(y_pron_train))
    n_words = len(word_encoder.classes_)

    # Initialize model with regularization
    model = ImprovedHeteronymProcessor(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        dropout_rate=0.5,
        meaning_output_dim=n_meaning_classes,
        wordtype_output_dim=n_wordtype_classes,
        pron_output_dim=n_pron_classes,
        word_embedding_dim=32,
        num_words=n_words
    )

    # Loss functions with optional class weighting
    meaning_criterion = nn.CrossEntropyLoss()
    wordtype_criterion = nn.CrossEntropyLoss()
    pron_criterion = nn.CrossEntropyLoss()
    time_criterion = nn.MSELoss()

    # Optimizer with stronger regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler - removed verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training with early stopping
    epochs = 100
    batch_size = 32
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0

        # Generate synthetic time targets
        def get_time_targets(wordtype_batch):
            base_time = torch.ones(len(wordtype_batch), 1) * 0.5
            heteronym_mask = (wordtype_batch == 1).float().unsqueeze(1)
            time_targets = base_time + (heteronym_mask * 0.13)
            return time_targets

        # Mini-batch processing
        indices = np.random.permutation(len(X_train_tensor))
        for start_idx in range(0, len(X_train_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(X_train_tensor))
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train_tensor[batch_indices]
            y_meaning_batch = y_meaning_train_tensor[batch_indices]
            y_wordtype_batch = y_wordtype_train_tensor[batch_indices]
            y_pron_batch = y_pron_train_tensor[batch_indices]
            word_indices_batch = word_indices_train_tensor[batch_indices]

            optimizer.zero_grad()

            meaning_outputs, wordtype_outputs, pron_outputs, act_time = model(
                X_batch, word_indices_batch
            )

            meaning_loss = meaning_criterion(meaning_outputs, y_meaning_batch)
            wordtype_loss = wordtype_criterion(wordtype_outputs, y_wordtype_batch)
            pron_loss = pron_criterion(pron_outputs, y_pron_batch)

            time_targets = get_time_targets(y_wordtype_batch)
            time_loss = time_criterion(act_time, time_targets)

            # Weighted loss combination
            loss = meaning_loss + 0.2 * wordtype_loss + 0.2 * pron_loss + 0.1 * time_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        avg_train_loss = train_loss / batches

        # Validation
        model.eval()
        with torch.no_grad():
            meaning_outputs, wordtype_outputs, pron_outputs, _ = model(
                X_val_tensor, word_indices_val_tensor
            )

            meaning_loss = meaning_criterion(meaning_outputs, y_meaning_val_tensor)
            wordtype_loss = wordtype_criterion(wordtype_outputs, y_wordtype_val_tensor)
            pron_loss = pron_criterion(pron_outputs, y_pron_val_tensor)

            val_loss = meaning_loss + 0.2 * wordtype_loss + 0.2 * pron_loss

            # Calculate validation accuracy
            meaning_pred = torch.argmax(meaning_outputs, dim=1)
            wordtype_pred = torch.argmax(wordtype_outputs, dim=1)
            pron_pred = torch.argmax(pron_outputs, dim=1)

            meaning_acc = (meaning_pred == y_meaning_val_tensor).float().mean()
            wordtype_acc = (wordtype_pred == y_wordtype_val_tensor).float().mean()
            pron_acc = (pron_pred == y_pron_val_tensor).float().mean()

        # Update learning rate - print message manually since verbose parameter was removed
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Epoch {epoch+1}: Reducing learning rate from {old_lr} to {new_lr}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"Val Acc: Meaning = {meaning_acc:.4f}, WordType = {wordtype_acc:.4f}, Pron = {pron_acc:.4f}")

    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        meaning_outputs, wordtype_outputs, pron_outputs, act_time = model(
            X_test_tensor, word_indices_test_tensor
        )

        meaning_pred = torch.argmax(meaning_outputs, dim=1)
        wordtype_pred = torch.argmax(wordtype_outputs, dim=1)
        pron_pred = torch.argmax(pron_outputs, dim=1)

        meaning_acc = accuracy_score(y_meaning_test_tensor, meaning_pred)
        wordtype_acc = accuracy_score(y_wordtype_test_tensor, wordtype_pred)
        pron_acc = accuracy_score(y_pron_test_tensor, pron_pred)

        # Save final fold predictions
        if fold == n_folds - 1:
            final_preds = (meaning_pred.numpy(), wordtype_pred.numpy(), pron_pred.numpy())
            final_act_times = act_time.numpy()

        # Calculate accuracy for heteronyms vs non-heteronyms
        heteronym_mask = (y_wordtype_test == wordtype_encoder.transform(['heteronym'])[0])
        heteronym_idx = np.where(heteronym_mask)[0]
        nonheteronym_idx = np.where(~heteronym_mask)[0]

        heteronym_acc = 0
        if len(heteronym_idx) > 0:
            heteronym_acc = accuracy_score(
                y_meaning_test_tensor[heteronym_idx], meaning_pred[heteronym_idx]
            )

        nonheteronym_acc = 0
        if len(nonheteronym_idx) > 0:
            nonheteronym_acc = accuracy_score(
                y_meaning_test_tensor[nonheteronym_idx], meaning_pred[nonheteronym_idx]
            )

    # Store cross-validation results
    cv_meaning_acc.append(meaning_acc)
    cv_wordtype_acc.append(wordtype_acc)
    cv_pron_acc.append(pron_acc)
    cv_heteronym_acc.append(heteronym_acc)
    cv_nonheteronym_acc.append(nonheteronym_acc)

    print(f"Test Accuracy: Meaning = {meaning_acc:.4f}, WordType = {wordtype_acc:.4f}, Pron = {pron_acc:.4f}")
    print(f"Heteronym Acc = {heteronym_acc:.4f}, Non-heteronym Acc = {nonheteronym_acc:.4f}")

# Step 10: Analyze cross-validation results
print("\nCross-validation Results:")
print(f"Meaning Accuracy: {np.mean(cv_meaning_acc):.4f} ± {np.std(cv_meaning_acc):.4f}")
print(f"WordType Accuracy: {np.mean(cv_wordtype_acc):.4f} ± {np.std(cv_wordtype_acc):.4f}")
print(f"Pronunciation Accuracy: {np.mean(cv_pron_acc):.4f} ± {np.std(cv_pron_acc):.4f}")
print(f"Heteronym Accuracy: {np.mean(cv_heteronym_acc):.4f} ± {np.std(cv_heteronym_acc):.4f}")
print(f"Non-heteronym Accuracy: {np.mean(cv_nonheteronym_acc):.4f} ± {np.std(cv_nonheteronym_acc):.4f}")

# Step 11: Analyze final fold results for detailed heteronym analysis
if final_test_idx is not None and final_preds is not None:
    meaning_pred, wordtype_pred, pron_pred = final_preds

    # Get indices for heteronyms and non-heteronyms
    heteronym_mask = (y_wordtype[final_test_idx] == wordtype_encoder.transform(['heteronym'])[0])
    heteronym_idx = np.where(heteronym_mask)[0]
    nonheteronym_idx = np.where(~heteronym_mask)[0]

    y_meaning_test = y_meaning[final_test_idx][test_valid_mask]
    df_test = df_filtered.iloc[final_test_idx[test_valid_mask]]

    print(f"\nAnalyzing {len(heteronym_idx)} heteronyms in the final test set...")

    if len(heteronym_idx) > 0:
        heteronym_words = []
        heteronym_correct = []
        heteronym_times = []

        # Get details for each heteronym test instance
        for i in heteronym_idx:
            word = df_test.iloc[i]['Word']
            correct = meaning_pred[i] == y_meaning_test_remapped[i]
            proc_time = final_act_times[i][0]

            heteronym_words.append(word)
            heteronym_correct.append(correct)
            heteronym_times.append(proc_time)

        # Calculate per-heteronym performance
        heteronym_performance = {}
        for word, correct, proc_time in zip(heteronym_words, heteronym_correct, heteronym_times):
            if word not in heteronym_performance:
                heteronym_performance[word] = {'correct': 0, 'total': 0, 'time': 0}
            heteronym_performance[word]['total'] += 1
            heteronym_performance[word]['time'] += proc_time
            if correct:
                heteronym_performance[word]['correct'] += 1

        # Calculate accuracy and average processing time
        for word in heteronym_performance:
            if heteronym_performance[word]['total'] > 0:
                heteronym_performance[word]['accuracy'] = (
                    heteronym_performance[word]['correct'] / heteronym_performance[word]['total']
                )
                heteronym_performance[word]['avg_time'] = (
                    heteronym_performance[word]['time'] / heteronym_performance[word]['total']
                )
            else:
                heteronym_performance[word]['accuracy'] = 0
                heteronym_performance[word]['avg_time'] = 0

        # Sort by accuracy
        sorted_by_accuracy = sorted(
            heteronym_performance.items(),
            key=lambda x: x[1]['accuracy']
        )

        # Print the most difficult heteronyms
        if sorted_by_accuracy:
            print("Most challenging heteronyms (by accuracy):")
            for word, stats in sorted_by_accuracy[:5]:
                print(f"  {word}: {stats['accuracy']:.2f} accuracy, {stats['avg_time']:.4f} processing time")

            # Export detailed results to CSV
            heteronym_results = []
            for word, stats in heteronym_performance.items():
                heteronym_results.append({
                    'Word': word,
                    'Accuracy': stats['accuracy'],
                    'Processing_Time': stats['avg_time'],
                    'Sample_Count': stats['total']
                })

            result_df = pd.DataFrame(heteronym_results)
            result_df.to_csv('heteronym_results_improved.csv', index=False)
            print("\nDetailed heteronym analysis exported to 'heteronym_results_improved.csv'")

# Step 11b: Generate all four key visualizations
# 1. heteronym_processing_analysis.png - Overall performance metrics
def generate_overall_performance_viz():
    plt.figure(figsize=(15, 10))

    # Subplot 1: Accuracy comparison
    plt.subplot(2, 2, 1)
    metrics = ['Overall', 'Heteronym', 'Non-heteronym']
    accuracies = [np.mean(cv_meaning_acc), np.mean(cv_heteronym_acc), np.mean(cv_nonheteronym_acc)]
    errors = [np.std(cv_meaning_acc), np.std(cv_heteronym_acc), np.std(cv_nonheteronym_acc)]

    plt.bar(metrics, accuracies, yerr=errors, alpha=0.7, capsize=7)
    plt.ylabel('Accuracy')
    plt.title('Disambiguation Accuracy by Word Type')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Confidence comparison
    plt.subplot(2, 2, 2)
    if len(heteronym_idx) > 0 and len(nonheteronym_idx) > 0:
        # Calculate confidence from softmax outputs
        meaning_probs = torch.softmax(meaning_outputs, dim=1).detach().numpy()
        het_conf = meaning_probs[heteronym_idx].max(axis=1).mean()
        nonhet_conf = meaning_probs[nonheteronym_idx].max(axis=1).mean()

        confidence = [het_conf, nonhet_conf]
        plt.bar(['Heteronym', 'Non-heteronym'], confidence, alpha=0.7)
        plt.ylabel('Mean Confidence')
        plt.title('Model Confidence in Disambiguation')
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Processing time comparison
    plt.subplot(2, 2, 3)
    if len(heteronym_idx) > 0 and len(nonheteronym_idx) > 0:
        het_time = final_act_times[heteronym_idx].mean()
        nonhet_time = final_act_times[nonheteronym_idx].mean()
        proc_times = [het_time, nonhet_time]

        plt.bar(['Heteronym', 'Non-heteronym'], proc_times, alpha=0.7)
        plt.ylabel('Simulated Processing Time')
        plt.title('Processing Time Comparison')
        plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 4: Confusion matrix for word type
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(y_wordtype_test, wordtype_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-heteronym', 'Heteronym'],
                yticklabels=['Non-heteronym', 'Heteronym'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Word Type Classification')

    plt.tight_layout()
    plt.savefig('heteronym_processing_analysis.png')
    plt.close()
    print("Overall performance metrics visualization saved to 'heteronym_processing_analysis.png'")

# 2. activation_curves.png - Simulated meaning activation over time
def generate_activation_curves():
    plt.figure(figsize=(10, 7))

    # Create a simulation of meaning activation over time
    time_points = np.linspace(0, 500, 100)  # 0-500ms

    # Simulate different activation curves based on research hypotheses
    def activation_curve(time_points, word_type, dominant=True, context='neutral'):
        # Parameters based on heteronym processing theory
        if word_type == 'heteronym':
            # Base parameters for heteronyms
            delay = 0.3 if dominant else 0.5
            slope = 0.012 if dominant else 0.008

            # Context effects
            if context == 'strong':
                delay *= 0.8  # Strong context speeds activation
                slope *= 1.2
            elif context == 'ambiguous':
                delay *= 1.2  # Ambiguous context slows activation
                slope *= 0.8
        else:  # non-heteronym
            # Base parameters for non-heteronyms (faster activation)
            delay = 0.2 if dominant else 0.35
            slope = 0.015 if dominant else 0.01

            # Context effects (less pronounced for non-heteronyms)
            if context == 'strong':
                delay *= 0.9
                slope *= 1.1
            elif context == 'ambiguous':
                delay *= 1.1
                slope *= 0.9

        # Calculate activation using sigmoid function
        activation = 1 / (1 + np.exp(-slope * (time_points - delay * 1000)))
        return activation

    # Generate curves for different conditions
    het_dom_strong = activation_curve(time_points, 'heteronym', True, 'strong')
    het_dom_neutral = activation_curve(time_points, 'heteronym', True, 'neutral')
    het_sub_neutral = activation_curve(time_points, 'heteronym', False, 'neutral')
    het_sub_ambig = activation_curve(time_points, 'heteronym', False, 'ambiguous')

    nonhet_dom = activation_curve(time_points, 'non-heteronym', True, 'neutral')
    nonhet_sub = activation_curve(time_points, 'non-heteronym', False, 'neutral')

    # Plot the curves
    plt.plot(time_points, het_dom_strong, 'b-', linewidth=2,
             label='Heteronym - Dominant Meaning (Strong Context)')
    plt.plot(time_points, het_dom_neutral, 'b--',
             label='Heteronym - Dominant Meaning (Neutral Context)')
    plt.plot(time_points, het_sub_neutral, 'b:',
             label='Heteronym - Subordinate Meaning (Neutral Context)')
    plt.plot(time_points, het_sub_ambig, 'b-.',
             label='Heteronym - Subordinate Meaning (Ambiguous Context)')

    plt.plot(time_points, nonhet_dom, 'r-', linewidth=2,
             label='Non-heteronym - Dominant Meaning')
    plt.plot(time_points, nonhet_sub, 'r--',
             label='Non-heteronym - Subordinate Meaning')

    # Add reference lines
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=250, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel('Processing Time (ms)')
    plt.ylabel('Meaning Activation')
    plt.title('Simulated Meaning Activation Over Time')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('activation_curves.png')
    plt.close()
    print("Activation curves visualization saved to 'activation_curves.png'")

# 3. context_effects.png - Effect of context strength on disambiguation
def generate_context_effects():
    plt.figure(figsize=(12, 7))

    # Estimate context strength for test samples
    def estimate_context_strength(embeddings, word_indices, heteronym_mask):
        # Using embedding coherence as a proxy for context strength
        embedding_norm = np.linalg.norm(embeddings, axis=1)
        normalized_embeddings = embeddings / embedding_norm[:, np.newaxis]

        # Calculate cosine similarity to estimate context strength
        context_strength = np.zeros(len(embeddings))

        for word_idx in np.unique(word_indices):
            word_mask = (word_indices == word_idx)
            if np.sum(word_mask) > 1:
                word_vectors = normalized_embeddings[word_mask]
                avg_vector = np.mean(word_vectors, axis=0)
                avg_vector = avg_vector / np.linalg.norm(avg_vector)

                similarities = np.dot(word_vectors, avg_vector)
                context_strength[word_mask] = similarities

        # Categorize context strength
        strong = context_strength > 0.8
        neutral = (context_strength > 0.6) & (context_strength <= 0.8)
        ambiguous = context_strength <= 0.6

        return context_strength, strong, neutral, ambiguous

    # Calculate context strength and categorization
    X_test_np = X_test_tensor.numpy()
    heteronym_mask = (y_wordtype_test == 1)
    context_strength, strong_ctx, neutral_ctx, ambiguous_ctx = estimate_context_strength(
        X_test_np, word_indices_test, heteronym_mask
    )

    # Calculate accuracy by context strength for heteronyms and non-heteronyms
    def accuracy_by_context(y_true, y_pred, wordtype_mask, context_mask):
        if np.sum(wordtype_mask & context_mask) > 0:
            return accuracy_score(y_true[wordtype_mask & context_mask],
                                y_pred[wordtype_mask & context_mask])
        return 0

    y_true = y_meaning_test_tensor.numpy()
    y_pred = meaning_pred.numpy()

    # Heteronym accuracy by context
    het_strong_acc = accuracy_by_context(y_true, y_pred, heteronym_mask, strong_ctx)
    het_neutral_acc = accuracy_by_context(y_true, y_pred, heteronym_mask, neutral_ctx)
    het_ambiguous_acc = accuracy_by_context(y_true, y_pred, heteronym_mask, ambiguous_ctx)

    # Non-heteronym accuracy by context
    nonhet_mask = ~heteronym_mask
    nonhet_strong_acc = accuracy_by_context(y_true, y_pred, nonhet_mask, strong_ctx)
    nonhet_neutral_acc = accuracy_by_context(y_true, y_pred, nonhet_mask, neutral_ctx)
    nonhet_ambiguous_acc = accuracy_by_context(y_true, y_pred, nonhet_mask, ambiguous_ctx)

    # Plot context effects
    context_types = ['Strong', 'Neutral', 'Ambiguous']
    heteronym_ctx_acc = [het_strong_acc, het_neutral_acc, het_ambiguous_acc]
    nonheteronym_ctx_acc = [nonhet_strong_acc, nonhet_neutral_acc, nonhet_ambiguous_acc]

    x = np.arange(len(context_types))
    width = 0.35

    # Main plot - Accuracy by context strength
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, heteronym_ctx_acc, width, label='Heteronym')
    plt.bar(x + width/2, nonheteronym_ctx_acc, width, label='Non-heteronym')
    plt.xlabel('Context Strength')
    plt.ylabel('Disambiguation Accuracy')
    plt.title('Effect of Context Strength on Disambiguation')
    plt.xticks(x, context_types)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add theoretical processing time by context
    plt.subplot(1, 2, 2)
    # Theoretical processing times (based on research hypotheses)
    # Higher values = more processing time needed
    het_times = [0.6, 0.8, 1.0]  # Strong context reduces processing time
    nonhet_times = [0.5, 0.6, 0.7]  # Less effect on non-heteronyms

    plt.bar(x - width/2, het_times, width, label='Heteronym')
    plt.bar(x + width/2, nonhet_times, width, label='Non-heteronym')
    plt.xlabel('Context Strength')
    plt.ylabel('Relative Processing Demand')
    plt.title('Theoretical Processing Cost by Context')
    plt.xticks(x, context_types)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('context_effects.png')
    plt.close()
    print("Context effects visualization saved to 'context_effects.png'")

# 4. heteronym_difficulty.png - Individual heteronym performance analysis
def generate_heteronym_difficulty():
    plt.figure(figsize=(12, 8))

    # Check if we have heteronym data to analyze
    if len(heteronym_idx) > 0 and 'heteronym_performance' in locals():
        # Plot 1: Accuracy vs Processing Time scatter plot
        plt.subplot(2, 1, 1)
        accuracies = [stats['accuracy'] for _, stats in heteronym_performance.items()]
        times = [stats['avg_time'] for _, stats in heteronym_performance.items()]
        words = [word for word, _ in heteronym_performance.items()]

        plt.scatter(times, accuracies, alpha=0.7)

        # Add word labels to points
        for i, word in enumerate(words):
            plt.annotate(word, (times[i], accuracies[i]), fontsize=8)

        plt.xlabel('Processing Time (normalized)')
        plt.ylabel('Accuracy')
        plt.title('Heteronym Difficulty: Accuracy vs Processing Time')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Bar chart of most difficult heteronyms
        plt.subplot(2, 1, 2)
        sorted_by_acc = sorted(heteronym_performance.items(), key=lambda x: x[1]['accuracy'])
        difficult_words = [word for word, _ in sorted_by_acc[:10]]
        difficulties = [1 - stats['accuracy'] for _, stats in sorted_by_acc[:10]]

        y_pos = np.arange(len(difficult_words))
        plt.barh(y_pos, difficulties)
        plt.yticks(y_pos, difficult_words)
        plt.xlabel('Error Rate (1 - Accuracy)')
        plt.title('Top 10 Most Difficult Heteronyms')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        # If no heteronym data available, show a placeholder
        plt.text(0.5, 0.5, "No heteronym performance data available",
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig('heteronym_difficulty.png')
    plt.close()
    print("Heteronym difficulty visualization saved to 'heteronym_difficulty.png'")

# Generate visualizations if we have data from the final fold
if final_test_idx is not None and final_preds is not None:
    try:
        generate_overall_performance_viz()
        generate_activation_curves()
        generate_context_effects()
        generate_heteronym_difficulty()
    except Exception as e:
        print(f"Error generating visualizations: {e}")

print("\nTraining and evaluation complete!")
print("All results and visualizations have been saved to disk.")

