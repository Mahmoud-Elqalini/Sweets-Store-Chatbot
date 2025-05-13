import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
from main_functions import (load_json_file, preprocess, bag_of_words, NeuralNet, get_arabert_embedding, ChatDataset,
                            pad_or_truncate)

# Load data by using the function
english_intents = load_json_file('file_training/english.json', 'English training data')
arabic_intents = load_json_file('file_training/arabic.json', 'Arabic training data')

# Merge data
intents = {'intents': english_intents['intents'] + arabic_intents['intents']}
print('Merge data successfully')

# Initialize augmenters
synonym_aug_en = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
random_swap_aug_en = naw.RandomWordAug(action="swap", aug_p=0.1)


# Data Augmentation function
def augment_patterns(patterns, num_augmentations=2, language='en'):
    augmented_patterns = patterns.copy()
    if language == 'en':
        for pattern in patterns:
            for _ in range(num_augmentations):
                aug_pattern = pattern
                try:
                    aug_result = synonym_aug_en.augment(aug_pattern)
                    aug_pattern = aug_result[0] if aug_result else aug_pattern
                except Exception as e:
                    print(f"Error in SynonymAug: {e}. Using original: {aug_pattern}")
                try:
                    aug_result = random_swap_aug_en.augment(aug_pattern)
                    aug_pattern = aug_result[0] if aug_result else aug_pattern
                except Exception as e:
                    print(f"Error in RandomSwapAug: {e}. Using original: {aug_pattern}")
                augmented_patterns.append(aug_pattern)
    else:
        for pattern in patterns:
            for _ in range(num_augmentations):
                words = pattern.split()
                if len(words) > 1:
                    idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                    words[idx1], words[idx2] = words[idx2], words[idx1]
                    augmented_patterns.append(' '.join(words))
                else:
                    augmented_patterns.append(pattern)
    return augmented_patterns


# Apply Data Augmentation to the patterns
augmented_intents = {'intents': []}
for intent in intents['intents']:
    if 'language' not in intent:
        print(f"Warning: Intent {intent.get('tag', 'unknown')} is missing 'language' key. Skipping.")
        continue
    tag = intent['tag']
    patterns = intent['patterns']
    language = intent['language']
    augmented_patterns = augment_patterns(patterns, num_augmentations=2, language=language)
    augmented_intents['intents'].append({
        'tag': tag,
        'patterns': augmented_patterns,
        'responses': intent['responses'],
        'language': language
    })
intents = augmented_intents

# Prepare training data
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    language = intent['language']
    for pattern in intent['patterns']:
        w = preprocess(pattern)
        if language == 'en':
            all_words.extend(w)
        xy.append((pattern, w, tag, language))

ignore_words = ['?', '.', '!', '؟']
all_words = [w for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words for English:", len(all_words))

# Determine input size (AraBERT embedding size is 768 for bert-base-arabertv2)
input_size = max(len(all_words), 768)
print(f"Input size: {input_size}")

# Prepare features
X = []
y = []
for (raw_pattern, pattern_sentence, tag, language) in xy:
    if language == 'en':
        bag = bag_of_words(pattern_sentence, all_words)
        bag = pad_or_truncate(bag, input_size)
        X.append(bag)
    else:
        embedding = get_arabert_embedding(raw_pattern)
        embedding = pad_or_truncate(embedding, input_size)
        X.append(embedding)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

hidden_size = 32
output_size = len(tags)
print(input_size, output_size)

# Create datasets
train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=4,
                        shuffle=False,
                        num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")
torch.cuda.empty_cache()

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

best_val_loss = float('inf')
patience = 50
counter = 0

for epoch in range(300):
    model.train()
    train_correct = 0
    train_total = 0
    for (words, labels) in train_loader:
        words = words.to(device, dtype=torch.float)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for (words, labels) in val_loader:
            words = words.to(device, dtype=torch.float)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            val_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/300], Train Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.2f}%,'
              f' Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save({
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "all_words": all_words,
            "tags": tags
        }, "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

print(f'final train loss: {loss.item():.4f}, final val loss: {val_loss:.4f}')

# Generate predictions for validation set
model.eval()
val_predictions = []
val_labels = []
with torch.no_grad():
    for (words, labels) in val_loader:
        words = words.to(device, dtype=torch.float)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        _, predicted = torch.max(outputs.data, 1)
        val_predictions.extend(predicted.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(val_labels, val_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=tags, yticklabels=tags, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Classification Report
class_report = classification_report(val_labels, val_predictions, target_names=tags, digits=4)
print("\nClassification Report:")
print(class_report)

# Save Classification Report
with open('classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("Classification Report:\n")
    f.write(class_report)

# Save the final model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
torch.save(data, "data.pth")
print(f'Training complete. File saved to data.pth')
