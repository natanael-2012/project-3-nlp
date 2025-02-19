# %%
# Load dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/TRAINING_DATA.txt', sep='\t', header=None)
df.columns = ['label', 'text']

# ⚡ Verify label mapping
print("Label distribution:\n", df['label'].value_counts())


# %%
# ⚡ Simplified preprocessing - remove clean_text step temporarily
# from nlp_utils import clean_text
# df['cleaned_text'] = df['text'].apply(clean_text)
df['cleaned_text'] = df['text']  # Bypass cleaning for debugging

# %%
# Split data properly
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# %%
#  ⚡ Convert to HuggingFace Dataset
%pip install datasets
from datasets import Dataset
train_ds = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
val_ds = Dataset.from_pandas(val_df[['cleaned_text', 'label']])
test_ds = Dataset.from_pandas(test_df[['cleaned_text', 'label']])

# %%
# ⚡ Spanish-optimized model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "dccuchile/distilbert-base-spanish-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ⚡ Batch tokenization with proper truncation
def tokenize_fn(batch):
    return tokenizer(
        batch['cleaned_text'],
        padding='max_length',
        truncation=True,
        max_length=256  # Increased from 200
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

# %%
# ⚡ Format datasets for PyTorch
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# %%
# After tokenization, verify keys
print(train_ds[0].keys())  # Should only have: input_ids, attention_mask, label

# %%
# ⚡ Enhanced training arguments
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import torch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

# 3. Train with default classification head
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Focus on F1 for balanced classes
    report_to="none"
)

# %%
import transformers
# ⚡ Improved trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics  # Your existing metrics function
)

trainer.train()

# %%
# Final evaluation
test_results = trainer.evaluate(test_ds)
print("\nTest set performance:")
print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"F1 Score: {test_results['eval_f1']:.4f}")
print(f"Macro F1: {test_results['eval_f1_macro']:.4f}")

# %%
# ⚡ Error analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

preds = trainer.predict(test_ds)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Machine', 'Human'],
            yticklabels=['Machine', 'Human'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# %%
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define Google Drive folder path
drive_folder_path = '/content/drive/My Drive/Colab Notebooks/animals10 CNN Project/distilbert3'

# Save the model
model.save_pretrained(drive_folder_path)
tokenizer.save_pretrained(drive_folder_path)
trainer.save_model(drive_folder_path)

# %%
pred_label = pd.DataFrame({'label': y_pred})
pred_label.to_csv('./data/predictions distilbert2 60 percent3.csv', index=False)
pred_label.to_csv(drive_folder_path + '/predictions distilbert2 60 percent3.csv', index=False)


