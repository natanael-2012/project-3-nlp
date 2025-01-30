# %% [markdown]
# # NLP Challenge - DistilBERT Classifier (Fixed Version)

# %% [markdown]
# ## Critical Fixes & Improvements
# 1. ⚡ **Batch Tokenization**: Fixed inefficient row-wise processing
# 2. ⚡ **Spanish-specific Model**: Using `dccuchile/distilbert-base-spanish-uncased`
# 3. ⚡ **Dataset Handling**: Proper HuggingFace Dataset integration
# 4. ⚡ **Training Optimization**: Added warmup, gradient clipping, and better metrics

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
# ⚡ Convert to HuggingFace Dataset
from datasets import Dataset
train_ds = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
val_ds = Dataset.from_pandas(val_df[['cleaned_text', 'label']])
test_ds = Dataset.from_pandas(test_df[['cleaned_text', 'label']])

# %%
# ⚡ Spanish-optimized model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "dccuchile/distilbert-base-spanish-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
# ⚡ Custom model with improved classifier head
from torch import nn
from transformers import DistilBertPreTrainedModel, DistilBertModel

class CustomDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(config.dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_labels)
        )
        self.init_weights()

    def forward(self, **inputs):
        outputs = self.distilbert(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

model = CustomDistilBert.from_pretrained(model_name, num_labels=2)

# %%
# ⚡ Enhanced training arguments
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# %%
# ⚡ Improved trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# Train
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