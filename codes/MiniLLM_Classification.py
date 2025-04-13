from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from scipy.special import expit

# ✅ Step 1: Load data
df = pd.read_csv("train.csv")

# Remove rows with non-numeric id or missing labels
df = df[pd.to_numeric(df['id'], errors='coerce').notnull()]
df = df[(df['winner_model_a'] == 1) | (df['winner_model_b'] == 1) | (df['winner_tie'] == 1)]

# Create labels based on the winner column
def label_from_winner(row):
    if row['winner_model_a'] == 1:
        return 0  # response_a wins
    elif row['winner_model_b'] == 1:
        return 1  # response_b wins
    else:
        return 2  # tie

df['label'] = df.apply(label_from_winner, axis=1)

# ✅ Step 2: Create pairwise data (binary classification)
data = []
for _, row in df.iterrows():
    if row['label'] == 0:
        data.append({'text': row['prompt'] + " [SEP] " + row['response_a'] + " [SEP] " + row['response_b'], 'label': 0})
        data.append({'text': row['prompt'] + " [SEP] " + row['response_b'] + " [SEP] " + row['response_a'], 'label': 1})
    elif row['label'] == 1:
        data.append({'text': row['prompt'] + " [SEP] " + row['response_a'] + " [SEP] " + row['response_b'], 'label': 1})
        data.append({'text': row['prompt'] + " [SEP] " + row['response_b'] + " [SEP] " + row['response_a'], 'label': 0})
    else:
        data.append({'text': row['prompt'] + " [SEP] " + row['response_a'] + " [SEP] " + row['response_b'], 'label': 2})
        data.append({'text': row['prompt'] + " [SEP] " + row['response_b'] + " [SEP] " + row['response_a'], 'label': 2})

score_df = pd.DataFrame(data)

# ✅ Step 3: Split into train/validation
train_df, val_df = train_test_split(score_df, test_size=0.2, random_state=42)

# ✅ Step 4: Tokenization
model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 256
def tokenize_fn(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

# ✅ Step 5: Model Setup (classification)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Change num_labels to 3 for the 3 classes (a, b, tie)

# ✅ Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./minilm-test",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

# ✅ Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,  # Pass eval dataset for validation during training
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(axis=1), p.label_ids)}  # Calculate accuracy
)

# ✅ Step 8: Training
trainer.train()
trainer.save_model("./minilm-test")


# Evaluate the model on the validation set (optional)
eval_results = trainer.evaluate(eval_dataset=val_ds)
print("Evaluation results: ", eval_results)

test_df = pd.read_csv('test.csv')
test_data = []

# Create pairs for prediction
for _, row in test_df.iterrows():
    test_data.append({'text': row['prompt'] + " [SEP] " + row['response_a'] + " [SEP] " + row['response_b']})

test_score_df = pd.DataFrame(test_data)

# Tokenize the test data
test_ds = Dataset.from_pandas(test_score_df)
test_ds = test_ds.map(tokenize_fn, batched=True)

# Get predictions for the test set
predictions = trainer.predict(test_ds)

# Apply softmax to the logits to get probabilities for each class
import torch.nn.functional as F
probs = F.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# Get the probability for A (0), B (1), and Tie (2)
prob_a = probs[:, 0]
prob_b = probs[:, 1]
prob_tie = probs[:, 2]

# Create output predictions (probabilities for A, B, Tie)
test_df['winner_model_a'] = prob_a
test_df['winner_model_b'] = prob_b
test_df['winner_tie'] = prob_tie

# Save the predictions to a CSV file
test_df[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']].to_csv('predictions_2.csv', index=False)
