from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
from scipy.special import expit

# ✅ Step 1: Load data
df = pd.read_csv("train.csv")

# Remove rows with non-numeric id or missing labels
df = df[pd.to_numeric(df['id'], errors='coerce').notnull()]
df = df[(df['winner_model_a'] == 1) | (df['winner_model_b'] == 1) | (df['winner_model_tie'] == 1)]

def label_from_winner(row):
    if row['winner_model_a'] == 1:
        return 0
    elif row['winner_model_b'] == 1:
        return 1
    else:
        return 2

df['label'] = df.apply(label_from_winner, axis=1)

# ✅ Step 2: Create pairwise data with soft scores (Option B)
data = []
for _, row in df.iterrows():
    if row['label'] == 0:
        data.append({'text': row['prompt'] + " " + row['response_a'], 'score': 1})
        data.append({'text': row['prompt'] + " " + row['response_b'], 'score': 0})
    elif row['label'] == 1:
        data.append({'text': row['prompt'] + " " + row['response_a'], 'score': 0})
        data.append({'text': row['prompt'] + " " + row['response_b'], 'score': 1})
    else:
        data.append({'text': row['prompt'] + " " + row['response_a'], 'score': 0.5})
        data.append({'text': row['prompt'] + " " + row['response_b'], 'score': 0.5})

score_df = pd.DataFrame(data)

# ✅ Step 3: Split into train/validation
train_df, val_df = train_test_split(score_df, test_size=0.2, random_state=42)

# ✅ Step 4: Tokenization
model_name = "microsoft/MiniLM-L6-v2"
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

train_ds = train_ds.rename_column("score", "labels")
val_ds = val_ds.rename_column("score", "labels")
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ✅ Step 5: Model Setup (regression)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# ✅ Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./minilm-results",
    evaluation_strategy="epoch",  # Evaluate after every epoch
    save_strategy="epoch",  # Save model after every epoch
    logging_dir="./logs",
    per_device_train_batch_size=32,
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
    eval_dataset=val_ds,
)

# Start training and evaluation
trainer.train()


# ✅ Step 9: Manual Evaluation
def manual_evaluate(trainer, val_ds):
    outputs = trainer.predict(val_ds)
    preds = outputs.predictions.flatten()
    labels = val_ds["labels"].numpy()
    mse = mean_squared_error(labels, preds)
    print(f"\n📏 Manual Validation MSE: {mse:.4f}")
    return mse

manual_evaluate(trainer, val_ds)


# Evaluate the model on the validation set (optional)
eval_results = trainer.evaluate(eval_dataset=val_ds)
print("Evaluation results: ", eval_results)

# ✅ Step 8: Prediction and score computation
test_df = pd.read_csv('test.csv')
test_data = []

for _, row in test_df.iterrows():
    # Create pairs for prediction
    test_data.append({'text': row['prompt'] + " " + row['response_a']})
    test_data.append({'text': row['prompt'] + " " + row['response_b']})

test_score_df = pd.DataFrame(test_data)

# Tokenize the test data
test_ds = Dataset.from_pandas(test_score_df)
test_ds = test_ds.map(tokenize_fn, batched=True)



# Get predictions for the test set
predictions = trainer.predict(test_ds)
scores = predictions.predictions.flatten()
scores_a = scores[::2]  # All predictions for response_a
scores_b = scores[1::2]  # All predictions for response_b

# Calculate score differences
diff = scores_a - scores_b

# Convert difference to probabilities using sigmoid
prob_a = expit(diff)  # probability of selecting response_a
prob_b = 1 - prob_a   # probability of selecting response_b

# Calculate tie probability (you can set a threshold for the tie)
tie_threshold = 0.05  # Adjust this threshold as needed
prob_tie = (abs(diff) < tie_threshold).astype(float)  # Tie if the difference is small

# Adjust probabilities to sum to 1
total_prob = prob_a + prob_b + prob_tie
prob_a /= total_prob
prob_b /= total_prob
prob_tie /= total_prob

# Create output predictions (probabilities for A, B, Tie)
test_df['winner_model_a'] = prob_a
test_df['winner_model_b'] = prob_b
test_df['winner_tie'] = prob_tie

# Save the predictions to a CSV file
test_df[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']].to_csv('predictions.csv', index=False)
