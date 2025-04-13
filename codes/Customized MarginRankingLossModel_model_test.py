import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Model Definition
# -----------------------------
class MarginRankingLossModel(nn.Module):
    def __init__(self, base_model):
        super(MarginRankingLossModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        output_a = self.base_model(input_ids_a, attention_mask=attention_mask_a).last_hidden_state[:, 0, :]
        output_b = self.base_model(input_ids_b, attention_mask=attention_mask_b).last_hidden_state[:, 0, :]
        score_a = self.fc(output_a).squeeze(1)
        score_b = self.fc(output_b).squeeze(1)
        return score_a, score_b

# -----------------------------
# Load tokenizer and model
# -----------------------------
model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large"
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
base_model = AutoModel.from_pretrained(model_name)
ranking_model = MarginRankingLossModel(base_model)
ranking_model.load_state_dict(torch.load("ranking_model.pth"))
ranking_model.to(device)
ranking_model.eval()

# -----------------------------
# Prepare validation set
# -----------------------------
# df = pd.read_csv("train.csv")

# # Filter for valid labels
# df = df[pd.to_numeric(df['id'], errors='coerce').notnull()]
# df = df[(df['winner_model_a'] == 1) | (df['winner_model_b'] == 1) | (df['winner_tie'] == 1)]

# # Label creation
# def label_from_winner(row):
#     if row['winner_model_a'] == 1:
#         return 1
#     elif row['winner_model_b'] == 1:
#         return -1
#     else:
#         return 0

# df['label'] = df.apply(label_from_winner, axis=1)

# # Create pairwise data
# data = []
# for _, row in df.iterrows():
#     if row['label'] == 1:
#         data.append({'text_a': row['prompt'] + " " + row['response_a'], 'text_b': row['prompt'] + " " + row['response_b'], 'label': 1.0})
#         data.append({'text_a': row['prompt'] + " " + row['response_b'], 'text_b': row['prompt'] + " " + row['response_a'], 'label': -1.0})
#     elif row['label'] == -1:
#         data.append({'text_a': row['prompt'] + " " + row['response_a'], 'text_b': row['prompt'] + " " + row['response_b'], 'label': -1.0})
#         data.append({'text_a': row['prompt'] + " " + row['response_b'], 'text_b': row['prompt'] + " " + row['response_a'], 'label': 1.0})
#     else:
#         data.append({'text_a': row['prompt'] + " " + row['response_a'], 'text_b': row['prompt'] + " " + row['response_b'], 'label': 0.0})

# score_df = pd.DataFrame(data)
# _, val_df = train_test_split(score_df, test_size=0.2, random_state=42)

# # Tokenize
# def tokenize_pairwise(example):
#     encoding_a = tokenizer(example['text_a'], truncation=True, padding='max_length', max_length=128)
#     encoding_b = tokenizer(example['text_b'], truncation=True, padding='max_length', max_length=128)
#     return {
#         'input_ids_a': encoding_a['input_ids'],
#         'attention_mask_a': encoding_a['attention_mask'],
#         'input_ids_b': encoding_b['input_ids'],
#         'attention_mask_b': encoding_b['attention_mask'],
#         'label': example['label']
#     }

# val_ds = Dataset.from_pandas(val_df)
# val_ds = val_ds.map(tokenize_pairwise, batched=True)
# val_ds.set_format("torch", columns=["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "label"])
# val_dataloader = DataLoader(val_ds, batch_size=8)

# # -----------------------------
# # Evaluate
# # -----------------------------
# def manual_evaluate(model, val_dataloader):
#     model.eval()
#     y_true = []
#     y_pred = []
    
#     with torch.no_grad():
#         for batch in val_dataloader:
#             input_ids_a = batch['input_ids_a'].to(device)
#             attention_mask_a = batch['attention_mask_a'].to(device)
#             input_ids_b = batch['input_ids_b'].to(device)
#             attention_mask_b = batch['attention_mask_b'].to(device)
#             labels = batch['label'].to(device)
            
#             score_a, score_b = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend((score_a - score_b).cpu().numpy())
    
#     mse = mean_squared_error(y_true, y_pred)
#     print(f"Validation MSE: {mse:.4f}")
#     return mse

# # Run evaluation
# manual_evaluate(ranking_model, val_dataloader)

# Load test data (assuming `test.csv` is similar to your train data)
test_df = pd.read_csv("test.csv")

# Preprocess the test data (same as before)
def preprocess_data(df):
    # Preprocess the test data similarly to how you did for the train data
    data = []
    for _, row in df.iterrows():
        data.append({
            'text_a': row['prompt'] + " " + row['response_a'],
            'text_b': row['prompt'] + " " + row['response_b']
        })
    
    return pd.DataFrame(data)

test_data = preprocess_data(test_df)

# Tokenization function
def tokenize_pairwise(example):
    encoding_a = tokenizer(example['text_a'], truncation=True, padding='max_length', max_length=128)
    encoding_b = tokenizer(example['text_b'], truncation=True, padding='max_length', max_length=128)
    
    return {
        'input_ids_a': encoding_a['input_ids'],
        'attention_mask_a': encoding_a['attention_mask'],
        'input_ids_b': encoding_b['input_ids'],
        'attention_mask_b': encoding_b['attention_mask'],
    }

# Convert to dataset format and tokenize
test_ds = Dataset.from_pandas(test_data)
test_ds = test_ds.map(tokenize_pairwise, batched=True)
test_ds.set_format("torch", columns=["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b"])

# Create a DataLoader for the test set
test_dataloader = DataLoader(test_ds, batch_size=8)



import torch.nn.functional as F
import numpy as np

# Prediction function
def predict(model, test_dataloader):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)

            score_a, score_b = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
            margins = score_a - score_b

            logits = torch.stack([margins, -margins, torch.zeros_like(margins)], dim=1)
            probs = F.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)

# Get predictions
probs = predict(ranking_model, test_dataloader)

prob_a = probs[:, 0]
prob_b = probs[:, 1]
prob_tie = probs[:, 2]

# Create output predictions (probabilities for A, B, Tie)
test_df['winner_model_a'] = prob_a
test_df['winner_model_b'] = prob_b
test_df['winner_tie'] = prob_tie

# Save the predictions to a CSV file
test_df[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']].to_csv('predictions_3.csv', index=False)
