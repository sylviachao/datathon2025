import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("train.csv")

# Remove rows with non-numeric id or missing labels
df = df[pd.to_numeric(df['id'], errors='coerce').notnull()]
df = df[(df['winner_model_a'] == 1) | (df['winner_model_b'] == 1) | (df['winner_tie'] == 1)]

# Function to create labels for pairwise ranking
def label_from_winner(row):
    if row['winner_model_a'] == 1:
        return 1  # Model A is better
    elif row['winner_model_b'] == 1:
        return -1  # Model B is better
    else:
        return 0  # Tie, we can treat this as neutral, but we want to label as tie (can be adjusted)

df['label'] = df.apply(label_from_winner, axis=1)

# Create pairwise data with scores (1 for better, -1 for worse, 0 for tie)
data = []
for _, row in df.iterrows():
    if row['label'] == 1:
        data.append({'text_a': row['prompt'] + " " + row['response_a'], 'text_b': row['prompt'] + " " + row['response_b'], 'label': 1.0})
        data.append({'text_a': row['prompt'] + " " + row['response_b'], 'text_b': row['prompt'] + " " + row['response_a'], 'label': -1.0})
    elif row['label'] == -1:
        data.append({'text_a': row['prompt'] + " " + row['response_a'], 'text_b': row['prompt'] + " " + row['response_b'], 'label': -1.0})
        data.append({'text_a': row['prompt'] + " " + row['response_b'], 'text_b': row['prompt'] + " " + row['response_a'], 'label': 1.0})
    else:
        data.append({'text_a': row['prompt'] + " " + row['response_a'], 'text_b': row['prompt'] + " " + row['response_b'], 'label': 0.0})

score_df = pd.DataFrame(data)

# Split the data into train/validation sets
train_df, val_df = train_test_split(score_df, test_size=0.2, random_state=42)

##########################################################
from transformers import AutoTokenizer
from datasets import Dataset

# Load tokenizer
model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_pairwise(example):
    encoding_a = tokenizer(example['text_a'], truncation=True, padding='max_length', max_length=128)
    encoding_b = tokenizer(example['text_b'], truncation=True, padding='max_length', max_length=128)
    
    return {
        'input_ids_a': encoding_a['input_ids'],
        'attention_mask_a': encoding_a['attention_mask'],
        'input_ids_b': encoding_b['input_ids'],
        'attention_mask_b': encoding_b['attention_mask'],
        'label': example['label']  # 1.0, -1.0, or 0.0
    }

# Apply tokenization to train and validation sets
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_ds = train_ds.map(tokenize_pairwise, batched=True)
val_ds = val_ds.map(tokenize_pairwise, batched=True)

# Set the format to PyTorch tensors
train_ds.set_format("torch", columns=["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "label"])
val_ds.set_format("torch", columns=["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "label"])


##########################################################
from transformers import AutoModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model
model = AutoModel.from_pretrained(model_name)

# Define Margin Ranking Loss
class MarginRankingLossModel(nn.Module):
    def __init__(self, base_model):
        super(MarginRankingLossModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.config.hidden_size, 1)  # Single score output for ranking
    
    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        # Forward pass for both responses
        output_a = self.base_model(input_ids_a, attention_mask=attention_mask_a).last_hidden_state[:, 0, :]
        output_b = self.base_model(input_ids_b, attention_mask=attention_mask_b).last_hidden_state[:, 0, :]
        
        # Pass through a linear layer to get the ranking score
        score_a = self.fc(output_a).squeeze(1)
        score_b = self.fc(output_b).squeeze(1)
        
        return score_a, score_b

# Instantiate the model with the pre-trained MiniLM
ranking_model = MarginRankingLossModel(base_model=model).to(device)

# Loss function (Margin Ranking Loss)
loss_fn = nn.MarginRankingLoss(margin=0.1)

# Optimizer
optimizer = torch.optim.AdamW(ranking_model.parameters(), lr=2e-5)


##########################################################
# Training Loop
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=8)

# Training
from tqdm import tqdm  # add this import if not present

for epoch in range(3):  # Train for 3 epochs
    ranking_model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        optimizer.zero_grad()

        input_ids_a = batch['input_ids_a'].to(device)
        attention_mask_a = batch['attention_mask_a'].to(device)
        input_ids_b = batch['input_ids_b'].to(device)
        attention_mask_b = batch['attention_mask_b'].to(device)
        labels = batch['label'].to(device)

        score_a, score_b = ranking_model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)

        loss = loss_fn(score_a, score_b, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item(), avg_loss=total_loss / num_batches)

    print(f"Epoch {epoch+1} completed. Avg Loss = {total_loss / num_batches:.4f}")


##########################################################
# Save the model
torch.save(ranking_model.state_dict(), "ranking_model.pth")
tokenizer.save_pretrained("tokenizer")

##########################################################
# Evaluation
from sklearn.metrics import mean_squared_error

# Manual Evaluation
def manual_evaluate(model, val_dataloader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            score_a, score_b = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
            
            # Collect true labels and predicted scores
            y_true.extend(labels.numpy())
            y_pred.extend((score_a - score_b).numpy())  # Predict score difference
    
    mse = mean_squared_error(y_true, y_pred)
    print(f"Validation MSE: {mse:.4f}")
    return mse

# Evaluate the model
manual_evaluate(ranking_model, val_dataloader)

# Save model for later inference/testing
ranking_model.save_pretrained("final_ranking_model")
