import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import StockDataset
from models import MLP, CNN1D, LSTMModel
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'combined.csv')
checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

def main():
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    
    train_df = df[df.index < '2022-01-01']
    val_df = df[(df.index >='2022-01-01') & (df.index < '2023-01-01')]

    train_ds = StockDataset(train_df)
    val_ds = StockDataset(val_df)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    for model_class, model_name in [(MLP, 'mlp'), (CNN1D, 'cnn'), (LSTMModel, 'lstm')]:
        model = model_class()
        train_model(model, model_name, train_loader, val_loader, checkpoint_dir)
            
def train_model(model, model_name, train_loader, val_loader, checkpoint_dir, num_epochs=10, patience=3):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f'\n===Training {model_name}===')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_samples += y_batch.size(0)

        train_accuracy = train_correct/train_samples        
        avg_train_loss = train_loss_total/len(train_loader)
        
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)
                val_loss_total += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds==y_batch).sum().item()
                val_samples += y_batch.size(0)
                
        val_accuracy = val_correct/val_samples
        avg_val_loss = val_loss_total/len(val_loader)     
        
        print(f'[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
if __name__ == '__main__':
    main()