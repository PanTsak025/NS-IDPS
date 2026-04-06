import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import time
from tqdm import tqdm

# ========== SIMPLE QUANTIZATION-READY MODEL ==========
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=13, hidden_size=13, output_size=2):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=False)
        )
    
    def forward(self, x):
        return self.layers(x)

def prepare_data():
    """Load and prepare data"""
    set = pd.read_csv('combined.csv')
    set.columns = set.columns.str.lstrip()
    set = set[~set['Label'].isin(['Heartbleed', 'Infiltration', 'Web Attack � Sql Injection'])]
    
    important_features = [
        "Destination Port", "Total Length of Fwd Packets", "Init_Win_bytes_forward", 
        "Fwd Packet Length Max", "Fwd Header Length.1", "Fwd IAT Max",
        "Total Fwd Packets", "min_seg_size_forward", "Fwd IAT Min",
        "Fwd Packet Length Min", "Label"
    ]
    
    set = set[important_features]
    set['Label'] = set['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    set_cleaned = set.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Add rules
    set_cleaned['rule_small_seg_size'] = (set_cleaned['min_seg_size_forward'] <= 20).astype(int)
    set_cleaned['rule_low_packet_count'] = (set_cleaned['Total Fwd Packets'] <= 3).astype(int)
    malicious_ports = [22, 21, 23, 445, 3389, 5900, 135, 1433, 1900, 2323, 4444, 6667, 31337, 12345, 69]
    set_cleaned['rule_suspicious_port'] = (set_cleaned['Destination Port'].isin(malicious_ports)).astype(int)
    
    # Memory optimization
    float_cols = set_cleaned.select_dtypes(include=['float64']).columns
    set_cleaned[float_cols] = set_cleaned[float_cols].astype('float32')
    
    int_cols = set_cleaned.select_dtypes(include='int64').columns
    for col in int_cols:
        col_min, col_max = set_cleaned[col].min(), set_cleaned[col].max()
        if col_min >= 0:
            if col_max < 255: set_cleaned[col] = set_cleaned[col].astype('uint8')
            elif col_max < 65535: set_cleaned[col] = set_cleaned[col].astype('uint16')
            else: set_cleaned[col] = set_cleaned[col].astype('uint32')
    
    X = set_cleaned.drop('Label', axis=1)
    y = set_cleaned['Label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
    
    # Scaling
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    excluded_features = ['rule_small_seg_size', 'rule_low_packet_count', 'rule_suspicious_port']
    features_to_scale = [col for col in X_train.columns if col.strip() not in excluded_features]
    
    scaler = StandardScaler()
    X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create datasets and loaders
    train_dataset_full = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    val_size = int(len(train_dataset_full) * 0.2)
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
    
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return train_loader, val_loader, test_loader, class_weights

def qat_training_simple(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """Simple QAT training - just fine-tune the FP32 model"""
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{epochs}")
        for batch_inputs, batch_labels in progress_bar:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_inputs)
            loss = criterion(y_pred, batch_labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels.long())
                val_epoch_loss += val_loss.item()
        
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"QAT Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return model, train_losses, val_losses

def print_size_of_model(model, label=""):
    """Print model size"""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print(f"Model: {label} \t Size (KB): {size/1e3:.2f}")
    os.remove('temp.p')
    return size

def measure_latency(model, dataloader, device, num_batches=100, warmup=10):
    """Measure model latency"""
    model.eval()
    model.to(device)
    
    # Warmup
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= warmup: break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Measurement
    total_time = 0.0
    count = 0
    batch_size = dataloader.batch_size
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches: break
                
            inputs = inputs.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
            count += 1

    avg_latency = total_time / count
    avg_per_sample = (avg_latency * 1000) / batch_size
    
    print(f"\nLatency Results ({device}):")
    print(f"- Avg latency per batch: {avg_latency*1000:.4f} ms")
    print(f"- Avg latency per sample: {avg_per_sample:.4f} ms")
    print(f"- Throughput: {batch_size/avg_latency:.2f} samples/sec")
    
    return avg_latency

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    print("📊 Loading and preparing data...")
    train_loader, val_loader, test_loader, class_weights = prepare_data()
    class_weights = class_weights.to(device)
    
    # Load FP32 model
    print("🔄 Loading pre-trained FP32 model...")
    model_fp32 = NeuralNetwork()
    model_fp32.load_state_dict(torch.load('model_state_dict33.pth', map_location=device))
    model_fp32.to(device)
    
    # Fine-tune for quantization
    print("🛠️  Fine-tuning model for quantization...")
    qat_epochs = 20
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model_fp32.parameters(), lr=0.00005)  # Very low LR
    
    model_qat_trained, _, _ = qat_training_simple(
        model_fp32, train_loader, val_loader, criterion, optimizer, qat_epochs, device
    )
    
    # Save fine-tuned model
    torch.save(model_qat_trained.state_dict(), 'model_qat_finetuned.pth')
    
    # Apply dynamic quantization (simpler and more reliable)
    print("⚡ Applying dynamic quantization...")
    model_qat_trained.eval()
    model_qat_trained.to('cpu')
    
    quantized_model = torch.quantization.quantize_dynamic(
        model_qat_trained,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), 'quantized_model_dynamic.pth')
    
    # Model size comparison
    print("\n📏 Model Size Comparison:")
    fp32_size = print_size_of_model(model_fp32, "FP32 Original")
    qat_size = print_size_of_model(quantized_model, "Dynamic INT8")
    print(f"📉 Compression ratio: {fp32_size/qat_size:.2f}x smaller")
    
    # Performance testing
    print("\n🧪 Testing Dynamic INT8 model performance...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    all_labels, all_preds = [], []
    quantized_model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = quantized_model(inputs)  # CPU inference
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"📊 Dynamic INT8 Model Performance:")
    print(f"  Accuracy:  {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1 Score:  {f1:.6f}")
    
    # Latency comparison
    print("\n⏱️  Latency Comparison:")
    print("FP32 Model (GPU):")
    measure_latency(model_fp32, test_loader, torch.device('cuda'))
    
    print("\nDynamic INT8 Model (CPU):")
    measure_latency(quantized_model, test_loader, torch.device('cpu'))
    
    print("\n✅ Fixed quantization process completed!")

if __name__ == "__main__":
    main()
