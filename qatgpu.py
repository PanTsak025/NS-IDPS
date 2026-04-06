import torch
from torch import nn
from torch import functional as F #moves data forward
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from torch.quantization import quantize_dynamic
import os
from sklearn.metrics import precision_recall_curve
from torch.profiler import profile, record_function, ProfilerActivity
tqdm.pandas()
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig_mapping

import time
from torch.utils.data import  random_split
class PrecisionWeightedLoss(nn.Module):
    def __init__(self, fp_weight=1.0, fn_weight=2.0, reduction='mean'):
        super().__init__()
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        probas = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        weights = torch.ones_like(bce_loss)

        fp_mask = (probas >= 0.5) & (targets == 0)
        fn_mask = (probas < 0.5) & (targets == 1)

        weights[fp_mask] = self.fp_weight
        weights[fn_mask] = self.fn_weight

        loss = bce_loss * weights
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

def measure_latency(model, dataloader, device, num_batches=100, warmup=10, use_profiler=False):
    model.eval()
    model.to(device)
    
    # Warmup (important for CUDA/quantized models)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= warmup:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Measurement
    total_time = 0.0
    count = 0
    batch_size = dataloader.batch_size
    
    if use_profiler:
        # Detailed profiling with torch.profiler
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                for i, (inputs, _) in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    inputs = inputs.to(device)
                    _ = model(inputs)
        
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        return
    
    # Regular timing
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            # Synchronize if CUDA
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
    
    print(f"\nLatency Results:")
    print(f"- Device: {device}")
    print(f"- Batches measured: {count}")
    print(f"- Batch size: {batch_size}")
    print(f"- Avg latency per batch: {avg_latency*1000:.10f} ms")
    print(f"- Avg latency per sample: {avg_per_sample:.10f} ms")
    print(f"- Throughput: {batch_size/avg_latency:.3f} samples/sec")
    
    return avg_latency
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, fuse_modules
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=13, hidden_size=13, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Layers with potential for fusion
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.quant(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.dequant(x)
        return x.squeeze(1)
    
    def fuse_model(self):
        # Fuse layers for better quantization performance
        fuse_modules(self, [['layer1', 'relu1'], ['layer2', 'relu2']], inplace=True)

def prepare_model_for_qat(model):
    # Fuse modules first
    model.fuse_model()
    
    # Set quantization configurations
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare with observer
    prepared_model = torch.ao.quantization.prepare_qat(model)
    return prepared_model

torch.manual_seed(25)
model = NeuralNetwork() 
batch_size = 1024     # Batch size
learning_rate = 0.00005 # Learning rate
epochs = 50     # Training iterations
enlargement_factor = 2 ** 16  # Scaling factor (usage depends on context)
losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device
#device = torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

set = pd.read_csv('combined.csv')      # get dataset
set = set[~set['Label'].isin(['Heartbleed', 'Infiltration', 'Web Attack � Sql Injection'])]  #drop 3 labels


important_features = [
    "Destination Port",
    "Init_Win_bytes_forward", 
    "min_seg_size_forward",
    " Fwd Packet Length Max", 
    "Subflow Fwd Bytes",
    "Fwd Header Length.1",
    "Fwd IAT Min", 
    "Subflow Fwd Packets", 
    "Fwd IAT Max",
    "Fwd IAT Total",
    "Fwd PSH Flags",
    "Fwd Packet Length Min",
    "Label"  
]

# Keep only important features and the label
set = set[important_features]

class_counts = set['Label'].value_counts()
#print(class_counts)
set['Label'] = set['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)  # Convert labels to binary
set_cleaned = set.replace([np.inf, -np.inf], np.nan).dropna() # I drop 2,876 out of 2,830,743 entries


malicious_ports = [22, 21, 23, 445, 3389, 5900, 135, 1433, 1900,2323,4444,6667,31337,12345,69]

set_cleaned['rule_suspicious_port'] = (
    set_cleaned['Destination Port'].isin(malicious_ports)
).astype(int)

X = set_cleaned.drop(columns=['Label'])

class_weights = compute_class_weight('balanced', classes=set_cleaned['Label'].unique(), y=set_cleaned['Label'])
#set_cleaned.info()

float_cols = set_cleaned.select_dtypes(include=['float64']).columns               # Downcasting floats and ints
set_cleaned[float_cols] = set_cleaned[float_cols].astype('float32')               # Result - 1.7 GB memory to 701.2MB

int_cols = set_cleaned.select_dtypes(include='int64').columns
for col in int_cols:
    col_min = set_cleaned[col].min()
    col_max = set_cleaned[col].max()
    
    # Downcast based on range
    if col_min >= 0:
        # Unsigned integers
        if col_max < 255:
            set_cleaned[col] = set_cleaned[col].astype('uint8')
        elif col_max < 65535:
            set_cleaned[col] = set_cleaned[col].astype('uint16')
        elif col_max < 4294967295:
            set_cleaned[col] = set_cleaned[col].astype('uint32')
        else:
            set_cleaned[col] = set_cleaned[col].astype('uint64')  
    else:
        # Signed integers
        if col_min >= -128 and col_max <= 127:
            set_cleaned[col] = set_cleaned[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            set_cleaned[col] = set_cleaned[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            set_cleaned[col] = set_cleaned[col].astype('int32')
        else:
            set_cleaned[col] = set_cleaned[col].astype('int64') 

#print("\nDowncasted Memory Usage (MB):", set_cleaned.memory_usage(deep=True).sum() / 1024**2)

#set_cleaned.info()
X = set_cleaned.drop('Label', axis=1)  # Features (all columns except 'target')
y = set_cleaned['Label'].values               # Target (only the 'target' column)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25)
# print("X_train, y_train:", X_train.shape, y_train.shape)
# print("X_test, y_test:", X_test.shape, y_test.shape)

# float_cols = X.select_dtypes(include=['float32']).columns     # δε κανω scale τα uints και τα κανω στη συνεχεια int32
# uint_cols = X.select_dtypes(include=['uint8', 'uint16', 'uint32']).columns    
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)         # use the same scaler on test set as train to not lose data
X_test = scaler.transform(X_test)

print("X_train mean (float cols):", X_train.mean())
print("X_train std (float cols):", X_train.std())
# rf = RandomForestClassifier(n_estimators=100, random_state=25, n_jobs=-1)
# rf.fit(X_train, y_train)
# # Calculate Pearson correlation between all numeric features
# corr_matrix = set_cleaned.corr(numeric_only=True)  # For mixed data, use `df.select_dtypes(include=np.number).corr()`

# # Display top correlations (absolute values)
# print(corr_matrix.abs().sort_values(by="Label", ascending=False))
# plt.figure(figsize=(12, 10))
# sns.heatmap(
#     corr_matrix, 
#     annot=True,   # Show values
#     fmt=".2f",    # 2 decimal places
#     cmap="coolwarm",  # Red (positive) vs. Blue (negative)
#     vmin=-1, vmax=1,  # Fix scale from -1 to 1
#     mask=np.triu(np.ones_like(corr_matrix))  # Hide upper triangle (optional)
# )
# plt.title("Feature Correlation Heatmap")
# plt.show()
# # Get feature importances
# importances = rf.feature_importances_
# feature_names = X.columns  # This is from original DataFrame
# sorted_idx = np.argsort(importances)[::-1]

# print("\nTop Feature Importances (importance > 0.01):")
# for i in range(len(importances)):
#     if importances[sorted_idx[i]] > 0.01:
#         print(f"{feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.5f}")

# print("\nLow Importance Features (importance <= 0.01):")
# for i in range(len(importances)):
#     if importances[sorted_idx[i]] <= 0.01:
#         print(f"{feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.5f}")

# # Optional: Plot top 20
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.barh(feature_names[sorted_idx[:20]][::-1], importances[sorted_idx[:20]][::-1])
# plt.xlabel("Feature Importance")
# plt.title("Top 20 Important Features (Random Forest)")
# plt.tight_layout()
# plt.show()
# X_train[uint_cols] = X_train[uint_cols].astype(np.int32)
# X_test[uint_cols] = X_test[uint_cols].astype(np.int32)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# X_train_tensor = X_train_tensor.to(device)
# y_train_tensor = y_train_tensor.to(device)
# X_test_tensor = X_test_tensor.to(device)
# y_test_tensor = y_test_tensor.to(device)

train_dataset_val_train = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_split = 0.2
val_size = int(len(train_dataset_val_train) * val_split)
train_size = len(train_dataset_val_train) - val_size

train_dataset, val_dataset = random_split(train_dataset_val_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

qat_model = prepare_model_for_qat(model)
from torch.quantization import MinMaxObserver, MovingAverageMinMaxObserver

qat_model.qconfig = torch.quantization.QConfig(
    activation=MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8,
        reduce_range=True
    ),
    weight=MinMaxObserver.with_args(
        dtype=torch.qint8,
        reduce_range=True
    )
)

train_losses = []
val_losses = []

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = PrecisionWeightedLoss(fp_weight=1.0, fn_weight=4.0)
optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


def convert_to_quantized(model):
    model.eval()
    model.to('cpu')  # Must be on CPU for conversion
    
    # Ensure all parameters are on CPU
    for param in model.parameters():
        param.data = param.data.cpu()
        if param.grad is not None:
            param.grad.data = param.grad.data.cpu()
    
    # Convert with proper settings
    quantized_model = torch.ao.quantization.convert(model)
    return quantized_model



def train_with_qat(model, device, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10):
    train_losses = []
    val_losses = []
    learning_rates = []  # Track LR changes

    model.train()  # Set to QAT training mode
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training Phase
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            y_pred = model(batch_inputs)
            #loss = criterion(y_pred, batch_labels.long())
            loss = criterion(y_pred, batch_labels.float())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels.float())

                #val_loss = criterion(val_outputs, val_labels.long())
                val_epoch_loss += val_loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if scheduler:
            scheduler.step(avg_val_loss)  # For ReduceLROnPlateau
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:  # LR changed
                print(f"LR reduced from {current_lr:.2e} to {new_lr:.2e}")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('QAT Training Progress')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(learning_rates, marker='s', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, train_losses, val_losses


def testing(neuralN, device, test_loader):
    all_labels = []
    all_preds = []
    all_probs = []  # For ROC curve

    neuralN.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = neuralN(inputs)
            probs = torch.sigmoid(outputs)  # sigmoid for binary output logits
            # _, preds = torch.max(outputs, 1)
            # probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for positive class

            all_labels.extend(labels.cpu().numpy())
            #all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    precision_vals, recall_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    print(optimal_threshold)
    all_probs = np.array(all_probs)
    preds_adjusted = (all_probs > optimal_threshold).astype(int)

    accuracy = accuracy_score(all_labels, preds_adjusted)
    precision = precision_score(all_labels, preds_adjusted, average='binary')
    recall = recall_score(all_labels, preds_adjusted, average='binary')
    f1 = f1_score(all_labels, preds_adjusted, average='binary')
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f'Accuracy: {accuracy:.10f}')
    print(f'Precision: {precision:.10f}')
    print(f'Recall: {recall:.10f}')
    print(f'F1 Score: {f1:.10f}')
    print(f'ROC AUC: {roc_auc:.10f}')
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.10f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.show()

# Train with QAT
print("\n=== Training QAT Model ===")
qat_model, qat_train_losses, qat_val_losses = train_with_qat(
    qat_model, device, train_loader, val_loader, criterion, optimizer, epochs=epochs
)
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# Evaluate QAT model before quantization
print("\n=== QAT Model Evaluation (Before Quantization) ===")
testing(qat_model, device, test_loader)

model.to('cpu')
# Convert to quantized model
print("\n=== Converting to Quantized Model ===")
quantized_model = convert_to_quantized(qat_model)

# Evaluate quantized model
print("\n=== Quantized Model Evaluation ===")
quantized_model = quantized_model.to('cpu')  # Explicitly move to CPU
testing(quantized_model, torch.device('cpu'), test_loader)  # Evaluate on CPU

# Compare model sizes
print("\n=== Model Size Comparison ===")
f = print_size_of_model(model, "FP32")
q = print_size_of_model(quantized_model, "INT8")
print(f"Size reduction: {f/q:.2f}x")

# Measure latencies
print("\n=== Latency Measurements ===")
print("\nFP32 Model:")
measure_latency(model, test_loader, device)

print("\nQAT Model (Before Quantization):")
measure_latency(qat_model, test_loader, device)

print("\nINT8 Quantized Model:")
measure_latency(quantized_model, test_loader, torch.device('cpu'))

# Save models
torch.save(model.state_dict(), 'model_fp32.pth')
torch.save(qat_model.state_dict(), 'model_qat.pth')
torch.save(quantized_model.state_dict(), 'model_int8.pth')

# quantized_model = quantize_dynamic(
#     new_model,
#     {torch.nn.Linear},
#     dtype=torch.qint8
# )

# def print_size_of_model(model, label=""):
#     torch.save(model.state_dict(), "temp.p")
#     size=os.path.getsize("temp.p")
#     print("model: ",label,' \t','Size (KB):', size/1e3)
#     os.remove('temp.p')
#     return size

# # compare the sizes
# f=print_size_of_model(model,"fp32")
# q=print_size_of_model(quantized_model,"int8")
# print("{0:.10f} times smaller".format(f/q))
# # compare the performance

# print("\nQuantized stats:")
# testing(quantized_model, torch.device('cpu'), test_loader)


# print("\nLatency - INT8 Quantized ")
# measure_latency(quantized_model, test_loader, torch.device('cpu'))