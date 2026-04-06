import torch
from torch import nn
from torch import functional as F #moves data forward
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from torch.quantization import QConfig, HistogramObserver, FakeQuantize
from torch.quantization import prepare, convert, get_default_qconfig
import os
from sklearn.metrics import precision_recall_curve
tqdm.pandas()
import time
from torch.utils.data import  random_split
from torch.quantization.observer import MovingAverageMinMaxObserver
torch.backends.quantized.engine = 'fbgemm'
print(torch.__version__)
print(torch.backends.quantized.supported_engines)
print(torch.backends.quantized.engine) 

print(torch.backends.quantized.supported_engines)

from torch.profiler import profile, record_function, ProfilerActivity
from torch.ao.quantization import  default_observer, default_per_channel_weight_observer
from torch.ao.quantization import default_fake_quant, default_per_channel_weight_fake_quant

from torch.nn.intrinsic import LinearReLU

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()

        self.fc1 = LinearReLU(nn.Linear(14, 14), nn.ReLU())
        self.fc2 = LinearReLU(nn.Linear(14, 14), nn.ReLU())

        self.fc3 = nn.Linear(14, 2)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dequant(x)
        return x

# Fuse properly before quantization
model = NeuralNetwork()
# model = torch.quantization.fuse_modules(model, [['fc1', 'relu1'], ['fc2', 'relu2']])

torch.manual_seed(25)
#model = NeuralNetwork()
batch_size = 512     # Batch size
learning_rate = 0.0001# Learning rate
epochs = 100   # Training iterations 
losses = []

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device
device = torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

set = pd.read_csv('combined.csv')      # get dataset
set = set[~set['Label'].isin(['Heartbleed', 'Infiltration', 'Web Attack � Sql Injection'])]  #drop 3 labels


important_features = [
    "Destination Port", 
    "Init_Win_bytes_forward", 
    "min_seg_size_forward",
    "Flow IAT Min", 
    "Fwd Header Length", 
    "Max Packet Length", 
    "Subflow Fwd Bytes",
    "Avg Fwd Segment Size", 
    "Flow IAT Std",
    "Flow Duration", 
    "Flow Packets/s",
    "Fwd IAT Std", 
    "Fwd PSH Flags",
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
rule_cols = [
    "rule_suspicious_port",
    # "is_low_port",
    # "is_high_port",
    # "small_pkt_var",
    # "high_pkt_var",
    # "fwd_iat_min_is_low",
    # "many_fwd_pkts",
    # "long_flow_duration",
    # "has_psh_flag",
    # "has_urg_flag",
    # "fwd_iat_total_is_high"
]

# Combine all rules — if *any* rule triggers, mark as 1
rule_column = (
    set_cleaned.loc[X_test.index, rule_cols]
    .any(axis=1)
    .astype(int)
    .values
)

# Now scale using the original scaler on the resampled data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# rf = RandomForestClassifier(n_estimators=100, random_state=25, n_jobs=-1)
# rf.fit(X_train, y_train)

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
# # X_train[uint_cols] = X_train[uint_cols].astype(np.int32)
# # X_test[uint_cols] = X_test[uint_cols].astype(np.int32)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor of class weights or scalar for binary
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Prob of the true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=6)
# custom_config = QConfig(
#     activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=127, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
#     weight=FakeQuantize.with_args(observer=torch.quantization.PerChannelMinMaxObserver, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
# )
model.train()
qconfig = QConfig(
    activation=default_fake_quant,
    weight=default_per_channel_weight_fake_quant
)
# model.qconfig = custom_config
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
qat_model = torch.quantization.prepare_qat(model, inplace=True)


# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
# # More aggressive weighting
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = FocalLoss(alpha=class_weights, gamma=2)

optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)


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
            
            start_time = time.perf_counter()  # More precise than time.time()
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

#def measure_latency(model, dataloader, device, num_batches=100):
    # model.eval()
    # model.to(device)

    # total_time = 0.0
    # count = 0

    # with torch.no_grad():
    #     for i, (inputs, _) in enumerate(dataloader):
    #         if i >= num_batches:
    #             break

    #         inputs = inputs.to(device)

    #         start_time = time.time()
    #         _ = model(inputs)
    #         end_time = time.time()

    #         total_time += (end_time - start_time)
    #         count += 1

    # avg_latency = total_time / count
    # print(f"Average latency per batch on {device}: {avg_latency*1000:.3f} ms")
    # return avg_latency

train_losses = []
val_losses = []
val_precisions = []
val_recalls = []
val_f1s = []
optimal_thresholds = []

for epoch in range(epochs):
    qat_model.train()
    epoch_loss = 0.0

    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = qat_model(batch_inputs)
        loss = criterion(y_pred, batch_labels.long())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # === VALIDATION ===
    qat_model.eval()
    val_epoch_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(device, non_blocking=True)
            val_labels = val_labels.to(device, non_blocking=True)
            val_outputs = qat_model(val_inputs)
            val_loss = criterion(val_outputs, val_labels.long())
            val_epoch_loss += val_loss.item()

            # Convert logits to probabilities
            probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(val_labels.cpu().numpy())

    avg_val_loss = val_epoch_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # === Optimal threshold calculation ===
    precision_vals, recall_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    optimal_thresholds.append(optimal_threshold)

    preds_optimal = (all_probs > optimal_threshold).astype(int)
    precision = precision_score(all_labels, preds_optimal, zero_division=0)
    recall = recall_score(all_labels, preds_optimal, zero_division=0)
    f1 = f1_score(all_labels, preds_optimal, zero_division=0)

    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# === Plot losses and metrics ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_precisions, label='Precision', marker='o')
plt.plot(val_recalls, label='Recall', marker='x')
plt.plot(val_f1s, label='F1 Score', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics (Optimal Threshold)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
def testing(neuralN, device, test_loader):
    all_labels = []
    all_preds = []
    all_probs = []  # For ROC curve

    neuralN.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = neuralN(inputs)

            if outputs.is_quantized:
                outputs = outputs.dequantize()  # Convert to FP32 for softmax
            if hasattr(outputs, 'int_repr'):  # Check if output is quantized
                outputs = outputs.dequantize()
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for positive class

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    precision_vals, recall_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    print(optimal_threshold)
    all_probs = np.array(all_probs)
    preds_final = (all_probs > optimal_threshold).astype(int)

    #preds_adjusted = (np.array(all_probs) > 0.9).astype(int)
    accuracy = accuracy_score(all_labels, preds_final)
    precision = precision_score(all_labels, preds_final, average='binary')
    recall = recall_score(all_labels, preds_final, average='binary')
    f1 = f1_score(all_labels, preds_final, average='binary')
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

qat_model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        _ = qat_model(inputs.to(device))
        break  # A few batches are enough
quantized_model = torch.quantization.convert(qat_model, inplace=False)
#testing(model, device, test_loader)
# def check_quantization(model):
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.quantized.Linear):
#             print(f"Quantized layer: {name}")
#             print(f"  Weight dtype: {module.weight().dtype}")
#             # New way to check activation dtype in recent PyTorch
#             if hasattr(module, 'scale'):
#                 print(f"  Scale: {module.scale}")
#                 print(f"  Zero point: {module.zero_point}")
#         elif isinstance(module, torch.quantization.QuantStub):
#             print(f"Input quantizer: {name}")
#         elif isinstance(module, torch.quantization.DeQuantStub):
#             print(f"Output dequantizer: {name}")
#         elif isinstance(module, nn.Linear):
#             print(f"Unquantized layer: {name}")
# check_quantization(quantized_model)
torch.save(quantized_model.state_dict(), 'quantized_model_state_dict.pth')
#new_model = NeuralNetwork()
#new_model = torch.quantization.fuse_modules(new_model, [['fc1', 'relu1'], ['fc2', 'relu2']])
#new_model.load_state_dict(torch.load('model_state_dict33.pth', weights_only=True))


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=print_size_of_model(model,"fp32")
q=print_size_of_model(quantized_model,"int8")
print("{0:.10f} times smaller".format(f/q))

print("\nQAT Model Performance:")
testing(qat_model, device, test_loader)

print("\nFully Quantized Model Performance:")
testing(quantized_model, device, test_loader)

print("Latency - FP32 model ")
measure_latency(model, test_loader, device)
# print("\nLatency - QAT model:")
# measure_latency(qat_model, test_loader, device)
print("\nLatency - INT8 Quantized:")
measure_latency(quantized_model, test_loader, device)