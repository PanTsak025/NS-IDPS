import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import precision_recall_curve
from torch.profiler import profile, record_function, ProfilerActivity
tqdm.pandas()
import time
from torch.utils.data import  random_split
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.optim.lr_scheduler import OneCycleLR

def compute_normalization_params(dataloader):                 # normalize NN params
    features = []
    for batch in dataloader:
        features.append(batch[0])  
    all_features = np.concatenate(features)
    
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    
    feature_std = np.where(feature_std < 1e-7, 1.0, feature_std)
    
    return feature_mean, feature_std

def measure_latency(model, dataloader, device, num_batches=100, warmup=10, use_profiler=False):    # only more latency measurement
    model.eval()
    model.to(device)
    
    # warmup
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= warmup:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # do the measurement
    total_time = 0.0
    count = 0
    batch_size = dataloader.batch_size
    
    if use_profiler:
        #detailed profiling
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                for i, (inputs, _) in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    inputs = inputs.to(device)
                    _ = model(inputs)
        
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        return
    
    #Regular timing
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            #Synchronize if CUDA
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

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=13, hidden_size=13, output_size=2):
        super(NeuralNetwork, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        #Layers 
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size,bias=False)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size,bias=False)
        
    def forward(self, x):
        x = self.quant(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        #Fuse layers for better quantization performance
        fuse_modules(self, [['layer1', 'relu1'], ['layer2', 'relu2']], inplace=True)


def prepare_model_for_qat(model):    #fake quantization so model is ready for QAT
    model.fuse_model()
    qconfig = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=0,  #ensures ReLU alignment
        quant_max=255,
        averaging_constant=0.01
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        averaging_constant=0.01
    )
    )

    model.qconfig = qconfig
    return torch.quantization.prepare_qat(model, inplace=False)


torch.manual_seed(25)
model = NeuralNetwork()
batch_size = 128 
learning_rate = 0.0001 
epochs = 65  
losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device
print(f"Using device: {device}")
model.to(device)

set = pd.read_csv('combined.csv')  
set.columns = set.columns.str.lstrip()
class_counts = set['Label'].value_counts()
#print(class_counts)    # get dataset
set = set[~set['Label'].isin(['Heartbleed', 'Infiltration', 'Web Attack � Sql Injection'])]  #drop 3 labels


missing_values = set.isna().sum()
missing_percentage = (missing_values / len(set)) * 100
duplicates = set.duplicated()
duplicate_count = duplicates.sum()
set.shape
# Output results
#print(f"Number of duplicate rows: {duplicate_count}")
set = set.drop_duplicates(keep='first')
del duplicates

# Identify columns with identical data
identical_columns = {}
columns = set.columns
list_control = columns.copy().tolist()

# Compare each pair of columns
for col1 in columns:
    for col2 in columns:
        if col1 != col2:
            if set[col1].equals(set[col2]):
                if (col1 not in identical_columns) and (col1 in list_control):
                    identical_columns[col1] = [col2]
                    list_control.remove(col2)
                elif (col1 in identical_columns) and (col1 in list_control):
                    identical_columns[col1].append(col2)
                    list_control.remove(col2)

# Print the result
#if identical_columns:
#    print("Identical columns found:")
#    for key, value in identical_columns.items():
 #       print(f"'{key}' is identical to {value}")
#else: print("No identical columns found.")


# Printing columns with missing values
#for column, count in missing_values.items():
#    if count != 0:
#        print(f"Column '{column}' has {count} missing values, which is {missing_percentage[column]:.2f}% of the total")


# Evaluating percentage of missing values per column
threshold = 10
missing_percentage = (set.isnull().sum() / len(set)) * 100

# Filter columns with missing values over the threshold
high_missing_cols = missing_percentage[missing_percentage > threshold]

# Print columns with high missing percentages
#if len(high_missing_cols) > 0:
#    print(f'The following columns have over {threshold}% of missing values:')
#    print(high_missing_cols)
#else:
#    print('There are no columns with missing values greater than the threshold')
row_missing_percentage = (set.isna().sum(axis=1) / set.shape[1]) * 100
print(row_missing_percentage.describe())

missing_rows = set.isna().any(axis=1).sum()
#print(f'\nTotal rows with missing values: {missing_rows}')


# Dropping missing values
data = set.dropna()
#print(f'Dataset shape after row-wise removal: {data.shape}')
class_counts = set['Label'].value_counts()
#print(class_counts)


important_features = [
    "Destination Port", 
    "Total Length of Fwd Packets",
    "Init_Win_bytes_forward", 
    "Fwd Packet Length Max",
    "Fwd Header Length.1",
    "Fwd IAT Max",
    "Total Fwd Packets",
    "min_seg_size_forward",
    "Fwd IAT Min",
    "Fwd Packet Length Min",
    "Label"  
]

# # # Keep only important features and the label
set = set[important_features]

class_counts = set['Label'].value_counts()
# print(class_counts)
set['Label'] = set['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)  # Convert labels to binary
set_cleaned = set.replace([np.inf, -np.inf], np.nan).dropna() # I drop 2,876 out of 2,830,743 entries
# print(class_counts)

set_cleaned['rule_small_seg_size'] = (
    set_cleaned['min_seg_size_forward'] <= 20
).astype(int)
set_cleaned['rule_low_packet_count'] = (
    set_cleaned['Total Fwd Packets'] <= 3
).astype(int)

# Binary flag for commonly abused/malicious ports
malicious_ports = [22, 21, 23, 445, 3389, 5900, 135, 1433, 1900, 2323, 4444, 6667, 31337, 12345, 69]

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
X = set_cleaned.drop('Label', axis=1)
y = set_cleaned['Label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25)

X_train = X_train.astype(np.float32).copy()
X_test = X_test.astype(np.float32).copy()

# float_cols = X_train.select_dtypes(include=['float32']).columns

# scaler = StandardScaler()
# X_train[float_cols] = scaler.fit_transform(X_train[float_cols])
# X_test[float_cols] = scaler.transform(X_test[float_cols])

# Explicitly exclude binary features
excluded_features = ['rule_small_seg_size','rule_low_packet_count', 'rule_large_packet', 'rule_suspicious_port']

# Scale all features except excluded ones
features_to_scale = [col for col in X_train.columns if col.strip() not in excluded_features]

# Scale
scaler = StandardScaler()
X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])

# print("X_train mean (float cols):", X_train.mean())
# print("X_train std (float cols):", X_train.std())

                                                                                                    # Uncomment for RandomForest significance + correlation per feature.

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

# # Plot top 20

# plt.figure(figsize=(10, 6))
# plt.barh(feature_names[sorted_idx[:20]][::-1], importances[sorted_idx[:20]][::-1])
# plt.xlabel("Feature Importance")
# plt.title("Top 20 Important Features (Random Forest)")
# plt.tight_layout()
# plt.show()                                                                             \\ IMPORTANCES + CORRELATION CODE ENDS HERE

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset_val_train = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_split = 0.2
val_size = int(len(train_dataset_val_train) * val_split)
train_size = len(train_dataset_val_train) - val_size

train_dataset, val_dataset = random_split(train_dataset_val_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=6)
feature_mean, feature_std = compute_normalization_params(train_loader)
qat_model = prepare_model_for_qat(model)

train_losses = []
val_losses = []
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.clone().detach())
optimizer = torch.optim.AdamW(qat_model.parameters(), lr=learning_rate, weight_decay=1e-4)

scheduler = OneCycleLR(
    optimizer,
    max_lr= 0.001,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.05,      # warm-up for 20% of training
    anneal_strategy='cos',  # cosine decay after warm-up
    div_factor=10,      # initial LR = max_lr/div_factor (1e-4)
    final_div_factor=100  # very low final LR
)

def convert_to_quantized(model):
    # Convert the trained QAT model to quantized
    quantized_model = torch.quantization.convert(model.eval(), inplace=False)
    return quantized_model

# def debug_quantization(qat_model, quantized_model, test_loader, device):             This was used for debugging qat, it is optional
#     """Compare outputs between QAT and quantized models"""
#     qat_model.eval()
#     quantized_model.eval()
    
#     #Gets a small batch for comparison
#     for inputs, labels in test_loader:
#         inputs = inputs[:32]  # First 32 samples
#         break
    
#     #QAT model outputs GPU
#     with torch.no_grad():
#         qat_outputs = qat_model(inputs.to(device))
#         qat_probs = torch.softmax(qat_outputs, dim=1)[:, 1]
    
#     #Quantized model outputs CPU
#     with torch.no_grad():
#         quant_outputs = quantized_model(inputs.to('cpu'))
#         quant_probs = torch.softmax(quant_outputs, dim=1)[:, 1]
    
#     print(f"\nDEBUG: Model Output Comparison")
#     print(f"QAT probabilities: {qat_probs[:5].cpu().numpy()}")
#     print(f"Quantized probabilities: {quant_probs[:5].numpy()}")
#     print(f"QAT mean/std: {qat_probs.mean():.6f}/{qat_probs.std():.6f}")
#     print(f"Quantized mean/std: {quant_probs.mean():.6f}/{quant_probs.std():.6f}")
#     print(f"Max difference: {torch.abs(qat_probs.cpu() - quant_probs).max():.6f}")

def train_with_qat(model, device, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10):
    train_losses = []
    val_losses = []
    learning_rates = []  #Track LR changes
    model.train()  #Set to QAT training mode
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        #Training Phase
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            y_pred = model(batch_inputs)
            loss = criterion(y_pred, batch_labels.long())
            loss.backward()
            optimizer.step()
            scheduler.step()  #  update LR after each batch
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation Phase
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels.long())
                val_epoch_loss += val_loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        #Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        if epoch == epochs // 2:
            print("Freezing observers...")
            model.apply(torch.quantization.disable_observer)

    #plot training curves
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

print("\n=== Training QAT Model ===")
qat_model, qat_train_losses, qat_val_losses = train_with_qat(
    qat_model, device, train_loader, val_loader, criterion, optimizer,scheduler=scheduler, epochs=epochs
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

qat_model.to('cpu')

# Convert to quantized model
print("\n=== Converting to Quantized Model ===")
quantized_model = convert_to_quantized(qat_model)

# Debug quantization issues     this is optional, debugs qat
# debug_quantization(qat_model, quantized_model, test_loader, torch.device('cpu'))

# Evaluate quantized model
print("\n=== Quantized Model Evaluation ===")  # Explicitly move to CPU
testing(quantized_model, torch.device('cpu'), test_loader)  # Evaluate on CPU

# Compare model sizes
print("\n=== Model Size Comparison ===")
f = print_size_of_model(qat_model, "FP32")
q = print_size_of_model(quantized_model, "INT8")
print(f"Size reduction: {f/q:.2f}x")

# Measure latencies
print("\n=== Latency Measurements ===")
print("\nFP32 Model:")
measure_latency(model, test_loader, torch.device('cpu'))

print("\nQAT Model (Before Quantization):")
measure_latency(qat_model, test_loader, torch.device('cpu'))

print("\nINT8 Quantized Model:")
measure_latency(quantized_model, test_loader, torch.device('cpu'))

print(quantized_model)
# Save models
torch.save(model.state_dict(), 'model_fp32.pth')
torch.save(qat_model.state_dict(), 'model_qat.pth')
torch.save(quantized_model.state_dict(), 'model_int8.pth')

def export_fixed_point(model, filename="nn_params.bpf.h", scale_bits=16):          #exports NN params to a header file for kernel reads
    model.eval()
    Q = 1 << 16  
    with open(filename, 'w') as f:
        #Header 
        f.write("#ifndef NN_PARAMS_BPF_H\n")
        f.write("#define NN_PARAMS_BPF_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define Q {Q}\n")
        f.write(f"#define SCALE_BITS 16\n\n")

        
        f.write("static const int64_t feature_mean[] = {\n    ")
        f.write(", ".join([str(int(round(m * Q))) for m in feature_mean]) + "\n};\n\n")

       
        f.write("static const int64_t feature_std[] = {\n    ")
        f.write(", ".join([str(int(round(s * Q))) for s in feature_std]) + "\n};\n\n")



        # Layer parameters
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.quantized.Linear):
                weight = module.weight()
                
                # Weights (int8)
                f.write(f"static const int8_t {name}_weights[] = {{\n    ")
                f.write(", ".join(map(str, weight.int_repr().flatten().tolist())) + "\n};\n\n")
                
                # Weight quantization params
                f.write(f"static const int32_t {name}_scales[] = {{\n    ")
                f.write(", ".join([str(int(s * Q)) for s in weight.q_per_channel_scales().numpy()]) + "\n};\n\n")
                
                f.write(f"static const int32_t {name}_zero_points[] = {{\n    ")
                f.write(", ".join(map(str, weight.q_per_channel_zero_points().numpy())) + "\n};\n\n")

        f.write("#endif // NN_PARAMS_BPF_H\n")
#export_fixed_point(quantized_model)   UNCOMMENT ONLY TO RE-TRAIN, nn_params.bpf.h is filled for now and will work as is.
