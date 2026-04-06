import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QConfig
import copy

# Your existing NeuralNetwork class
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

def prepare_model_for_qat(model):
    """Prepare model for Quantization-Aware Training"""
    # Create a copy to avoid modifying the original
    model_qat = copy.deepcopy(model)
    model_qat.eval()
    
    # Set quantization configuration
    model_qat.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare the model for QAT
    model_prepared = torch.quantization.prepare_qat(model_qat, inplace=False)
    
    return model_prepared

def qat_training_loop(model_qat, train_loader, val_loader, criterion, optimizer, epochs, device):
    """QAT training loop"""
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model_qat.train()
        epoch_loss = 0.0
        
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            y_pred = model_qat(batch_inputs)
            loss = criterion(y_pred, batch_labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model_qat.eval()
        val_epoch_loss = 0.0
        
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                val_outputs = model_qat(val_inputs)
                val_loss = criterion(val_outputs, val_labels.long())
                val_epoch_loss += val_loss.item()
        
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"QAT Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return model_qat, train_losses, val_losses

def finalize_quantized_model(model_qat):
    """Convert QAT model to quantized model"""
    model_qat.eval()
    quantized_model = torch.quantization.convert(model_qat, inplace=False)
    return quantized_model

# Main QAT workflow
def implement_qat(state_dict_path, train_loader, val_loader, class_weights, device, qat_epochs=30):
    """Complete QAT implementation"""
    
    # 1. Load the pre-trained FP32 model
    model_fp32 = NeuralNetwork()
    model_fp32.load_state_dict(torch.load(state_dict_path, map_location=device))
    model_fp32.to(device)
    print("✅ Loaded pre-trained FP32 model")
    
    # 2. Prepare model for QAT
    model_qat = prepare_model_for_qat(model_fp32)
    model_qat.to(device)
    print("✅ Prepared model for QAT")
    
    # 3. Setup training components
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
    
    print(f"🚀 Starting QAT training for {qat_epochs} epochs...")
    
    # 4. QAT Training
    model_qat_trained, train_losses, val_losses = qat_training_loop(
        model_qat, train_loader, val_loader, criterion, optimizer, qat_epochs, device
    )
    
    # 5. Convert to quantized model
    quantized_model = finalize_quantized_model(model_qat_trained)
    print("✅ Converted to quantized INT8 model")
    
    return quantized_model, model_qat_trained, train_losses, val_losses

# Usage example (add this to your existing code):
"""
# After loading your data loaders and class_weights:
quantized_model, qat_model, qat_train_losses, qat_val_losses = implement_qat(
    state_dict_path='model_state_dict33.pth',
    train_loader=train_loader,
    val_loader=val_loader, 
    class_weights=class_weights,
    device=device,
    qat_epochs=30
)

# Save the quantized model
torch.save(quantized_model.state_dict(), 'model_qat_int8.pth')

# Test the quantized model
print("\\nQAT INT8 Model Performance:")
testing(quantized_model, torch.device('cpu'), test_loader)  # QAT models typically run on CPU

# Measure latency 
print("\\nLatency - QAT INT8 model:")
measure_latency(quantized_model, test_loader, torch.device('cpu'))
"""
