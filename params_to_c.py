import numpy as np
import torch

def array_to_c(arr, name, dtype="int32_t"):
    """Convert numpy array to C-style string."""
    arr_str = ", ".join(map(str, arr.flatten().astype(int)))
    return f"static const {dtype} {name}[{len(arr)}] = {{{arr_str}}};\n"

# Load the stats and weights
data = torch.load('model_norm_stats.pth', weights_only=False)

mean = data['mean']
scale = data['scale']
state_dict = data['state_dict']

FIXED_POINT = 16

mean_fixed = (mean * (2 ** FIXED_POINT)).round().astype(np.int64)
scale_fixed = (scale * (2 ** FIXED_POINT)).round().astype(np.int64)

with open('nn_params.bpf.h', 'w') as f:
    f.write('#ifndef MLP_PARAMS_BPF_H\n')
    f.write('#define MLP_PARAMS_BPF_H\n\n')
    f.write(f'#define FIX_POINT {FIXED_POINT}\n\n')

    # Loop through layers and dump quantized weights
    for i, (name, param) in enumerate(state_dict.items()):
        if 'weight' in name:
            layer_idx = name.split('.')[0][-1]  # Extract layer index
            f.write(f'#define N{layer_idx} {param.shape[0]}\n')

            # Get quantized int8 weights
            weights = param.int_repr().numpy().flatten()

            # Write to header file
            c_array = array_to_c(weights, f'layer_{layer_idx}_weight', 'int8_t')
            f.write(c_array)

            # Also print to stdout
            print(f"\n// Quantized weights for layer {layer_idx}")
            print(c_array)

    # Write normalization parameters
    f.write("\n// Normalization parameters\n")
    f.write(array_to_c(mean_fixed, "mean", "int64_t"))
    f.write(array_to_c(scale_fixed, "scale", "int64_t"))

    f.write("\n#endif\n")

# Final message
print("\nGenerated nn_params.bpf.h for eBPF with quantized weight arrays printed.")
