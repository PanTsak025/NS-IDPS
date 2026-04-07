#include <bpf/bpf.h>
#include <stdio.h>
#include "nn_params.bpf.h"

struct NeuralNetwork {
    // Weights
    int8_t w_layer0[100];  // 10x10
    int8_t w_layer1[100];  // 10x10
    int8_t w_layer2[20];   // 10x2
    
    // Layer 0 (10x10)
    int32_t layer0_scales[10];
    int32_t layer0_zp[10];
    
    // Layer 1 (10x10)  
    int32_t layer1_scales[10];
    int32_t layer1_zp[10];
    
    // Layer 2 (10x2)
    int32_t layer2_scales[2];
    int32_t layer2_zp[2];
    
    // Normalization (Q16.16)
    int64_t mean[10];
    int64_t std[10];
};

int main()
{
    const char *nn_parameters_path = "/sys/fs/bpf/nn_params";
    const char *nn_idx_path = "/sys/fs/bpf/nn_index";

    int nn_parameters_fd = bpf_obj_get(nn_parameters_path);
    int nn_idx_fd = bpf_obj_get(nn_idx_path);

    int zero_idx = 0;
    int nn_idx = 0;

    int err = bpf_map_lookup_elem(nn_idx_fd, &zero_idx, &nn_idx);
    printf("load nn index error: %s\n", strerror(err));
    printf("NN index: %d\n", nn_idx);

    int new_nn_idx = (nn_idx + 1) % 2;
    struct NeuralNetwork net;
       // Copy parameters from header to struct
       memcpy(net.mean, feature_mean, sizeof(net.mean));
       memcpy(net.std, feature_std, sizeof(net.std));
       
       // Layer 0 (13x13)
       memcpy(net.w_layer0, layer1_weights, sizeof(net.w_layer0));
       memcpy(net.layer0_scales, layer1_scales, sizeof(net.layer0_scales));
       memcpy(net.layer0_zp, layer1_zero_points, sizeof(net.layer0_zp));
       
       // Layer 1 (13x13)
       memcpy(net.w_layer1, layer2_weights, sizeof(net.w_layer1));
       memcpy(net.layer1_scales, layer2_scales, sizeof(net.layer1_scales));
       memcpy(net.layer1_zp, layer2_zero_points, sizeof(net.layer1_zp));
       
       // Layer 2 (13x2)
       memcpy(net.w_layer2, layer3_weights, sizeof(net.w_layer2));
       memcpy(net.layer2_scales, layer3_scales, sizeof(net.layer2_scales));
       memcpy(net.layer2_zp, layer3_zero_points, sizeof(net.layer2_zp));

       printf("Loading new NN index: %d\n", new_nn_idx);
   
       // Update BPF maps
       err = bpf_map_update_elem(nn_parameters_fd, &new_nn_idx, &net, BPF_ANY);
       if (err) {
           printf("Failed to update NN parameters: %s\n", strerror(err));
           return 1;
       }
   
       err = bpf_map_update_elem(nn_idx_fd, &zero_idx, &new_nn_idx, BPF_ANY);
       if (err) {
           printf("Failed to update NN index: %s\n", strerror(err));
           return 1;
       }
   
       printf("Successfully updated NN parameters!\n");
       return 0;
   }