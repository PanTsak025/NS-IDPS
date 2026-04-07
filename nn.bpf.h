#ifndef NN_BPF_H
#define NN_BPF_H
#include <stdint.h>
#define Q (1 << 16)
#define SCALE_BITS 16

struct NeuralNetwork {
    // Weights (INT8)
    int8_t w_layer0[169];  // 13x13 
    int8_t w_layer1[169];  // 13x13 
    int8_t w_layer2[26];   // 13x2  
    
    // Layer 0 (13x13) Quantization
    int32_t layer0_scales[13];      
    int32_t layer0_zp[13];          
    
    // Layer 1 (13x13) Quantization  
    int32_t layer1_scales[13];     
    int32_t layer1_zp[13];        
    
    // Layer 2 (13x2) Quantization
    int32_t layer2_scales[2];      
    int32_t layer2_zp[2];         
    
    // Normalization (Q16.16)
    int64_t mean[13];               
    int64_t std[13];                
};

struct    // will be accessed from user space too for hot updating
{
    __uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, __u32);
	__type(value, struct NeuralNetwork);
	__uint(max_entries, 2);              // 1 for idle and 1 for running
	__uint(pinning, LIBBPF_PIN_BY_NAME); // <- pin
}nn_params SEC(".maps");

struct
{
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, __u32);
	__type(value, int32_t);
	__uint(max_entries, 1);              // 0 for idle, 1 for running,checks to avoid locks.
	__uint(pinning, LIBBPF_PIN_BY_NAME); // <- pin
}nn_index SEC(".maps");

struct norm_params                    //struct to save normialization parameters for preprocessing
{                                    
    const int64_t *no_n_x;
    const int8_t *x;
    int8_t *q_x;
    const int64_t *mean;
    const int64_t *std;
    const int32_t *scales;
    const int32_t *zps;
    int N;
    int M;
    const int8_t* w;
};
static __always_inline void normalize(struct norm_params *params)
{
    #pragma clang loop unroll(full)
    for (int i = 0; i < 13; i++) 
    {
        if (i >= params->N) break;

            //skip normalization for binary feature s
        if (i == 10 || i == 11 || i == 12) 
        {
            params->q_x[i] = params->no_n_x[i];  // direct copy, 0 or 1
            //bpf_printk("Feature[%d] (binary): raw=%d normalized=%d\n", i, params->no_n_x[i], params->q_x[i]);   //debugging
            continue;
        }
        int64_t safe_std = params->std[i] != 0 ? (int64_t)params->std[i] : 1;
        int32_t safe_scale = params->scales[i] != 0 ? (int32_t)params->scales[i] : 1;

        int64_t normalized = ((int64_t)(params->no_n_x[i] - params->mean[i]) * Q) / (uint64_t)safe_std;

        params->q_x[i] = (int8_t)(((int64_t)((uint64_t)normalized / safe_scale)) + params->zps[i]);
        //bpf_printk("Feature[%d]: raw=%d normalized=%d\n", i, params->no_n_x[i], params->q_x[i]);       //debugging
    }
}

static __always_inline void linear_relu(struct norm_params *params)       // merged input-INNER layer linear-relu
{
    #pragma clang loop unroll(full)
    for (int i = 0; i < params->M; i++) 
    {
        int32_t acc = 0;
        
        //matrix multiply (int8 x int8 -> int32)
        #pragma clang loop unroll(full)
        for (int j = 0; j < params->N; j++) 
        {
            acc += (int32_t)params->x[j] * (int32_t)params->w[i * params->N + j];
        }
        
        //Fixed-point scaling (Q16.16)
        int32_t res = (acc * params->scales[i]) >> 16;
        int8_t q_res = (int8_t)(res + params->zps[i]);
        
        //Fused ReLU
        params->q_x[i] = q_res > 0 ? q_res : 0;
    }
}

static __always_inline void linear(struct norm_params *params) // linear OUTPUT layer
{
    #pragma clang loop unroll(full)
    for (int i = 0; i < params->M; i++)  // Output 2 dimensions
    {
        int32_t acc = 0;
        
        //Matrix multiply (int8 x int8 -> int32)
        #pragma clang loop unroll(full)
        for (int j = 0; j < params->N; j++)  // Input dim (13)
        {
            acc += (int32_t)params->x[j] * (int32_t)params->w[i * params->N + j];
        }
        
        //Fixed-point scaling (Q16.16)
        int32_t res = (acc * params->scales[i]) >> 16;

        
        //Quantize and add zero point
        params->q_x[i] = (int8_t)(res + params->zps[i]);
    }
}
#endif
