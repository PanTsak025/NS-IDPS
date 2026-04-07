#ifndef FLOW_H
#define FLOW_H
#include <linux/bpf.h>
#include <stdint.h>
#include <bpf/bpf_helpers.h>
// XDP can only see Ingress traffic aka Forward.
// The inference will be fed with flows, not packets.
// A flow consists of 5 attributes. a 5-tuple.
struct flow_key {
    __u32 source_ip;
    __u32 destination_ip;
    __u16 source_port;
    __u16 destination_port;
    __u8 protocol;
} __attribute__((packed));


struct flow_value
{
    __u64 first_seen;       
    __u64 last_seen; 

    __u16 dst_port;         // Destination port
    __u16 init_win;         // Initial window size 
    __u32 min_pkt_len;      // Smallest packet seen
    __u32 max_pkt_len;      // Largest packet seen
    __u32 total_bytes;      // Sum of all packet lengths
    __u64 header_length;
    __u32 pkt_count;   
    __u64 seg_size_min;
    __u8 rule_1_mal_ports;
    __u8 rule_2_small_seg_size;
    __u8 rule_3_low_pack_count;
    struct 
    {
        __u64 min;
        __u64 max;
    } iat_stats;            // Inter-arrival time statistics
        
    __u64 detection_start_time;
    __u64 feat_extraction_time;

    int32_t nn_index;
    int8_t normalized_weights[13];
    int8_t layer_output[13];
};

struct 
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);           //standard limit of 8192 flows
    __type(key, struct flow_key);
    __type(value, struct flow_value);
}flows_map SEC(".maps");


static inline void new_flow(struct flow_value *values)       //initialize new flow
{
    values->first_seen = 0;
    values->last_seen = 0;
    values->header_length = 0;
    values->seg_size_min = 65535;
    values->pkt_count = 0;
    values->total_bytes = 0;
    values->max_pkt_len = 0;
    values->min_pkt_len = 0xFFFFFFFFU;
    values->dst_port = 0;
    values->init_win = 0;
    values->iat_stats.min = 0;
    values->iat_stats.max = 0;
    values->rule_1_mal_ports = 2;
    values->rule_2_small_seg_size = 2;
    values->rule_3_low_pack_count = 2;
}

static inline void update_flow(struct flow_value *values, __u64 packet_length, __u64 packet_time, struct tcphdr *tcp,__u64 segment_size)      //update flow
{
    __u64 start_extraction_time = bpf_ktime_get_ns();
    values->pkt_count++;
    values->total_bytes += packet_length;
    
    values->header_length += (tcp->doff * 4);

    if (packet_length > values->max_pkt_len)
    {
        values->max_pkt_len = packet_length;
    }
    if (packet_length < values->min_pkt_len)
    {
        values->min_pkt_len = packet_length;
    }

    if((values->seg_size_min == 0) && (segment_size > 0))
    {
        values->seg_size_min = segment_size;
    }
    if ((values->seg_size_min > 0) && (segment_size > 0)) //Skips ACKs/SYN/FIN
    {  
        if (segment_size < values->seg_size_min) 
        {
            values->seg_size_min = segment_size;
        }
    }
    if(values->pkt_count > 1)
    {
        __u64 iat = packet_time - values->last_seen;
        if (iat < values->iat_stats.min || values->pkt_count == 2) 
        {
            values->iat_stats.min = iat;
        }
        if (iat > values->iat_stats.max) 
        {
            values->iat_stats.max = iat;
        }
    }
    values->last_seen = packet_time;
    values->feat_extraction_time += bpf_ktime_get_ns() - start_extraction_time;
}

#endif
