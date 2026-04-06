#include <linux/bpf.h>
#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include "flow.h"
#include "nn.bpf.h"
#define TIMEOUT_FLOW  1000000000ULL // 1 seconds
#define BLOCK_SECONDS 10000000000ULL // 10 seconds  
struct
{
    __uint(type, BPF_MAP_TYPE_PROG_ARRAY);
    __uint(max_entries, 1024);
    __type(key, __u32);              
    __type(value, __u32);
} progs SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);         // src IP
    __type(value, __u64);       // timestamp (in ns)
} blacklist SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 2); // Only two labels: benign (0) and malicious (1), (2) total ns, (3) packets extraction time, (4) all packets
    __type(key, __u32);
    __type(value, __u64);
} label_counters SEC(".maps");
static inline int get_key(struct xdp_md *ctx,  struct flow_key *key)
{
    void *data_start = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    if (data_start + sizeof(struct ethhdr) > data_end) {
        bpf_printk("eth hdr too short\n");
        return -1;
    }
    struct ethhdr *eth = data_start;
    if (eth->h_proto != bpf_htons(ETH_P_IP)) {
        bpf_printk("not IPv4\n");
        return -1;
    }
    struct iphdr *ip = data_start + sizeof(struct ethhdr);
    if ((void *)ip + sizeof(struct iphdr) > data_end) {
        bpf_printk("IP hdr too short\n");
        return -1;
    }
    __u32 ip_header_len = ip->ihl * 4;
    if (ip_header_len < sizeof(struct iphdr) ||
        data_start + sizeof(struct ethhdr) + ip_header_len > data_end) {
        bpf_printk("invalid IP hdr len\n");
        return -1;
    }
    if (ip->protocol != 6) {
        bpf_printk("not TCP\n");
        return -1;
    }
    struct tcphdr *tcp = (void *)ip + ip_header_len;
    if ((void *)tcp + sizeof(struct tcphdr) > data_end) {
        bpf_printk("TCP hdr too short\n");
        return -1;
    }
    __u32 tcp_header_len = tcp->doff * 4;
    if (tcp_header_len < sizeof(struct tcphdr) ||
        (void *)tcp + tcp_header_len > data_end) {
        bpf_printk("invalid TCP hdr len\n");
        return -1;
    }
    key->source_ip = ip->saddr;                   
    key->destination_ip = ip->daddr;
    key->source_port = bpf_ntohs(tcp->source);         
    key->destination_port = bpf_ntohs(tcp->dest);     
    key->protocol = ip->protocol;
    return 0;
}
SEC("xdp")
int xdp_nn(struct xdp_md *ctx)
{
    void *data_start = (void *)(long)ctx->data;   //start of network packet
    void *data_end = (void *)(long)ctx->data_end; //end of network packet
    struct ethhdr *eth = data_start;
    struct iphdr *ip = data_start + sizeof(struct ethhdr);

    if(data_start + sizeof(struct ethhdr) > data_end)   //verifier requirement
    {
        bpf_printk(" eth header too short\n");
        return XDP_ABORTED;
    }

    if ((void *)ip + sizeof(struct iphdr) > data_end) 
    {
        bpf_printk("IP header too short\n");
        return XDP_ABORTED;
    }
    __u32 ip_header_len = ip->ihl * 4;                     //invalid header length
    if (data_start + sizeof(struct ethhdr) + ip_header_len > data_end) {
        bpf_printk("Invalid IP header length\n");
        return XDP_ABORTED;
    }
    // Block all 127.x.x.x traffic
    if ((ip->saddr & 0xFF000000) == 0x7F000000) {
        bpf_printk("Blocking loopback traffic\n");
        return XDP_DROP;
    }

    if(ip->protocol == 1) // ICMP
    {
        //bpf_printk("ICMP packet\n");
        return XDP_DROP;
    }
    if(ip->protocol == 17) // UDP
    {
        //bpf_printk("UDP packet\n");
        return XDP_DROP;
    }
    if(ip->protocol == 6) // TCP
    {
        struct tcphdr *tcp = (void *)ip + ip_header_len;
        // if (tcp->syn && tcp->fin) 
        // {
        //     return XDP_DROP;  // SYN-FIN scan
        // }
        // if (!(tcp->syn || tcp->fin || tcp->rst || tcp->ack || tcp->psh || tcp->urg)) 
        // {
        //     return XDP_DROP;  // NULL scan
        // }
        // if (tcp->fin && tcp->psh && tcp->urg)
        // {
        //     return XDP_DROP;  // Xmas scan
        // }
        __u32 src_ip = ip->saddr;
        __u64 *time = bpf_map_lookup_elem(&blacklist, &src_ip);
        __u64 now = bpf_ktime_get_ns();

        if(time) 
        {
            if ((now - *time) < BLOCK_SECONDS) 
            {
                __u32 labelkeyy = 1;
                bpf_printk("PACKET BLOCKED\n");
                __u64 *count = bpf_map_lookup_elem(&label_counters, &labelkeyy);
                if (count) 
                {
                    __sync_fetch_and_add(count, 1); 
                }
                return XDP_DROP;
            } 
            else 
            {
                // bpf_printk("DELETING LIST\n");
                bpf_map_delete_elem(&blacklist, &src_ip);
            }
        }

        if ((void *)tcp + sizeof(struct tcphdr) > data_end) {
            bpf_printk("TCP header too short\n");
            return XDP_ABORTED;
        }

        __u32 tcp_header_len = tcp->doff * 4;
        if (tcp_header_len < sizeof(struct tcphdr)) {
            bpf_printk("Invalid TCP header length\n");
            return XDP_ABORTED;
        }

        if ((void *)tcp + tcp_header_len > data_end) {
            bpf_printk("TCP header exceeds packet bounds\n");
            return XDP_ABORTED;
        }
        __u64 packet_time = bpf_ktime_get_ns();
        __u16 ip_total_len = bpf_ntohs(ip->tot_len);
        __u64 packet_length = (__u64)(ip_total_len - ip_header_len);
        __u64 segment_size = (__u64)(ip_total_len - ip_header_len - tcp_header_len);
        //bpf_printk("\nSegment size: %u\n", segment_size);
        struct flow_key key = {
            .source_ip = ip->saddr,                        
            .destination_ip = ip->daddr,
            .source_port = bpf_ntohs(tcp->source),         
            .destination_port = bpf_ntohs(tcp->dest),     
            .protocol = ip->protocol
        };
        
        struct flow_value *value = bpf_map_lookup_elem(&flows_map, &key);
        int is_new_flow = 0;
        int16_t malicious_ports[] = {22, 21, 23, 445, 3389, 5900, 135, 
            1433, 1900,2323,4444,6667,31337,12345,69};

        if(!value)
        {
            struct flow_value new_values = {};
            new_flow(&new_values);
            new_values.first_seen = packet_time;
            new_values.last_seen = new_values.first_seen;
            new_values.dst_port = bpf_ntohs(tcp->dest);
            if (new_values.dst_port == 22   || new_values.dst_port == 21    ||
                new_values.dst_port == 23   || new_values.dst_port == 445   || 
                new_values.dst_port == 3389 || new_values.dst_port == 5900  ||
                new_values.dst_port == 135  || new_values.dst_port == 1433  ||
                new_values.dst_port == 1900 || new_values.dst_port == 2323  ||
                new_values.dst_port == 4444 || new_values.dst_port == 6667  ||
                new_values.dst_port == 31337|| new_values.dst_port == 12345 ||
                new_values.dst_port == 69) 
            {
                new_values.rule_1_mal_ports = 1;
            } 
            else 
            {
                new_values.rule_1_mal_ports = 0;
            }

            new_values.init_win = bpf_ntohs(tcp->window);
            //bpf_printk("Initial TCP window size: %u", new_values.init_win);
            //bpf_printk("New flow: src=%x spt=%u dst=%x dpt=%u\n",
           //     key.source_ip, key.source_port,
             //   key.destination_ip, key.destination_port);
            if (bpf_map_update_elem(&flows_map, &key, &new_values, BPF_NOEXIST)) 
            {
                return XDP_PASS;
            }
            //bpf_printk("Flow added to map\n");

            is_new_flow = 1;
        
            value = bpf_map_lookup_elem(&flows_map, &key);
            if (!value) {
                return XDP_PASS;
            }
        }
        update_flow(value, packet_length, packet_time, tcp, segment_size);
        //bpf_printk("Flow updated: pkts=%u bytes=%u\n", value->pkt_count, value->total_bytes);

        __u64 ddos_now = bpf_ktime_get_ns();

        if (tcp->fin || tcp->rst) 
        {
                if (value->seg_size_min <= 20)
                {
                    value->rule_2_small_seg_size = 1;
                }
                else
                {
                    value->rule_2_small_seg_size = 0;
                }

                if(value->pkt_count <= 3)
                {
                    value->rule_3_low_pack_count = 1;
                }
                else
                {
                    value->rule_3_low_pack_count = 0;
                }
                // bpf_printk("src:%x dst:%x spt:%u dpt:%u", key.source_ip, key.destination_ip, key.source_port, key.destination_port);
                // bpf_printk("FLOW dst:%hu pkts:%u bytes:%u\n",
                //     value->dst_port, value->pkt_count, value->total_bytes);
                
                // bpf_printk("min_pkt:%u max_pkt:%u\n",
                //     value->min_pkt_len, value->max_pkt_len);
                
                // bpf_printk("seg_min:%llu psh:%hhu\n",
                //     value->seg_size_min, value->psh_flags);
                
                // bpf_printk("iat_min:%llu iat_max:%llu, rule1: %u\n",
                //     value->iat_stats.min, value->iat_stats.max, value->rule_1_mal_ports);
                
                // bpf_printk("Flow %s: %s",
                //      (tcp->fin || tcp->rst) ? "ended by FIN/RST" : "timed out",
                //      (ddos_now - value->last_seen > TIMEOUT_FLOW) ? "flow timed out" : "normal");
                __u32 idx = 0;
                int32_t *idx_ptr = bpf_map_lookup_elem(&nn_index, &idx);
                //bpf_printk("POINTER :%d\n",idx_ptr);

                if(idx_ptr && value)
                {
                    value->nn_index = *idx_ptr;
                    value->detection_start_time = bpf_ktime_get_ns();
                    bpf_tail_call(ctx, &progs, 0);
                }

                // int32_t idx = 0;
                // int32_t *idx_ptr = bpf_map_lookup_elem(&nn_index, &idx);
                // struct flow_value *attr_ptr = (struct flow_value *)bpf_map_lookup_elem(&flows_map, &key);

                // if (idx_ptr && attr_ptr)
                // {
                //     attr_ptr->nn_index = *idx_ptr;
                //     bpf_tail_call(ctx, &progs, 0);
                // }
                
        } 
        return XDP_PASS;

    }
    //bpf_printk("packet passed\n");
    return XDP_PASS;
}
SEC("xdp")
int xdp_live_feature_preprocessing(struct xdp_md *ctx)
{
    
    struct flow_key key = {};
    if(get_key(ctx, &key)  < 0)
    {
        bpf_printk("failed0\n");
        return XDP_PASS;
    }
    struct flow_value *value = (struct flow_value *) bpf_map_lookup_elem(&flows_map, &key);
    if(!value)
    { 
        bpf_printk("failed1\n");
        return XDP_PASS;
    }
    struct NeuralNetwork *nn = bpf_map_lookup_elem(&nn_params, &(value->nn_index)); //FETCH THE RUNNING PARAMS
    if(!nn)
    {
        bpf_printk("failed2\n");
        return XDP_PASS;
    }
    int64_t features[13] = 
    {
        value->dst_port,
        value->init_win,
        value->seg_size_min,
        value->max_pkt_len,
        value->total_bytes,
        value->header_length,
        value->iat_stats.min,
        value->pkt_count,
        value->iat_stats.max,
        value->min_pkt_len,
        value->rule_1_mal_ports,
        value->rule_2_small_seg_size,
        value->rule_3_low_pack_count
    };
    struct norm_params p = {
        .no_n_x = features,
        .q_x = value->normalized_weights,
        .mean = nn->mean,
        .std = nn->std,
        .scales = nn->layer0_scales,
        .zps = nn->layer0_zp,
        .N = 13
    };
    //value->detection_start_time = bpf_ktime_get_ns();
    normalize(&p);
    // bpf_printk("Norm weights: %d %d %d %d", 
    //     value->normalized_weights[0],
    //     value->normalized_weights[1], 
    //     value->normalized_weights[2],
    //     value->normalized_weights[3]);

    // bpf_printk("           : %d %d %d %d",
    //     value->normalized_weights[4],
    //     value->normalized_weights[5],
    //     value->normalized_weights[6],
    //     value->normalized_weights[7]);

    // bpf_printk("           : %d %d %d %d",
    //     value->normalized_weights[8],
    //     value->normalized_weights[9],
    //     value->normalized_weights[10],
    //     value->normalized_weights[11]);

    // bpf_printk("           : %d",
    //     value->normalized_weights[12]);

    bpf_tail_call(ctx, &progs, 1);
    return XDP_PASS;
}


SEC("xdp")
int xdp_layer_0(struct xdp_md *ctx)
{
    struct flow_key key = {};
    if(get_key(ctx, &key)  < 0)
    {
        bpf_printk("failed0\n");
        return XDP_PASS;
    }
    struct flow_value *value = (struct flow_value *) bpf_map_lookup_elem(&flows_map, &key);
    if(!value)
    { 
        bpf_printk("failed1\n");
        return XDP_PASS;
    }

    struct NeuralNetwork *nn = bpf_map_lookup_elem(&nn_params, &(value->nn_index)); //FETCH THE RUNNING PARAMS
    if(!nn)
    {
        bpf_printk("failed2\n");
        return XDP_PASS;
    }
    struct norm_params p = 
    {
        .x = value->normalized_weights,
        .q_x = value->layer_output,
        .scales = nn->layer0_scales,
        .zps = nn->layer0_zp,
        .N = 13,
        .M = 13,
        .w = nn->w_layer0
    };
    linear_relu(&p);
    // bpf_printk("Layer 0 output: %d %d %d %d", 
    //     value->layer_output[0],
    //     value->layer_output[1], 
    //     value->layer_output[2],
    //     value->layer_output[3]);

    // bpf_printk("           : %d %d %d %d",
    //     value->layer_output[4],
    //     value->layer_output[5],
    //     value->layer_output[6],
    //     value->layer_output[7]);

    // bpf_printk("           : %d %d %d %d",
    //     value->layer_output[8],
    //     value->layer_output[9],
    //     value->layer_output[10],
    //     value->layer_output[11]);

    // bpf_printk("           : %d",
    //     value->layer_output[12]);

    bpf_tail_call(ctx, &progs, 2);
    return XDP_PASS;
}

SEC("xdp")
int xdp_layer_1(struct xdp_md *ctx)
{
    struct flow_key key = {};
    if(get_key(ctx, &key)  < 0)
    {
        bpf_printk("failed0\n");
        return XDP_PASS;
    }
    struct flow_value *value = (struct flow_value *) bpf_map_lookup_elem(&flows_map, &key);
    if(!value)
    { 
        bpf_printk("failed1\n");
        return XDP_PASS;
    }

    struct NeuralNetwork *nn = bpf_map_lookup_elem(&nn_params, &(value->nn_index)); //FETCH THE RUNNING PARAMS
    if(!nn)
    {
        bpf_printk("failed2\n");
        return XDP_PASS;
    }
    struct norm_params p = 
    {
        .x = value->layer_output,
        .q_x = value->normalized_weights,
        .scales = nn->layer1_scales,
        .zps = nn->layer1_zp,
        .N = 13,
        .M = 13,
        .w = nn->w_layer1
    };
    linear_relu(&p);
    // bpf_printk("Layer 1 output: %d %d %d %d", 
    //     value->normalized_weights[0],
    //     value->normalized_weights[1], 
    //     value->normalized_weights[2],
    //     value->normalized_weights[3]);

    // bpf_printk("           : %d %d %d %d",
    //     value->normalized_weights[4],
    //     value->normalized_weights[5],
    //     value->normalized_weights[6],
    //     value->normalized_weights[7]);

    // bpf_printk("           : %d %d %d %d",
    //     value->normalized_weights[8],
    //     value->normalized_weights[9],
    //     value->normalized_weights[10],
    //     value->normalized_weights[11]);

    // bpf_printk("           : %d",
    //     value->normalized_weights[12]);

    bpf_tail_call(ctx, &progs, 3);
    return XDP_PASS;
}


SEC("xdp")
int xdp_layer_2(struct xdp_md *ctx)
{
    struct flow_key key = {};
    if(get_key(ctx, &key)  < 0)
    {
        bpf_printk("failed0\n");
        return XDP_PASS;
    }
    struct flow_value *value = (struct flow_value *) bpf_map_lookup_elem(&flows_map, &key);
    if(!value)
    { 
        bpf_printk("failed1\n");
        return XDP_PASS;
    }

    struct NeuralNetwork *nn = bpf_map_lookup_elem(&nn_params, &(value->nn_index)); //FETCH THE RUNNING PARAMS
    if(!nn)
    {
        bpf_printk("failed2\n");
        return XDP_PASS;
    }

    struct norm_params p = 
    {
        .x = value->normalized_weights,
        .q_x = value->layer_output,
        .scales = nn->layer2_scales,
        .zps = nn->layer2_zp,
        .N = 13,
        .M = 2,
        .w = nn->w_layer2
    };
    linear(&p);
    __u64 det_time = bpf_ktime_get_ns() - value->detection_start_time;
    int diff = value->layer_output[1] - value->layer_output[0];
    int label = diff > 53;  // e.g. 53 for 0.7085299 threshold

    __u32 labelkey = label ? 1 : 0;
    __u64 *count = bpf_map_lookup_elem(&label_counters, &labelkey);
    if (count) 
    {
        __sync_fetch_and_add(count, 1); 
    }
    if (label) 
    {
        __u32 src_ip = key.source_ip; 
        __u64 now = bpf_ktime_get_ns();
        bpf_map_update_elem(&blacklist, &src_ip, &now, BPF_ANY);
        //bpf_printk("ADDDED TO BLACKLIST\n");
    }
    // bpf_printk("Classification: %s (Confidence: %d vs %d), detection time: %lld ns",
    //         label ? "MALICIOUS" : "BENIGN",
    //         value->layer_output[0],
    //         value->layer_output[1],
    //         det_time);

    // __u32 key2 = 2;
    // __u64 *total_time = bpf_map_lookup_elem(&label_counters, &key2);
    // if (total_time) 
    // {
    //     __sync_fetch_and_add(total_time, det_time);
    // }     
    
    // __u32 key3 = 3;
    // __u64 *total_pack_time = bpf_map_lookup_elem(&label_counters, &key3);
    // if (total_pack_time) 
    // {
    //     __sync_fetch_and_add(total_pack_time, value->feat_extraction_time);
    // }  
    // __u32 key4 = 4;
    // __u64 *total_packets = bpf_map_lookup_elem(&label_counters, &key4);
    // if (total_packets) 
    // {
    //     __sync_fetch_and_add(total_packets, value->pkt_count);
    // }  

    bpf_map_delete_elem(&flows_map, &key);
    return XDP_PASS;
}
//     "Destination Port",
//     "Init_Win_bytes_forward", 
//     "min_seg_size_forward",
//     " Fwd Packet Length Max", 
//     "Subflow Fwd Bytes",
//     "Fwd Header Length.1",
//     "Fwd IAT Min", 
//     "Subflow Fwd Packets", 
//     "Fwd IAT Max",
//     "Fwd IAT Total",
//     "Fwd PSH Flags",
//     "Fwd Packet Length Min",
char LICENSE[] SEC("license") = "GPL";