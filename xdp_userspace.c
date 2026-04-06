#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <signal.h>
#include <unistd.h>
#include <net/if.h>
#include "xdp.skel.h"

struct bpf_progs_desc {
    char name[256];
    enum bpf_prog_type type;
    int map_prog_idx;
    struct bpf_program *prog;
};
static struct bpf_progs_desc progs[] = {
    {"xdp_live_feature_preprocessing", BPF_PROG_TYPE_XDP, 0, NULL},
    {"xdp_layer_0", BPF_PROG_TYPE_XDP, 1, NULL},
    {"xdp_layer_1", BPF_PROG_TYPE_XDP, 2, NULL},
    {"xdp_layer_2", BPF_PROG_TYPE_XDP, 3, NULL}
};
static volatile bool exiting = false;

static void sig_handler(int sig) {
    exiting = true;
}

int main(int argc, char **argv) {
    const char *interface = "vboxnet0"; 
    unsigned int ifindex = if_nametoindex(interface);
    struct xdp_ebpf *skel = NULL;
    struct bpf_link *link = NULL;
    int map_progs_fd, prog_count;
    int err = 0;

    if (!ifindex) {
        fprintf(stderr, "Failed to get ifindex for %s\n", interface);
        return 1;
    }

    // Set up signal handler
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Open and load BPF skeleton
    skel = xdp_ebpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = xdp_ebpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton\n");
        goto cleanup;
    }

    // Initialize prog array map
    map_progs_fd = bpf_map__fd(skel->maps.progs);
    prog_count = sizeof(progs) / sizeof(progs[0]);

    for (int i = 0; i < prog_count; i++) {
        progs[i].prog = bpf_object__find_program_by_name(skel->obj, progs[i].name);
        if (!progs[i].prog) {
            fprintf(stderr, "Program %s not found\n", progs[i].name);
            err = -ENOENT;
            goto cleanup;
        }
        bpf_program__set_type(progs[i].prog, progs[i].type);
    }

    for (int i = 0; i < prog_count; i++) {
        int prog_fd = bpf_program__fd(progs[i].prog);
        __u32 key = progs[i].map_prog_idx;
        
        if (bpf_map_update_elem(map_progs_fd, &key, &prog_fd, BPF_ANY)) {
            fprintf(stderr, "Failed to update prog array\n");
            err = -EINVAL;
            goto cleanup;
        }
    }

    // update nn index
    __u32 nn_idx = 0;
    int32_t nn_idx_value = 1;

    err = bpf_map__update_elem(skel->maps.nn_index,
                            &nn_idx, sizeof(__u32),
                            &nn_idx_value, sizeof(int32_t),
                            BPF_ANY);
    if (err)
    {
        fprintf(stderr, "Error: updating nn index\n");
        return 1;
    }

    // Attach XDP program 
    link = bpf_program__attach_xdp(skel->progs.xdp_nn, ifindex);
    if (!link) {
        fprintf(stderr, "Failed to attach XDP program\n");
        err = -EINVAL;
        goto cleanup;
    }

    printf("Successfully loaded! Press Ctrl-C to exit\n");

    while (!exiting) {
        sleep(1);
    }

cleanup:
    // Proper cleanup
    if (link) {
        bpf_link__destroy(link);
    }
    if (skel) {
        xdp_ebpf__destroy(skel);
    }
    remove("/sys/fs/bpf/nn_params");
    remove("/sys/fs/bpf/nn_index");
    return err < 0 ? -err : 0;
}