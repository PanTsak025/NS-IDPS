# NS-IDPS
Enhanced NN-based Intrusion Detection &amp; Prevention System

  520  clang -O2 -target bpf -I/usr/lib/linux/uapi/ -I/usr/include/x86_64-linux-gnu/ -c -g xdp.c -o xdp.ebpf.o\n
  521  sudo bpftool prog load xdp.ebpf.o /sys/fs/bpf/nn\n
  522  sudo bpftool prog list\n
  523  sudo bpftool net attach xdp id 95 dev lo
  524  sudo cat /sys/kernel/debug/tracing/trace_pipe \n
  525  sudo bpftool net detach xdp dev lo
  526  sudo rm /sys/fs/bpf/nn

  


