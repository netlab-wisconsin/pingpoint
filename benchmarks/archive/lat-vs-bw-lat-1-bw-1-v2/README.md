1 K1 "lat" to local/remote
1 K2 "bw" on same xcd with "lat" to local/remote
bpx is scaled in a loop
tpb is configured at compile time
fused kernel approach
set LEN param appropriately for K1 to access either LLC or HBM

(01/25/25) Notes
- bytes 계산 needs fix
- chunk idx 계산 로직 needs fix (w.r.t. tpb) 
- K1 1<<22 이상일 때 illegal memaccess error나는 버그 있음. 그 이하는 괜찮은데 고쳐야함
Due to integer overflow; const size_t cycles_index = (size_t)xcc_id * n_dtypes + index; (k1.h L94)