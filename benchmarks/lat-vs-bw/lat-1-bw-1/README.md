1 K1 "lat" to local/remote
1 K2 "bw" on same xcd with "lat" to local/remote
bpx is scaled in a loop
tpb is configured at compile time
stream approach
cu binding to isolate lat CU from bw CUs
set LEN param appropriately for K1 to access either LLC or HBM

(Note 01/25/25) While removed, two K2 flows were instantiated to test YX routing's path overlap
- flow 1: K2_PINNED_XCD_1, K2_PINNED_HBM_1
- flow 2: K2_PINNED_XCD_2, K2_PINNED_HBM_2
 K1 was disabled, and K2 kernel preluded the two XCDs. k2_hbm was recognized using below code:
- `const uint32_t k2_hbm = (xcc_id == k2_xcd1) ? k2_hbm1 : k2_hbm2;`