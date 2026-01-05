#include "party3pc.h"
#include "rss_protocols.h"
#include "test_base.h"
#include "llmConfig.h"
#include <vector>
#include <string>
#include "dcf.h"
#include "dpf.h"
#include "fss.h"
#include <algorithm>
#include <omp.h>



static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = print_res ? 1 : 2;
static int warmup_rounds = print_res ? 0 : 1;
static constexpr size_t BLOCK_BYTES = 16;
static bool Verification = false;

static size_t bytes_for_bits(int bits){
    return (bits + 7) / 8;
}

static inline int conv_out_dim(int in, int kernel, int stride, int pad){
    return (in + 2 * pad - kernel) / stride + 1;
}

static inline int pool_out_dim(int in, int kernel, int stride, int pad){
    // same formula as conv
    return (in + 2 * pad - kernel) / stride + 1;
}


// Calculate DPF key size (in bytes)
// Parameters:
//  Bin: input domain bits
//  Bout: output/bucket bits
//  groupSize: number of groups (for g array length)
// For DPF, per user's instruction we count these fields as key material:
//  block *k0, block *k1   -> each length Bin+1, each block is 16 bytes
//  GroupElement *g         -> length groupSize, element size = bytes_for_bits(Bout)
//  std::array<std::array<block,2>,2> cs -> 4 blocks

static inline size_t dpf_key_size_bytes(int Bin, int Bout, int groupSize, bool Verification=false){
    size_t blocks_k = Bin + 1; // k0 and k1 send to 3 parties
    size_t bytes = blocks_k * BLOCK_BYTES;
    // g array: groupSize elements of Bout bits -> count bytes
    bytes += groupSize * bytes_for_bits(Bout);
    // r
    bytes += groupSize * bytes_for_bits(Bin);
    if (Verification) {
        //Bin * g and alpha check
        bytes += 2 * groupSize * bytes_for_bits(Bin);
        // cs: 2 blocks
        bytes += 2 * BLOCK_BYTES;
    }
    return bytes;
}

// Calculate DCF key size (in bytes)
// Fields to count (per user's instruction):
//  block *k0, *k1 -> each length Bin+1
//  GroupElement *g -> length groupSize
//  GroupElement *v -> length Bin * groupSize,Vcw
//  cs: 4 blocks

static inline size_t dcf_key_size_bytes(int Bin, int Bout, int groupSize,bool Verification=false){
    size_t blocks_k = Bin  * 2 + 1; // k0 and k1
    size_t bytes = blocks_k * BLOCK_BYTES;
    // g 
    bytes += groupSize * bytes_for_bits(Bout);
    // v 
    bytes += groupSize * (Bin - 1) * bytes_for_bits(Bin);
    // r
    bytes +=  2 * groupSize * bytes_for_bits(Bin);
    if (Verification) {
        //Bin * g and alpha check
        bytes += 2 * groupSize * bytes_for_bits(Bin);
        // cs: 2 blocks
        bytes += 2 * BLOCK_BYTES;
    }
    return bytes;
}

//bit_len = cmp_size,count = number of comparisons
static DPFKeyPack offline_pointcmp(int bit_len, size_t count,bool Verification=false) {
    auto &party = Party3PC::getInstance();
    // Use DPF key size (Bin = Bout = bit_len, groupSize = 1)
    size_t key_size = dpf_key_size_bytes(bit_len, bit_len, 1, Verification);
    size_t total_key_bytes = count * key_size;

    // Generate DPF keys `count` times to simulate offline generation cost.
    // Run serially to avoid large concurrent allocations that can cause OOM.
    #pragma omp parallel for num_threads(1)
    for (long long ii = 0; ii < (long long)count; ++ii) {
        size_t i = (size_t)ii;
        GroupElement idx(0, bit_len);
        GroupElement payload(0, bit_len);
        auto keys = keyGenDPF(bit_len, bit_len, idx, payload, GenParams(Verification));
        DPFKeyPack kp0 = std::get<0>(keys);
        DPFKeyPack kp1 = std::get<1>(keys);
        DPFKeyPack kp2 = std::get<2>(keys);
    }
    
    // Setup: Pb+2 sends keys to Pb and Pb+1
    // Send/recv in chunks to avoid allocating huge contiguous buffer
    {   
        
        const size_t CHUNK = (size_t)1 << 20; // 1MB
        std::vector<char> buf(std::min(CHUNK, total_key_bytes), 0);
        size_t rem = total_key_bytes;
        while (rem) {
            size_t t = std::min(rem, buf.size());
            // zero-copy: wrap existing buffer
            Tensor<char> send_t(buf.data(), {(uint32_t)t});
            Tensor<char> recv_t({(uint32_t)t});
            std::thread t1([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
            party.recv_tensor_from<char>(party.pre_party_id, recv_t);
            t1.join();
            rem -= t;
        }
        rem = total_key_bytes;
        while (rem) {
            size_t t = std::min(rem, buf.size());
            Tensor<char> send_t(buf.data(), {(uint32_t)t});
            Tensor<char> recv_t({(uint32_t)t});
            std::thread t1([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
            party.recv_tensor_from<char>(party.pre_party_id, recv_t);
            t1.join();
            rem -= t;
        }
    }
    // Generate one keypair to return a usable DPF key for evaluation.
    auto single_keys = keyGenDPF(bit_len, bit_len, GroupElement(0, bit_len), GroupElement(0, bit_len), GenParams(Verification));
    DPFKeyPack kp0 = std::get<0>(single_keys);
    if (Verification){
        size_t beta = 0;
        size_t alaha_check = 0;
        // 1.Full-domain evaluation to simulate verification cost
        // limit to single thread to reduce memory pressure
        #pragma omp parallel for num_threads(1) reduction(+:beta, alaha_check)
        for (size_t i = 0; i < count; ++i) {
            GroupElement res_local(0, bit_len);
            GroupElement idx(0, bit_len);
            for(size_t j = 0; j < (1 << bit_len); ++j){
                idx = GroupElement(j, bit_len);
                evalDPF(party.party_id, &res_local, idx, kp0,Verification);
            }
            beta += res_local.value;
            alaha_check += res_local.value * i;
        }
        auto _run_gpt2_end = std::chrono::steady_clock::now();
        // 2. Simulate hash verification cost and Open t, s 
        size_t hash_len = (1 << bit_len) * 2 * BLOCK_BYTES;
        {
            size_t t = hash_len + 2 * bytes_for_bits(bit_len);
            Tensor<char> send_t({(uint32_t)t});
            Tensor<char> recv_t({(uint32_t)t});
            std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
            party.recv_tensor_from<char>(party.pre_party_id, recv_t);
            st.join();
        }

    }
    // Return the first share as the representative key for eval calls.
    return kp0;
}

//bit_len = table_size
static void offline_lut(int bit_len, size_t count,bool Verification=false) {
    auto &party = Party3PC::getInstance();
    // 1. VDPF Keys 
    DPFKeyPack kp0 = offline_pointcmp(bit_len, count, Verification);
    
    if(not Verification){
    // Run serially to avoid memory spikes from parallel evalDPF
    #pragma omp parallel for num_threads(1)
        for (size_t i = 0; i < count; ++i) {
            GroupElement res_local(0, bit_len);
            GroupElement idx(0, bit_len);
            for(size_t j = 0; j < (1 << bit_len); ++j){
                evalDPF(party.party_id, &res_local, idx, kp0,Verification);
            }
        }
        freeDPFKeyPack(kp0);
    }
}

static void offline_dcf(int bit_len, size_t count,bool Verification=false, size_t extra_comm = 0) {
    auto &party = Party3PC::getInstance();
    // Use DPF key size (Bin = Bout = bit_len, groupSize = 1)
    size_t key_size = dcf_key_size_bytes(bit_len, bit_len, 1, Verification);
    size_t total_key_bytes = count * key_size + extra_comm;


    // Generate DCF keys `count` times to simulate offline generation cost.
    // Use OpenMP to parallelize key generation.
    // Run serially to avoid large concurrent allocations
    #pragma omp parallel for num_threads(1) 
    for (long long ii = 0; ii < (long long)count; ++ii) {
        size_t i = (size_t)ii;
        GroupElement idx(0, bit_len);
        GroupElement payload(0, bit_len);
        auto keys = keyGenDCF(bit_len, bit_len, idx, payload, GenParams(Verification));
        DCFKeyPack kp0 = std::get<0>(keys);
        DCFKeyPack kp1 = std::get<1>(keys);
        DCFKeyPack kp2 = std::get<2>(keys);
    }
    
    // Setup: Pb+2 sends keys to Pb and Pb+1
    // Send/recv in chunks to avoid allocating huge contiguous buffer
    {
        const size_t CHUNK = (size_t)1 << 20; // 4MB
        std::vector<char> buf(std::min(CHUNK, total_key_bytes), 0);
        size_t rem = total_key_bytes;
        while (rem) {
            size_t t = std::min(rem, buf.size());
            Tensor<char> send_t(buf.data(), {(uint32_t)t});
            Tensor<char> recv_t({(uint32_t)t});
            std::thread t1([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
            party.recv_tensor_from<char>(party.pre_party_id, recv_t);
            t1.join();
            rem -= t;
        }
        rem = total_key_bytes;
        while (rem) {
            size_t t = std::min(rem, buf.size());
            Tensor<char> send_t(buf.data(), {(uint32_t)t});
            Tensor<char> recv_t({(uint32_t)t});
            std::thread t1([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
            party.recv_tensor_from<char>(party.pre_party_id, recv_t);
            t1.join();
            rem -= t;
        }
    }

    
    if (Verification){
        auto single_keys = keyGenDCF(bit_len, bit_len, GroupElement(0, bit_len), GroupElement(0, bit_len), GenParams(Verification));
        DCFKeyPack kp0 = std::get<0>(single_keys);
        size_t beta = 0;
        size_t alaha_check = 0;
        // 1.Full-domain evaluation to simulate verification cost
        
        // limit verification loop threading to 1 to reduce memory usage
        #pragma omp parallel for num_threads(1)
            for (size_t i = 0; i < count; ++i) {
            GroupElement res_local(0, bit_len);
            GroupElement idx(0, bit_len);
            for(size_t j = 0; j < (1 << bit_len); ++j){
                idx = GroupElement(j, bit_len);
                evalDCF(party.party_id, &res_local, idx, kp0,Verification);
            }
            beta += res_local.value;
            alaha_check += res_local.value * i;
        }
        // 2. Simulate hash verification cost and Open t, s 
        size_t hash_len = (1 << bit_len) * 2 * BLOCK_BYTES;
        {
            size_t t = hash_len + 2 * bytes_for_bits(bit_len);
            Tensor<char> send_t({(uint32_t)t});
            Tensor<char> recv_t({(uint32_t)t});
            std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
            party.recv_tensor_from<char>(party.pre_party_id, recv_t);
            st.join();
        }
    }
}

static void offline_drelu(int bit_len, size_t count,bool Verification=false) {
    auto &party = Party3PC::getInstance();
    // 1. Shares of c,
    size_t share_bytes = count * bytes_for_bits(bit_len) * 2;
    offline_dcf(bit_len, count,Verification,share_bytes);
    // 2. Reveal delta_c, delta_y
    // if (Verification){
    //     size_t a_val=0, b_val=0;
    //     initFalconCompat();
    //     RSSVectorMyType a(count);
    //     RSSVectorSmallType theta(count);
    //     // Initialize a with some shares
    //     #pragma omp parallel for
    //     for(long long ii=0; ii<(long long)count; ++ii) {
    //         size_t i = (size_t)ii;
    //         a[i].first = a_val;
    //         a[i].second = b_val;
    //     }
    //     funcWrap(a, theta, count);
    //     // if (party.party_id == 0) std::cout << "funcWrap executed successfully." << std::endl;
    //     // verify delta_c, delta_y according to protocol in the image:
    //     RSSTensor<size_t> delta_c_shared({(uint32_t)count});
    //     RSSTensor<size_t> delta_y_shared({(uint32_t)count});
    //     #pragma omp parallel for
    //     for (long long ii = 0; ii < (long long)count; ++ii) {
    //         size_t i = (size_t)ii;
    //         delta_c_shared.first.data[i] = 1 - theta[i].first;
    //         delta_c_shared.second.data[i] = 1 - theta[i].second;
    //         delta_y_shared.first.data[i] = 1 - theta[i].first;
    //         delta_y_shared.second.data[i] = 1 - theta[i].second;
    //     }
    //     __int128 total_key_bytes_128 = ( __int128)2 * ( __int128)count * ( __int128)bytes_for_bits(bit_len);
    //     if (total_key_bytes_128 > ( __int128)std::numeric_limits<size_t>::max()){
    //     } else {
    //         size_t total_key_bytes = (size_t)total_key_bytes_128;
    //         const size_t CHUNK = (size_t)1 << 20; // 1MB
    //         std::vector<char> buf(std::min(CHUNK, total_key_bytes), 0);
    //         size_t rem = total_key_bytes;
    //         while (rem) {
    //             size_t t = std::min(rem, buf.size());
    //             Tensor<char> send_t({(uint32_t)t});
    //             std::memcpy(send_t.data, buf.data(), t);
    //             Tensor<char> recv_t({(uint32_t)t});
    //             std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //             party.recv_tensor_from<char>(party.pre_party_id, recv_t);
    //             st.join();
    //             rem -= t;
    //         }
    //         rem = total_key_bytes;
    //         while (rem) {
    //             size_t t = std::min(rem, buf.size());
    //             Tensor<char> send_t({(uint32_t)t});
    //             std::memcpy(send_t.data, buf.data(), t);
    //             Tensor<char> recv_t({(uint32_t)t});
    //             std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
    //             party.recv_tensor_from<char>(party.pre_party_id, recv_t);
    //             st.join();
    //             rem -= t;
    //         }
    //     }
        
    // }
    
    
}

// Batched wrapper to avoid sending too-large key material in a single call.
static void offline_drelu_batched(int bit_len, size_t total_count, bool Verification=false) {
    // Estimate bytes per DReLU using DCF key size + small reveal overhead
    size_t per_dcf = dcf_key_size_bytes(bit_len, bit_len, 1, Verification);
    size_t reveal_overhead = 2 * bytes_for_bits(bit_len); // delta reveal bytes per item (approx)
    size_t est_per = per_dcf + reveal_overhead;
    // Choose a conservative target max bytes per sub-call (4 MiB)
    const size_t MAX_BYTES = (size_t)1 << 22;
    size_t max_count = est_per == 0 ? 1 : std::max((size_t)1, MAX_BYTES / est_per);

    size_t rem = total_count;
    while (rem) {
        size_t this_count = std::min(rem, max_count);
        offline_drelu(bit_len, this_count, Verification);
        rem -= this_count;
    }
}

static void run_offline_alex(int party_id) {
    // compute DReLU counts for AlexNet based on cnn3pc input shape (batch=128, 3x33x33)
    // limit OpenMP threads to 1 to reduce memory pressure during offline key generation
    omp_set_num_threads(1);
    const int batch = 1; // match tests/cnn3pc.cpp
    int h = 33, w = 33;

    // conv1: (3->96) k=11 s=4 p=9
    const int c1_out = 96;
    int h_c1 = conv_out_dim(h, 11, 4, 9);
    int w_c1 = conv_out_dim(w, 11, 4, 9);
    // pool1: k=3 s=2 p=0
    int h_p1 = pool_out_dim(h_c1, 3, 2, 0);
    int w_p1 = pool_out_dim(w_c1, 3, 2, 0);
    size_t elems_pool1 = batch *c1_out * h_p1 * w_p1;
    // one DReLU for pool1 and one for relu1
        offline_drelu(31, elems_pool1 * (9 - 1));
        offline_drelu(31, elems_pool1);

    // conv2: (96->256) k=5 s=1 p=1
    const int c2_out = 256;
    int h_c2 = conv_out_dim(h_p1, 5, 1, 1);
    int w_c2 = conv_out_dim(w_p1, 5, 1, 1);
    // pool2: k=3 s=2 p=0
    int h_p2 = pool_out_dim(h_c2, 3, 2, 0);
    int w_p2 = pool_out_dim(w_c2, 3, 2, 0);
    size_t elems_pool2 = batch *c2_out * h_p2 * w_p2;
    offline_drelu(31, elems_pool2 * (9 - 1));
    offline_drelu(31, elems_pool2);

    // conv3: (256->384) k=3 s=1 p=1
    const int c3_out = 384;
    int h_c3 = conv_out_dim(h_p2, 3, 1, 1);
    int w_c3 = conv_out_dim(w_p2, 3, 1, 1);
    size_t elems_relu3 = batch *c3_out * h_c3 * w_c3;
    offline_drelu(31, elems_relu3);

    // conv4: (384->384) k=3 s=1 p=1
    const int c4_out = 384;
    int h_c4 = conv_out_dim(h_c3, 3, 1, 1);
    int w_c4 = conv_out_dim(w_c3, 3, 1, 1);
    size_t elems_relu4 = batch * c4_out * h_c4 * w_c4;
    offline_drelu(31, elems_relu4);

    // conv5: (384->256) k=3 s=1 p=1
    const int c5_out = 256;
    int h_c5 = conv_out_dim(h_c4, 3, 1, 1);
    int w_c5 = conv_out_dim(w_c4, 3, 1, 1);
    size_t elems_relu5 = batch *c5_out * h_c5 * w_c5;
    offline_drelu(31, elems_relu5);

    // keep small final LUT/DReLU simulation for FC-related ops
    const int fc3_out = 10;
    size_t elems_fc3 = batch * fc3_out * h_c5 * w_c5;
    offline_drelu(31, 2*elems_relu5+elems_fc3);
}


static void run_offline_resnet(int party_id) {
    // limit OpenMP threads to 1 to reduce memory pressure during offline key generation
    omp_set_num_threads(1);
    const int batch = 1;
    int h = 224, w = 224;

    // conv1: 7x7 s=2 p=3 -> out 64
    int h1 = conv_out_dim(h, 7, 2, 3);
    int w1 = conv_out_dim(w, 7, 2, 3);
    size_t elems_conv1 = (size_t)batch * 1 * h1 * w1;
    offline_drelu(31, elems_conv1); // relu after conv1

    // pool1: 3x3 s=2 p=1
    int hp = pool_out_dim(h1, 3, 2, 1);
    int wp = pool_out_dim(w1, 3, 2, 1);
    size_t elems_pool1 = (size_t)batch * 1 * hp * wp;
    offline_drelu(31, elems_pool1 * (9 - 1));

    // ResNet50 bottleneck groups: (out_channels, blocks, stride)
    struct Group { int out_ch; int blocks; int stride; };
    std::vector<Group> groups = { {1, 3, 1}, {2, 4, 2}, {4, 6, 2}, {8, 3, 2} };

    int hcur = hp, wcur = wp;
    int in_ch = 64;
    const int expansion = 4;

    for (const auto &g : groups) {
        for (int bi = 0; bi < g.blocks; ++bi) {
            int stride_used = (bi == 0) ? g.stride : 1;

            // conv1: 1x1
            int h_a = conv_out_dim(hcur, 1, 1, 0);
            int w_a = conv_out_dim(wcur, 1, 1, 0);
            size_t elems_relu1 = (size_t)batch * g.out_ch * h_a * w_a;
            offline_drelu_batched(31, elems_relu1);

            // conv2: 3x3 (may downsample)
            int h_b = conv_out_dim(h_a, 3, stride_used, 1);
            int w_b = conv_out_dim(w_a, 3, stride_used, 1);
            size_t elems_relu2 = (size_t)batch * g.out_ch * h_b * w_b;
            offline_drelu_batched(31, elems_relu2);

            // conv3: 1x1 -> expansion
            int h_c = conv_out_dim(h_b, 1, 1, 0);
            int w_c = conv_out_dim(w_b, 1, 1, 0);
            size_t elems_after_add = (size_t)batch * g.out_ch * expansion * h_c * w_c;
            offline_drelu_batched(31, elems_after_add);

            // update current spatial dims and channels
            hcur = h_c; wcur = w_c; in_ch = g.out_ch * expansion;
        }
    }
}


void OfflineCNN3pcTest::main_test(int party_id)
{   
    fss_init();
    test_func<Party3PC>(run_offline_alex, "Offline AlexNet", loop, warmup_rounds, party_id);
    Party3PC::getInstance().sync();
    test_func<Party3PC>(run_offline_resnet, "Offline ResNet", loop, warmup_rounds, party_id);
    // Party3PC::getInstance().sync();
    // run_func_wrap_test(party_id);
}