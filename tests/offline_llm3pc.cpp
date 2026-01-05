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
#include "falcon_compat.h"

// Define globals for Falcon compat
smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
smallType subtractModPrime[PRIME_NUMBER][PRIME_NUMBER];
smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];
// Precompute PrecomputeObject;

static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = print_res ? 1 : 2;
static int warmup_rounds = print_res ? 0 : 1;
static constexpr size_t BLOCK_BYTES = 16;
static bool Verification = false;

static size_t bytes_for_bits(int bits){
    return (bits + 7) / 8;
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

size_t dpf_key_size_bytes(int Bin, int Bout, int groupSize, bool Verification=false){
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

size_t dcf_key_size_bytes(int Bin, int Bout, int groupSize,bool Verification=false){
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
    // Use OpenMP parallel for to accelerate key generation.
    #pragma omp parallel for
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
        GroupElement res(0, bit_len);
        GroupElement idx(0, bit_len);
        size_t beta = 0;
        size_t alaha_check = 0;
        // 1.Full-domain evaluation to simulate verification cost
        #pragma omp parallel for
        for (size_t i = 0; i < count; ++i) {
            GroupElement idx(0, bit_len);
            for(size_t j = 0; j < (1 << bit_len); ++j){
                idx = GroupElement(j, bit_len);
                evalDPF(party.party_id, &res, idx, kp0,Verification);
            }
            beta += res.value;
            alaha_check += res.value * i;
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
    // Return the first share as the representative key for eval calls.
    return kp0;
}

//bit_len = table_size
static void offline_lut(int bit_len, size_t count,bool Verification=false) {
    auto &party = Party3PC::getInstance();
    // 1. VDPF Keys 
    DPFKeyPack kp0 = offline_pointcmp(bit_len, count, Verification);
    
    if(not Verification){
        GroupElement res(0, bit_len);
        GroupElement idx(0, bit_len);
    	#pragma omp parallel for
        for (size_t i = 0; i < count; ++i) {
            for(size_t j = 0; j < (1 << bit_len); ++j){
                evalDPF(party.party_id, &res, idx, kp0,Verification);
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
    #pragma omp parallel for
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
        GroupElement res(0, bit_len);
        GroupElement idx(0, bit_len);
        size_t beta = 0;
        size_t alaha_check = 0;
        // 1.Full-domain evaluation to simulate verification cost
        #pragma omp parallel for
        for (size_t i = 0; i < count; ++i) {
            GroupElement idx(0, bit_len);
            for(size_t j = 0; j < (1 << bit_len); ++j){
                idx = GroupElement(j, bit_len);
                evalDCF(party.party_id, &res, idx, kp0,Verification);
            }
            beta += res.value;
            alaha_check += res.value * i;
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
    if (Verification){
        size_t a_val=0, b_val=0;
        initFalconCompat();
        RSSVectorMyType a(count);
        RSSVectorSmallType theta(count);
        // Initialize a with some shares
        #pragma omp parallel for
        for(long long ii=0; ii<(long long)count; ++ii) {
            size_t i = (size_t)ii;
            a[i].first = a_val;
            a[i].second = b_val;
        }
        funcWrap(a, theta, count);
        // if (party.party_id == 0) std::cout << "funcWrap executed successfully." << std::endl;

        // verify delta_c, delta_y according to protocol in the image:
        RSSTensor<size_t> delta_c_shared({(uint32_t)count});
        RSSTensor<size_t> delta_y_shared({(uint32_t)count});
        #pragma omp parallel for
        for (long long ii = 0; ii < (long long)count; ++ii) {
            size_t i = (size_t)ii;
            delta_c_shared.first.data[i] = 1 - theta[i].first;
            delta_c_shared.second.data[i] = 1 - theta[i].second;
            delta_y_shared.first.data[i] = 1 - theta[i].first;
            delta_y_shared.second.data[i] = 1 - theta[i].second;
        }
        __int128 total_key_bytes_128 = ( __int128)2 * ( __int128)count * ( __int128)bytes_for_bits(bit_len);
        if (total_key_bytes_128 > ( __int128)std::numeric_limits<size_t>::max()){
        } else {
            size_t total_key_bytes = (size_t)total_key_bytes_128;
            const size_t CHUNK = (size_t)1 << 20; // 1MB
            std::vector<char> buf(std::min(CHUNK, total_key_bytes), 0);
            size_t rem = total_key_bytes;
            while (rem) {
                size_t t = std::min(rem, buf.size());
                Tensor<char> send_t({(uint32_t)t});
                std::memcpy(send_t.data, buf.data(), t);
                Tensor<char> recv_t({(uint32_t)t});
                std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
                party.recv_tensor_from<char>(party.pre_party_id, recv_t);
                st.join();
                rem -= t;
            }
            rem = total_key_bytes;
            while (rem) {
                size_t t = std::min(rem, buf.size());
                Tensor<char> send_t({(uint32_t)t});
                std::memcpy(send_t.data, buf.data(), t);
                Tensor<char> recv_t({(uint32_t)t});
                std::thread st([&](){ party.send_tensor_to<char>(party.next_party_id, send_t); });
                party.recv_tensor_from<char>(party.pre_party_id, recv_t);
                st.join();
                rem -= t;
            }
        }
        
    }
    
    
}

static void run_offline_MHA(int party_id,size_t exp_num, size_t inv_num, size_t max_num){
   // Softmax： uses UCMP for Max
    offline_drelu(31, max_num);
    // Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * inv_num);
    offline_lut(2, inv_num);
    offline_lut(12, inv_num);
    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, exp_num);
    offline_lut(4, exp_num); // compute >>z
}

static void run_offline_ln(int party_id,size_t ln_num){
   // Rsqt:One for base LUT; One for rsqt LUT; One for exsqrt
    offline_drelu(31, 2 * ln_num);
    offline_lut(2, ln_num);
    offline_lut(2, ln_num);
    offline_lut(12, ln_num);
}

static void run_offline_ffn(int party_id,size_t gelu_num){
   //Gelu
    offline_drelu(15, gelu_num * 3);
    offline_lut(8, gelu_num);
}

static void offline_tf_test(int party_id){
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    size_t seq_len = tf_config.seq_len;
    size_t hidden = tf_config.hidden_size;
    size_t intermediate = tf_config.intermediate_size;
    size_t heads = tf_config.num_attention_heads;
    // GELU (16-bit LUT)
    // FFN structure: hidden -> intermediate -> (GELU) -> hidden
    size_t gelu_count =  seq_len * intermediate;

    // Softmax (32-bit LUT)
    // Attention scores matrix: batch * heads * seq_len * seq_len
    size_t softmax_exp_count =  heads * seq_len * seq_len;
    size_t softmax_inv_count =  heads * seq_len;

    // LayerNorm (32-bit LUT)
    // Encoder layers have 2 LNs each.
    // bert_test in llm3pc.cpp adds 1 extra SecMixedLayerNorm.
    size_t ln_count =  seq_len;

    // Softmax Max (UCMP)
    // Finding max of seq_len elements requires seq_len - 1 comparisons.
    size_t softmax_max_count =  heads * seq_len * (seq_len - 1);
    omp_set_num_threads(1);
    // test_func<Party3PC>(run_offline_MHA, "Offline MHA", loop, warmup_rounds, party_id, softmax_exp_count, softmax_inv_count, softmax_max_count);
    test_func<Party3PC>(run_offline_MHA, "Offline MMHA", loop, warmup_rounds, party_id, softmax_exp_count, softmax_inv_count, softmax_max_count/2);
    test_func<Party3PC>(run_offline_ffn, "Offline FFN", loop, warmup_rounds, party_id, gelu_count);
    test_func<Party3PC>(run_offline_ln, "Offline LayerNorm", loop, warmup_rounds, party_id, ln_count);
}

static void run_offline_bert(int party_id) {
    // set OpenMP threads for this workload
    omp_set_num_threads(16);
    TransformerConfig config("bert-base", 128);
    size_t layers = config.num_layers;
    size_t seq_len = config.seq_len;
    size_t hidden = config.hidden_size;
    size_t intermediate = config.intermediate_size;
    size_t heads = config.num_attention_heads;
    size_t batch = batch_size;

    // GELU (16-bit LUT)
    // FFN structure: hidden -> intermediate -> (GELU) -> hidden
    size_t gelu_count = layers * batch * seq_len * intermediate;
    
    // Softmax (32-bit LUT)
    // Attention scores matrix: batch * heads * seq_len * seq_len
    size_t softmax_exp_count = layers * batch * heads * seq_len * seq_len;
    size_t softmax_inv_count = layers * batch * heads * seq_len;
    
    // LayerNorm (32-bit LUT)
    // Encoder layers have 2 LNs each.
    // bert_test in llm3pc.cpp adds 1 extra SecMixedLayerNorm.
    size_t ln_count = (layers * 2 + 1) * batch * seq_len;

    // Softmax Max (UCMP)
    // Finding max of seq_len elements requires seq_len - 1 comparisons.
    size_t softmax_max_count = layers * batch * heads * seq_len * (seq_len - 1);

    // cout << "Gelun number" << gelu_count << endl;   //4718592  3072*128*12
    // cout << "Softmax exp number" << softmax_exp_count << endl; //2359296    1536*128*12
    // cout << "Softmax inv number" << softmax_inv_count << endl; //18432  128*12*12
    // cout << "Ln number" << ln_count << endl;    //3200  128*25
    // cout << "Softmax max number" << softmax_max_count << endl;  //2340864   1524*128*12
    
    //Gelu
    offline_drelu(15, 36864 * 3);
    offline_lut(8, 36864);

    // Rsqt:One for base LUT; One for rsqt LUT; One for exsqrt
    offline_drelu(31, 2 * 25);
    offline_lut(2, 25); //ex_base
    offline_lut(2, 25); //exsqrt_base
    offline_lut(12, 25);

    //Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * 12*12);
    offline_lut(2, 12*12); //ex_base
    offline_lut(12, 12*12); 

    // Softmax: uses UCMP for Max
    offline_drelu(31, 1524*12);
    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, 1536*12);
    offline_lut(4, 1536*12); // compute >>z
}

static void run_offline_gpt2(int party_id) {
    // set OpenMP threads for this workload
    omp_set_num_threads(16);
    TransformerConfig config("gpt2", 64);
    size_t layers = config.num_layers;
    size_t seq_len = config.seq_len;
    size_t hidden = config.hidden_size;
    size_t intermediate = config.intermediate_size;
    size_t heads = config.num_attention_heads;
    size_t batch = batch_size;
	

    // GELU (16-bit LUT)
    size_t gelu_count = layers * batch * seq_len * intermediate;
    
    // Softmax (32-bit LUT)
    // Attention scores matrix: batch * heads * seq_len * seq_len
    size_t softmax_exp_count = layers * batch * heads * seq_len * seq_len;
    size_t softmax_inv_count = layers * batch * heads * seq_len;
    
    // LayerNorm (32-bit LUT)
    // GPT2 has 2 LNs per layer + 1 final LN (ln_f).
    size_t ln_count = (layers * 2 + 1) * batch * seq_len;

    // Softmax Max (UCMP)
    size_t softmax_max_count = layers * batch * heads * seq_len * (seq_len - 1);
    
    // cout << "Gelun number" << gelu_count << endl;   //2359296  3072*64*12
    // cout << "Softmax exp number" << softmax_exp_count << endl; //589824    768*64*12
    // cout << "Softmax inv number" << softmax_inv_count << endl; //9216  64*12*12
    // cout << "Ln number" << ln_count << endl;    //1600  64*25
    // cout << "Softmax max number" << softmax_max_count << endl;  //580608   756*64*12

    
    //Gelu
    offline_drelu(15, 3072 * 12 * 3);
    offline_lut(8, 3072 * 12);

    // Rsqt:One for base LUT; One for rsqt LUT; One for exsqrt
    offline_drelu(31, 2 * 25);
    offline_lut(2, 25);
    offline_lut(2, 25);
    offline_lut(12, 25);

    // Assuming Softmax uses UCMP for Max
    offline_drelu(31, 756 * 12);
    // Inv: One for base LUT; One for Inv LUT
    offline_drelu(31, 2 * 12 * 12);
    offline_lut(2, 12 * 12);
    offline_lut(12, 12 * 12);

    // Assuming Softmax uses UCMP for e^x
    offline_drelu(11, 768 * 12);
    offline_lut(4, 768 * 12);
}

static void run_func_wrap_test(int party_id) {
    // set OpenMP threads for funcWrap testing
    omp_set_num_threads(8);
    initFalconCompat();
    size_t size = 10;
    RSSVectorMyType a(size);
    RSSVectorSmallType theta(size);
    
    // Initialize a with some shares
    for(size_t i=0; i<size; ++i) {
        a[i].first = rand();
        a[i].second = rand();
    }
    
    funcWrap(a, theta, size);
    if (party_id == 0) std::cout << "funcWrap executed successfully." << std::endl;
}

void OfflineLLM3pcTest::main_test(int party_id)
{   
    fss_init();
    // offline_tf_test(party_id);
    // Party3PC::getInstance().sync();
    test_func<Party3PC>(run_offline_bert, "Offline BERT", loop, warmup_rounds, party_id);
    Party3PC::getInstance().sync();
    test_func<Party3PC>(run_offline_gpt2, "Offline GPT-2", loop, warmup_rounds, party_id);
    // Party3PC::getInstance().sync();
    // test_func<Party3PC>(run_func_wrap_test, "funcWrap", loop, warmup_rounds, party_id);
}

