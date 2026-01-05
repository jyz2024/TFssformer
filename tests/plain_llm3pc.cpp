#include "plain_layers.h"
#include "llmConfig.h"
#include "test_base.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include <cmath>
#include "cnpy/cnpy.h"

using namespace plain;

// Helper to load weights from .npz
static std::map<std::string, Tensor<double>> load_weights(const std::string& path) {
    std::map<std::string, Tensor<double>> weights;
    try {
        cnpy::npz_t npz = cnpy::npz_load(path);
        for(auto& it : npz) {
            std::string name = it.first;
            cnpy::NpyArray& arr = it.second;
            std::vector<uint32_t> shape;
            for(auto s : arr.shape) shape.push_back((uint32_t)s);
            
            Tensor<double> t;
            t.allocate(shape);
            
            if (arr.word_size == 4) {
                std::vector<float> data_f = arr.as_vec<float>();
                for(size_t i=0; i<data_f.size(); ++i) t.data[i] = (double)data_f[i];
            } else if (arr.word_size == 8) {
                std::vector<double> data_d = arr.as_vec<double>();
                std::copy(data_d.begin(), data_d.end(), t.data);
            } else {
                std::cerr << "Unsupported word size: " << arr.word_size << " for " << name << std::endl;
            }
            weights[name] = t;
        }
        std::cout << "Loaded " << weights.size() << " tensors from " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights from " << path << ": " << e.what() << std::endl;
    }
    return weights;
}

static void bert_accuracy_test(const std::string& task_name) {
    std::cout << "\n=== BERT Accuracy Test: " << task_name << " ===" << std::endl;
    
    std::string data_path = "data/plain_data/Bert_" + task_name + "_data.npz";
    std::string weight_path = "data/plain_data/Bert_" + task_name + "_weights.npz";
    
    std::cout << "Loading data from " << data_path << std::endl;
    // Load Data
    Tensor<double> inputs;
    std::vector<double> labels;
    try {
        cnpy::npz_t npz = cnpy::npz_load(data_path);
        if (npz.count("inputs")) {
            cnpy::NpyArray& arr = npz["inputs"];
            std::vector<uint32_t> shape;
            for(auto s : arr.shape) shape.push_back((uint32_t)s);
            inputs.allocate(shape);
            if (arr.word_size == 4) {
                std::vector<float> d = arr.as_vec<float>();
                for(size_t i=0; i<d.size(); ++i) inputs.data[i] = (double)d[i];
            } else {
                std::vector<double> d = arr.as_vec<double>();
                std::copy(d.begin(), d.end(), inputs.data);
            }
        }
        if (npz.count("labels")) {
            cnpy::NpyArray& arr = npz["labels"];
            if (task_name == "STS-B") {
                if (arr.word_size == 4) {
                    std::vector<float> f = arr.as_vec<float>();
                    for(auto v : f) labels.push_back((double)v);
                } else {
                    std::vector<double> d = arr.as_vec<double>();
                    labels = d;
                }
            } else {
                if (arr.word_size == 4) {
                    std::vector<int32_t> i32 = arr.as_vec<int32_t>();
                    for(auto v : i32) labels.push_back((double)v);
                } else if (arr.word_size == 8) {
                    std::vector<int64_t> i64 = arr.as_vec<int64_t>();
                    for(auto v : i64) labels.push_back((double)v);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Skipping " << task_name << ": " << e.what() << std::endl;
        return;
    }
    
    if (inputs.shape.empty()) {
        std::cout << "No inputs found for " << task_name << std::endl;
        return;
    }
    
    std::cout << "Loaded inputs: " << inputs.shape[0] << "x" << inputs.shape[1] << "x" << inputs.shape[2] << std::endl;
    std::cout << "Loaded labels: " << labels.size() << std::endl;

    std::cout << "Loading model weights from " << weight_path << std::endl;
    // Load Model
    TransformerConfig config("bert-base", 128);
    
    using P32 = FixedPoint<32, 12>;
    using P16 = FixedPoint<16, 6>;
    BertEncoder<double, P32, P16> bert(config);
    
    auto weights = load_weights(weight_path);
    bert.load_param(weights, "");
    
    int num_labels = (task_name == "STS-B") ? 1 : 2;
    Linear<double, P32, P16> classifier(config.hidden_size, num_labels, "classifier");
    classifier.weight.zero();
    classifier.bias.zero();
    
    int correct = 0;
    std::vector<double> preds;
    
    int N = inputs.shape[0];
    if (N > 50) {
        std::cout << "Limiting test samples from " << N << " to 50 for speed." << std::endl;
        N = 50;
    }
    int S = inputs.shape[1];
    int H = inputs.shape[2];
    
    std::cout << "Starting inference for " << N << " samples..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int i=0; i<N; ++i) {
        Tensor<double> x({1, (uint32_t)S, (uint32_t)H});
        for(int s=0; s<S; ++s) {
            for(int h=0; h<H; ++h) {
                x.data[s*H + h] = inputs.data[i*S*H + s*H + h];
            }
        }
        
        Tensor<double> enc_out = bert.forward(x);
        
        Tensor<double> cls_out({1, (uint32_t)H});
        for(int h=0; h<H; ++h) {
            cls_out.data[h] = enc_out.data[h];
        }
        
        Tensor<double> logits = classifier.forward(cls_out);
        
        if (task_name == "STS-B") {
            double pred = logits.data[0];
            preds.push_back(pred);
        } else {
            int pred_label = 0;
            if (logits.data[1] > logits.data[0]) pred_label = 1;
            
            if (pred_label == (int)labels[i]) correct++;
        }
        
        if (i % 100 == 0) std::cout << "." << std::flush;
    }
    std::cout << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Inference time: " << duration << " ms (" << duration/N << " ms/sample)" << std::endl;
    
    if (task_name == "STS-B") {
        double sum_p = 0, sum_l = 0;
        for(auto p : preds) sum_p += p;
        for(auto l : labels) sum_l += l;
        double mean_pred = sum_p / N;
        double mean_label = sum_l / N;
        
        double num = 0, den1 = 0, den2 = 0;
        for(int i=0; i<N; ++i) {
            num += (preds[i] - mean_pred) * (labels[i] - mean_label);
            den1 += (preds[i] - mean_pred) * (preds[i] - mean_pred);
            den2 += (labels[i] - mean_label) * (labels[i] - mean_label);
        }
        double pearson = (den1 > 0 && den2 > 0) ? num / std::sqrt(den1 * den2) : 0;
        std::cout << "Pearson Correlation: " << pearson << std::endl;
    } else {
        double acc = (double)correct / N;
        std::cout << "Accuracy: " << acc * 100.0 << "%" << std::endl;
    }
}

static void gpt2_perplexity_test() {
    std::cout << "\n=== GPT2 Perplexity Test ===" << std::endl;
    
    std::string data_path = "data/plain_data/GPT2_Wiki_data.npz";
    std::string weight_path = "data/plain_data/GPT2_Wiki_weights.npz";
    
    std::cout << "Loading data from " << data_path << std::endl;
    Tensor<double> inputs;
    std::vector<int64_t> labels;
    
    try {
        cnpy::npz_t npz = cnpy::npz_load(data_path);
        if (npz.count("inputs")) {
            cnpy::NpyArray& arr = npz["inputs"];
            std::vector<uint32_t> shape;
            for(auto s : arr.shape) shape.push_back((uint32_t)s);
            inputs.allocate(shape);
            if (arr.word_size == 4) {
                std::vector<float> d = arr.as_vec<float>();
                for(size_t i=0; i<d.size(); ++i) inputs.data[i] = (double)d[i];
            } else {
                std::vector<double> d = arr.as_vec<double>();
                std::copy(d.begin(), d.end(), inputs.data);
            }
        }
        if (npz.count("labels")) {
            cnpy::NpyArray& arr = npz["labels"];
            if (arr.word_size == 8) {
                labels = arr.as_vec<int64_t>();
            } else if (arr.word_size == 4) {
                std::vector<int32_t> d = arr.as_vec<int32_t>();
                for(auto v : d) labels.push_back(v);
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Skipping GPT2: " << e.what() << std::endl;
        return;
    }
    
    if (inputs.shape.empty()) return;
    
    std::cout << "Loaded inputs: " << inputs.shape[0] << "x" << inputs.shape[1] << "x" << inputs.shape[2] << std::endl;
    std::cout << "Loaded labels: " << labels.size() << std::endl;

    std::cout << "Loading model weights from " << weight_path << std::endl;
    TransformerConfig config("gpt2", 64);
    using P32 = FixedPoint<32, 12>;
    using P16 = FixedPoint<16, 6>;
    GPT2<double, P32, P16> gpt2(config);
    
    auto weights = load_weights(weight_path);
    gpt2.load_param(weights, "");
    
    Tensor<double> wte;
    if (weights.count("wte.weight")) {
        wte = weights["wte.weight"];
    } else {
        std::cout << "Warning: wte.weight not found, using random." << std::endl;
        wte.allocate(std::vector<uint32_t>{(uint32_t)config.vocab_size, (uint32_t)config.hidden_size});
        wte.zero();
    }
    Tensor<double> wte_T = wte.transpose();
    
    int N = inputs.shape[0];
    if (N > 50) {
        std::cout << "Limiting GPT2 test samples from " << N << " to 50 for speed." << std::endl;
        N = 50;
    }
    int S = inputs.shape[1];
    int H = inputs.shape[2];
    
    std::cout << "Starting inference for " << N << " samples..." << std::endl;
    double total_nll = 0;
    int total_tokens = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int i=0; i<N; ++i) {
        Tensor<double> x({1, (uint32_t)S, (uint32_t)H});
        for(int s=0; s<S; ++s) {
            for(int h=0; h<H; ++h) {
                x.data[s*H + h] = inputs.data[i*S*H + s*H + h];
            }
        }
        
        Tensor<double> hidden = gpt2.forward(x);
        
        hidden.reshape({(uint32_t)S, (uint32_t)H});
        Tensor<double> logits;
        logits.allocate(std::vector<uint32_t>{(uint32_t)S, (uint32_t)config.vocab_size});
        Tensor<double>::matmul(logits, hidden, wte_T);
        
        for(int t=0; t<S-1; ++t) {
            int label = labels[i*S + t + 1];
            if (label == -100) continue;
            
            double max_val = -1e9;
            for(int v=0; v<config.vocab_size; ++v) {
                if (logits.data[t*config.vocab_size + v] > max_val) max_val = logits.data[t*config.vocab_size + v];
            }
            
            double sum_exp = 0;
            for(int v=0; v<config.vocab_size; ++v) {
                sum_exp += std::exp(logits.data[t*config.vocab_size + v] - max_val);
            }
            
            double log_sum_exp = max_val + std::log(sum_exp);
            double log_prob = logits.data[t*config.vocab_size + label] - log_sum_exp;
            
            total_nll -= log_prob;
            total_tokens++;
        }
        
        if (i % 10 == 0) std::cout << "." << std::flush;
    }
    std::cout << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Inference time: " << duration << " ms" << std::endl;
    
    if (total_tokens > 0) {
        double avg_nll = total_nll / total_tokens;
        double perplexity = std::exp(avg_nll);
        std::cout << "Average NLL: " << avg_nll << std::endl;
        std::cout << "Perplexity: " << perplexity << std::endl;
    }
}

void PlainLLM3pcTest::main_test(int party_id) {
    if (party_id != 0) {
        std::cout << "Plaintext test only runs on party 0 (or single process)." << std::endl;
        return;
    }
    
    bert_accuracy_test("QNLI");
    bert_accuracy_test("RTE");
    bert_accuracy_test("STS-B");
    gpt2_perplexity_test();
}
