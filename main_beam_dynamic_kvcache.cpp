
// #include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
// #include <gperftools/profiler.h>
#include <vector>
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
// #include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <map>
#include <numeric>
// #include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
// #include "det_util.h"
#include "../riscv64/sentencepiece/include/sentencepiece_processor.h"
#include<limits.h>
#include<algorithm>
#include<cstring>
#include <float.h>
#include <set>
#include <fstream> // c++文件操作
#include <iomanip> // 设置输出格式
#include<time.h> 
#include <sys/time.h>
#include <queue>
#include <chrono>



using namespace std;
// using namespace cv;
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

/// 统计时间
vector<double> vevtor_time_0;
vector<double> vevtor_time_1;
vector<double> vevtor_time_2;
vector<double> vevtor_time_3;
vector<double> vevtor_time_4;
vector<double> vevtor_time_5;


// / Define a struct to hold both the value and its index
struct ValueIndexPair {
    float value;
    int index;
};


// struct Finalized {
//     std::vector<std::vector<int>> tokens;
//     std::vector<std::vector<float>> scores;
// };


struct Finalized_Sent{ 
    std::vector<int> token;
    float scores_tmp;

};

// Function to perform the equivalent of torch.topk() in C++
void topk(const std::vector<float>& input, const int k, std::vector<float>& values, std::vector<int>& indices) {
    // Initialize a vector of ValueIndexPair structs to hold the input values and their indices
    std::vector<ValueIndexPair> pairs(input.size());
    for (int i = 0; i < input.size(); ++i) {
        pairs[i] = {input[i], i};
    }
    
    // Use the nth_element algorithm to find the k-th largest element in the vector
    auto comp = [](const ValueIndexPair& a, const ValueIndexPair& b) {
        return a.value > b.value;
    };
    std::nth_element(pairs.begin(), pairs.begin() + k, pairs.end(), comp);
    
    // Extract the top k values and their indices from the sorted pairs
    values.resize(k);
    indices.resize(k);
    for (int i = 0; i < k; ++i) {
        values[i] = pairs[i].value;
        indices[i] = pairs[i].index;
    }
}



// 排序函数，返回排序后元素的索引数组
std::vector<int> sort_index_k_v2(std::vector<float>& lprobs, int k) {
    std::priority_queue<std::pair<float,int>, std::vector<std::pair<float,int>>,  less<> > topK_index;
    for(int i = 0; i < lprobs.size(); i++){
        if(topK_index.size()>=k && lprobs[i]<=topK_index.top().first){
            topK_index.pop();
            topK_index.emplace(lprobs[i], i);
        }
        else if(topK_index.size()<k ){
            topK_index.emplace(lprobs[i], i);
        }
    }
    std::vector<int> index(k);
    for (int i = 0; i < k; ++i) {
        // cout<< vec_pair_new[i].second<< endl;
        index[k-1-i] = topK_index.top().second;
        topK_index.pop();
    }
    return index;
}


vector<int> finalize_hypos(int step,vector<int> bbsz_idx,std::vector<float> eos_scores,std::vector<std::vector<int>> tokens,
                            std::vector<std::vector<std::vector<float>>> scores,std::vector<std::vector<Finalized_Sent>>& finalized, vector<int>& finished,int beam_size,int max_len){

    std::vector<std::vector<int>> token_clone;
    for (int i = 0; i < bbsz_idx.size(); i++) {
        int row = bbsz_idx[i];
        std::vector<int> tmp;
        for (int j = 1; j < step+2; j++) {
            if (j ==step+1){
                tmp.push_back(2);//eos =2
            }
            else {
                tmp.push_back(tokens[row][j]);
            }
        
        }
        // eos_bbsz_idx.insert(eos_bbsz_idx.end(), tmp);
        token_clone.push_back(tmp);
    }                                   
    
    std::vector<std::vector<float>> scores_tmp;
    for (int i = 0; i < scores.size(); i++) {
    
        for (int j = 0; j < scores[0].size(); j++) {
            std::vector<float> tmp;
            for (int k = 0; k < scores[0][0].size(); k++) {
                tmp.push_back(scores[i][j][k]);
            }
            scores_tmp.push_back(tmp);
        }
    }


    std::vector<std::vector<float>> pos_scores;
    for (int i = 0; i < bbsz_idx.size(); i++) {
        int row = bbsz_idx[i];
        std::vector<float> tmp;
        float prior_value=0;
        for (int j = 0; j < step+1; j++) {
            if (j ==step){
                tmp.push_back(eos_scores[row] - prior_value);//##TODO:eos =2
                // prior_value = scores_tmp[row][j];
            }
            else {
                tmp.push_back(scores_tmp[row][j]-prior_value);
                prior_value = scores_tmp[row][j];
            }
        
        }
        // eos_bbsz_idx.insert(eos_bbsz_idx.end(), tmp);
        pos_scores.push_back(tmp);
    }    
    //长度惩罚
    for (int j=0; j<eos_scores.size(); j++){
        eos_scores[j] = eos_scores[j]/(step+1); //##TODO:self.len_penalty =1
    }

    std::vector<int> cum_unfin ;
    int prev =0;
    for (int i = 0; i < finished.size(); i++) {
        if (finished[i]){
            prev++;
        }
        else{
            cum_unfin.push_back(prev);
        }
    }
    std::vector<int> unfin_idx ;
    for (int i = 0; i < bbsz_idx.size(); i++) {
        unfin_idx.push_back(round(bbsz_idx[i] / beam_size));
    }

    std::vector<int> sent ;
    for (int i = 0; i <  unfin_idx.size(); i++) {
        int index = unfin_idx[i];
        sent.push_back( unfin_idx[i]+ cum_unfin[unfin_idx[i]]);
    }

    std::vector<int> seen ;
    std::vector<int> unique_seen ;
    // for (int i = 0; i <  unfin_idx.size(); i++) {
    //     int index = unfin_idx[i];
    //     sent.push_back( unfin_idx[i]+ cum_unfin[unfin_idx[i]]);
    // }

    // std::vector<unsigned long long> unique_seen;
    std::set<float> seen_set;

    // Loop through the sent and unfin_idx pairs and add unique values to unique_seen
    for (int i; i<sent.size(); i++) {
        float seen = (sent[i] << 32) + unfin_idx[i];
        if (seen_set.find(seen) == seen_set.end()) {
            unique_seen.push_back(seen);
            seen_set.insert(seen);
        }
    }
    std::vector<int> newly_finished ;
    for (int i = 0; i < bbsz_idx.size(); ++i) {
            // An input sentence (among those in a batch) is finished when
            // beam_size hypotheses have been collected for it
            if (finalized[sent[i]].size() < beam_size) {
                // std::vector<Finalized_Sent> finalized_tmp;
                Finalized_Sent one_sent;
                one_sent.token=token_clone[i] ;
                one_sent.scores_tmp=eos_scores[i] ;
                finalized[sent[i]].push_back(one_sent);
                // finalized_tmp.push_back(one_sent);
                // finalized[sent[i]].push_back(eos_scores[i]);
            }
            
        }

    for (auto unique_s : unique_seen) {
        // check termination conditions for this sentence
        int unique_sent = unique_s >> 32;
        int unique_unfin_idx = unique_s - (unique_sent << 32);
        int flag;
        if ((finalized[unique_sent].size() == beam_size) || (step == max_len)) {
            flag = 1;
        } else {
            flag = 0;
        }
        if (!finished[unique_sent] && flag) {
            finished[unique_sent] = 1;
            newly_finished.push_back(unique_unfin_idx);
        }
    }

    return newly_finished;

}


void log_softmax(std::vector<std::vector<std::vector<float>>> input, vector<vector<vector<float>>> &output){

    // vector<vector<vector<float>>> output;
	for (int i = 0; i < input.size(); ++i) {
        vector<vector<float>> tmp1;
        for (int j = 0; j < input[0].size(); ++j) {
            
            float sum =0;
            for (int k = 0; k < input[0][0].size(); ++k) {
                sum = sum+ std::exp(input[i][j][k] ) ;
            } 
            vector<float> tmp;
            for (int k = 0; k < input[0][0].size(); ++k) {
                tmp.push_back(std::log(std::exp(input[i][j][k] )/sum) );
            }
            tmp1.push_back(tmp);
        }
        output.push_back(tmp1);
	}
    return;
}

// template <class T>
// std::vector<T>read_binary_file(const char *file_name)
// {
//   cout<< 0<< endl;
//   std::ifstream ifs(file_name, std::ios::binary);
//   cout<< 1<< endl;
//   ifs.seekg(0, ifs.end);
//   cout<< 2<< endl;
//   size_t len = ifs.tellg();
//   cout<< 3<< endl;
//   std::vector<T> vec(len / sizeof(T), 0);
//   cout<< 1<< endl;
//   ifs.seekg(0, ifs.beg);
//   cout<< 1<< endl;
//   ifs.read(reinterpret_cast<char*>(vec.data()), len);
//   cout<< 5<< endl;
//   ifs.close();
//   return vec;
// }

// template <class T>
// std::vector<T>read_binary_file(const char *file_name)
// {
//   std::ifstream ifs(file_name, std::ios::binary);
//   ifs.seekg(0, ifs.end);
//   size_t len = ifs.tellg();
//   std::vector<T> vec(len / sizeof(T), 0);
//   ifs.seekg(0, ifs.beg);
//   ifs.read(reinterpret_cast<char*>(vec.data()), len);
//   ifs.close();
//   return vec;
// }

template <class T>
std::vector<T>read_binary_file_2(const char *file_name)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end); //将读取位置设置为文件末尾
    size_t len = ifs.tellg();
    std::vector<T> vec(len / sizeof(T), 0);
    ifs.seekg(0, ifs.beg);

    int base = 5*1024*1024;
    int round = (len-1) / base +1;
    //   std::cout<< round<<" "<<std::endl;;
    for (int count=0; count<round-1; count++){
        std::cout<< count<<" "<<std::endl;
        ifs.read(reinterpret_cast<char*>(vec.data()+ count*base), base);
    }
    ifs.read(reinterpret_cast<char*>(vec.data()+ (round-1)*base), len % base);
    
//   ifs.read(reinterpret_cast<char*>(vec.data()), 5);
//   ifs.read(reinterpret_cast<char*>(vec.data()+5), len-5);
//   std::cout<< unsigned(vec[3*10*1024*1024]) <<" "<< unsigned(vec[4*3*1024*1024]) << unsigned(vec[5*10*1024*1024]) << unsigned(vec[8])  <<std::endl;
  ifs.close();
  return vec;
}


template <class T>
std::vector<T>read_binary_file_1(const char *file_name)
{
  std::ifstream ifs(file_name, std::ios::binary);
  ifs.seekg(0, ifs.end);
  size_t len = ifs.tellg();
  std::vector<T> vec(len / sizeof(T), 0);
  ifs.seekg(0, ifs.beg);
  ifs.read(reinterpret_cast<char*>(vec.data()), len);
  ifs.close();
  return vec;
}

std::vector<unsigned char> read_binary_file(const char *file_name)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    std::vector<unsigned char> vec(len / sizeof(unsigned char), 0);
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char *>(vec.data()), len);
    ifs.close();
    return vec;
}

result<std::vector<value_t>> to_values(value_t v)
{
    if (v.is_a<tensor>())
    {
        return ok(std::vector { v });
    }
    else if (v.is_a<nncase::tuple>())
    {
        auto out_fields = v.as<nncase::tuple>().unwrap()->fields();
        return ok(std::vector(out_fields.begin(), out_fields.end()));
    }
    else
    {
        return err(std::errc::invalid_argument);
    }
}

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}


void encoder_preprocess(sentencepiece::SentencePieceProcessor& sp_en, string input_text, vector<int64_t>& source_seq, vector<uint8_t>& source_mask){
    
    /// encoder preprocess
    int end_symbol = 2;
    // std::vector<int> ids = {5485, 44, 13, 1377, 90, 554, 1358};
    std::vector<int> ids;
    sp_en.Encode(input_text, &ids);
    ids.push_back(end_symbol);
    for(int i=0; i<ids.size(); i++){
        // source_seq[i] = ids[i];
        source_seq.push_back(ids[i]);
        source_mask.push_back(0);
        cout<< ids[i]<< " ";
        // source_mask[i] = 1;
    }
    cout<< endl;
    cout<< "src lenght:"<< ids.size()<<endl;
    // for(int i=ids.size(); i<source_mask.size(); i++){
    //     source_mask[i] = 1;
    //     source_seq[i] = 1;
    // } 
    // for(int i=0; i<source_seq.size()-ids.size(); i++){
    //     source_seq[i] = 1;
        
    // }
    // for(int i=0; i<ids.size(); i++){
    //     source_seq[i + source_seq.size()-ids.size()] = ids[i];
    // }

    return;
}


void encoder_inference(interpreter& interp, vector<int64_t>& source_seq, vector<float>& vector_encoder_output_0, vector<float>& vector_encoder_output_1,vector<float>& vector_encoder_output_2, vector<float>& vector_encoder_output_4, vector<float>& vector_encoder_output_5 ){//float** encoder_output_data){
    
    // clock_t start2,end2;
    // start2=clock();

    // struct timeval t1,t2;
    // double timeuse;
    // gettimeofday(&t1,NULL);    
    /// load kmodel
    // cout << "encoder in " << endl ;
    // interpreter interp;
    // std::ifstream ifs(kmodel, std::ios::binary);
    // interp.load_model(ifs).expect("Invalid kmodel");  

    auto entry = interp.entry_function().expect("entry function is nullptr");

    std::vector<value_t> inputs;
    for (size_t i = 0; i < interp.inputs_size(); i++)
    {
        auto type = entry->parameter_type(i).expect("parameter type out of index");
        auto ts_type = type.as<tensor_type>().expect("input is not a tensor type");
        // dims_t shape = ts_type->shape().as_fixed().unwrap();
        dims_t shape {1,source_seq.size()} ;

        auto in_data = source_seq; //read_binary_file(argv[i + 2]);
        auto data_type = ts_type->dtype()->typecode();
        // cout << "host_runtime_tensor::create" <<source_seq.size()<< endl ;
        // for (int j = 0; j < source_seq.size(); j++)
        // {  
        //     cout << "host_runtime " <<source_seq[j]<< endl ;
        // }
        auto tensor = host_runtime_tensor::create(data_type, shape, { (gsl::byte *)in_data.data(), (size_t)in_data.size()*sizeof(in_data[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
        // cout << "host_runtime_tensor::create" << endl ;
        hrt::sync(tensor, sync_op_t::sync_write_back, true).unwrap();
        inputs.push_back(tensor.impl());
    }
    // kmodel run
    auto start = std::chrono::steady_clock::now();
    auto return_value = interp.entry_function().expect("no entry_function")->invoke(inputs).expect("run entry_function failed");
    auto stop = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "encoder interp run: " << duration << " ms, fps = " << 1000 / duration << std::endl;

    // 输出结果整理
    auto values = to_values(return_value).expect("unsupported value type");
    auto t = values[0].as<tensor>().expect("value is not a tensor");
    auto unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    auto mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_0 = (float *)mapped_buf.buffer().data();

    int data_shape = source_seq.size()*192;
    // auto out_shape_0 = interp.output_shape(0);
    cout << "encoder out_shape: " << data_shape << ", " << endl;
    for(int i=0; i<data_shape; i++){
        vector_encoder_output_0.push_back(output_data_0[i]);
    }
    cout << "encoder out_shape: " <<  values.size() << ", " << endl;

    t = values[1].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_1 = (float *)mapped_buf.buffer().data();

    data_shape = source_seq.size()*192;
    // auto out_shape_0 = interp.output_shape(0);
    cout << "encoder out_shape: " << data_shape << ", " << endl;
    for(int i=0; i<data_shape; i++){
        vector_encoder_output_1.push_back(output_data_1[i]);
    }

    t = values[2].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_2 = (float *)mapped_buf.buffer().data();
    // auto out_shape_0 = interp.output_shape(0);
    cout << "encoder out_shape: " << data_shape << ", " << endl;
    for(int i=0; i<data_shape; i++){
        vector_encoder_output_2.push_back(output_data_2[i]);
    }
    t = values[4].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_4 = (float *)mapped_buf.buffer().data();
    // auto out_shape_0 = interp.output_shape(0);
    cout << "encoder out_shape: " << data_shape << ", " << endl;
    for(int i=0; i<data_shape; i++){
        vector_encoder_output_4.push_back(output_data_4[i]);
    }

    t = values[5].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_5 = (float *)mapped_buf.buffer().data();
    // auto out_shape_0 = interp.output_shape(0);
    cout << "encoder out_shape: " << data_shape << ", " << endl;
    for(int i=0; i<data_shape; i++){
        vector_encoder_output_5.push_back(output_data_5[i]);
    }

    // float *output_data_2 = (float *)(mapped_buf.buffer().data() + source_seq.size()*192 *2*4 );
    // for(int i=0; i<source_seq.size()*192; i++){
    //     vector_encoder_output_2.push_back(output_data_2[i]);
    // }
    // float *output_data_4 = (float *)(mapped_buf.buffer().data() + source_seq.size()*192*3 *4+ source_seq.size()*1 );
    // for(int i=0; i<source_seq.size()*192; i++){
    //     vector_encoder_output_4.push_back(output_data_4[i]);
    // }    
    // float *output_data_5 = (float *)(mapped_buf.buffer().data() + source_seq.size()*192*4 *4+ source_seq.size()*1 );
    // for(int i=0; i<source_seq.size()*192; i++){
    //     vector_encoder_output_5.push_back(output_data_5[i]);
    // }  

    // string save_path="dynamic_shape_kvcache_230/encoder_kcache1.txt";
    // ofstream outfile;
    // outfile.open(save_path, ios::trunc);//打开文件
    // //ios::in可替换
    // // unsigned(a)
    // //ios：：app，表示打开文件后，在写入的文件不会覆盖原文件中的内容，也就是原来文件中的数据会得到保存。
    // //ios::trunc,文件里面的内容会清零
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < vector_encoder_output_1.size(); i++){
    //     outfile << setprecision(8) << vector_encoder_output_1[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/encoder_vcache1.txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < vector_encoder_output_2.size(); i++){
    //     outfile << setprecision(8) << vector_encoder_output_2[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。
    // save_path="dynamic_shape_kvcache_230/encoder_kcache2.txt";

    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < vector_encoder_output_4.size(); i++){
    //     outfile << setprecision(8) << vector_encoder_output_4[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。
    // save_path="dynamic_shape_kvcache_230/encoder_vcache2.txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < vector_encoder_output_5.size(); i++){
    //     outfile << setprecision(8) << vector_encoder_output_5[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    return;

}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

static dims_t parse_digits(const std::string& s) {
    dims_t digits;
    auto strs = split(s, ' ');
    for (size_t i = 0; i < strs.size(); i++)
    {
        digits.push_back(std::stoi(strs[i]));
    }
    return digits;
}
static std::vector<dims_t> parse_multilines(const std::vector<std::string>& strs, size_t begin, size_t size) {
    std::vector<dims_t> result;
    for(auto i = strs.begin() + begin; i != strs.begin() + begin + size; ++i)
    {
        auto shape = parse_digits(*i);
        if(shape[0] == 0)
        {
            shape = dims_t{};
        }
        result.push_back(shape);
    }
    return result;
}
struct data_desc {
    std::vector<dims_t> input_shape;
    std::vector<dims_t> output_shape;
    bool is_empty() { return input_shape.empty() && output_shape.empty(); }
};

data_desc parse_desc(const unsigned char *kmodel_desc_raw) {
    auto kmode_desc = std::string(reinterpret_cast<const char*>(kmodel_desc_raw));
    auto descs = split(kmode_desc, '\n');
    auto nums = parse_digits(descs[0]);
    auto input_num = nums[0];
    auto output_num = nums[1];
    auto in_shapes = parse_multilines(descs, 1, input_num);
    auto out_shapes = parse_multilines(descs, 1 + input_num, output_num);
    return data_desc{in_shapes, out_shapes};
}

std::vector<float> decoder_single_inference(interpreter& interp,vector<float>& self_k_cache1 ,vector<float>& self_v_cache1 , vector<float>& self_k_cache2 ,vector<float>& self_v_cache2 , 
                                             vector<float>& vector_encoder_output1,vector<float>& vector_encoder_output2,vector<float>& vector_encoder_output4,vector<float>& vector_encoder_output5, 
                                             vector<float>& src_mask, vector<int64_t>& tgt_seq, vector<int> decode_round, std::vector<std::vector<std::vector<float>>>& lprobs, 
                                             vector<vector<vector<vector<float>>>>& self_k_cache1_ ,vector<vector<vector<vector<float>>>>& self_v_cache1_ , vector<vector<vector<vector<float>>>>& self_k_cache2_ ,vector<vector<vector<vector<float>>>>& self_v_cache2_  
                                             ){   

    struct timeval t0,t4,t1,t2,t3;
    if (decode_round[0]==3){
        // clock_t start2,end2;
        // start2=clock();
        gettimeofday(&t0,NULL);  
    } 
    // cout << "vector_encoder_output:"<<vector_encoder_output.size()<<" " << vector_encoder_output[10]<<" "<< vector_encoder_output[11]<<" "<< vector_encoder_output[12]<<endl;  

    auto entry = interp.entry_function().expect("entry function is nullptr");

 
    // cout << "decoder create1  "<<decode_round[0]+1<<endl; 
    std::vector<value_t> inputs;
    size_t index =0;
    auto type0 = entry->parameter_type(index).expect("parameter type out of index");
    auto ts_type0 = type0.as<tensor_type>().expect("input is not a tensor type");
    // dims_t shape = ts_type->shape().as_fixed().unwrap();
    dims_t shape_0 {3,decode_round[0]+1} ;
    auto in_data_0 = tgt_seq; //read_binary_file(argv[i + 2]);
    auto data_type0 = ts_type0->dtype()->typecode();
    auto in_tensor0 = host_runtime_tensor::create(data_type0, shape_0, { (gsl::byte *)in_data_0.data(), (size_t)in_data_0.size()*sizeof(in_data_0[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor0, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor0.impl());
    // cout << "decoder create2 "<<endl; 
    
    auto type1 = entry->parameter_type(index+1).expect("parameter type out of index");
    auto ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    dims_t shape_1 {3,8,decode_round[0] ,24} ;
    auto in_data_1 = self_k_cache1; //read_binary_file(argv[i + 2]);
    auto data_type1 = ts_type1->dtype()->typecode();    
    auto in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_1.data(), (size_t)in_data_1.size()*sizeof(in_data_1[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "self_k_cache1.size()/3 "<< self_k_cache1.size()/3 <<endl; 

    auto type2 = entry->parameter_type(index+2).expect("parameter type out of index");
    auto ts_type2 = type2.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();     dims_t shape_2 {3,8,src_mask.size()/3 ,24} ;
    dims_t shape_2 {3,8,decode_round[0],24} ;
    auto in_data_2 = self_v_cache1; //read_binary_file(argv[i + 2]);
    auto data_type2 = ts_type2->dtype()->typecode();   
    auto in_tensor2 = host_runtime_tensor::create(data_type2, shape_2, { (gsl::byte *)in_data_2.data(), (size_t)in_data_2.size()*sizeof(in_data_2[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor2, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor2.impl());
    // cout << "self_v_cache1.size()/3 "<< self_v_cache1.size()/3 <<endl; 

    type1 = entry->parameter_type(index+3).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1=dims_t{3,8,src_mask.size()/3 ,24} ;
    in_data_1 = vector_encoder_output1; //read_binary_file(argv[i + 2]);
    data_type1 = ts_type1->dtype()->typecode();    
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_1.data(), (size_t)in_data_1.size()*sizeof(in_data_1[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "decoder create3 "<< vector_encoder_output1.size()/3 <<endl; 

    type1 = entry->parameter_type(index+4).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1 =dims_t{3,8,src_mask.size()/3 ,24} ;
    in_data_1 = vector_encoder_output2; //read_binary_file(argv[i + 2]);
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_1.data(), (size_t)in_data_1.size()*sizeof(in_data_1[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "src_mask.size()/3 "<< src_mask.size()/3 <<endl; 

    auto in_data_3 = src_mask;
    type1 = entry->parameter_type(index+5).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1=dims_t {3,src_mask.size()/3} ;
    data_type1 = ts_type1->dtype()->typecode();  
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_3.data(), (size_t)in_data_3.size()*sizeof(in_data_3[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "decoder create3 "<< src_mask.size()/3 <<endl; 

    type1 = entry->parameter_type(index+6).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1=dims_t {3,8,decode_round[0],24} ;
    in_data_1 = self_k_cache2; //read_binary_file(argv[i + 2]);
    data_type1 = ts_type1->dtype()->typecode();    
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_1.data(), (size_t)in_data_1.size()*sizeof(in_data_1[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "decoder create3 "<< src_mask.size()/3 <<endl; 

    type1 = entry->parameter_type(index+7).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1=dims_t {3,8,decode_round[0],24} ;
    in_data_1 = self_v_cache2; //read_binary_file(argv[i + 2]);
    data_type1 = ts_type1->dtype()->typecode();    
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_1.data(), (size_t)in_data_1.size()*sizeof(in_data_1[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "decoder create3 "<< src_mask.size()/3 <<endl; 

    type1 = entry->parameter_type(index+8).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1=dims_t {3,8,src_mask.size()/3 ,24} ;
    in_data_1 = vector_encoder_output4; //read_binary_file(argv[i + 2]);
    data_type1 = ts_type1->dtype()->typecode();    
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_1.data(), (size_t)in_data_1.size()*sizeof(in_data_1[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "decoder create3 "<< src_mask.size()/3 <<endl; 

    type1 = entry->parameter_type(index+9).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1=dims_t {3,8,src_mask.size()/3 ,24} ;
    in_data_1 = vector_encoder_output5; //read_binary_file(argv[i + 2]);
    data_type1 = ts_type1->dtype()->typecode();    
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_1.data(), (size_t)in_data_1.size()*sizeof(in_data_1[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "decoder create3 "<< src_mask.size()/3 <<endl; 

    type1 = entry->parameter_type(index+10).expect("parameter type out of index");
    ts_type1 = type1.as<tensor_type>().expect("input is not a tensor type");
    // shape = ts_type->shape().as_fixed().unwrap();
    shape_1=dims_t {3,src_mask.size()/3 } ;
    in_data_3 = src_mask; //read_binary_file(argv[i + 2]);
    
    in_tensor1 = host_runtime_tensor::create(data_type1, shape_1, { (gsl::byte *)in_data_3.data(), (size_t)in_data_3.size()*sizeof(in_data_3[0]) }, true, hrt::pool_shared).expect("cannot create input tensor");
    hrt::sync(in_tensor1, sync_op_t::sync_write_back, true).unwrap();
    inputs.push_back(in_tensor1.impl());
    // cout << "decoder create3 "<< src_mask.size()/3 <<endl; 

    // get output
    auto start = std::chrono::steady_clock::now();
    auto return_value = interp.entry_function().expect("no entry_function")->invoke(inputs).expect("run entry_function failed");
    auto stop = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "run decoder kmodel once time: " << duration << " ms, fps = " << 1000 / duration << std::endl;
    auto values = to_values(return_value).expect("unsupported value type");

    auto t = values[0].as<tensor>().expect("value is not a tensor");
    auto unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    auto mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());

    float *output_data_0 = (float *)mapped_buf.buffer().data();//output_data_0  (3,100,12002)
    int data_shape = 1*3*12002;
    // auto out_shape_0 = interp.output_shape(0);
    // cout << "out_shape: " << data_shape << ", "  << endl;
    std::vector<float> output_vec;
    for(int i=0; i<data_shape; i++){
        output_vec.push_back(output_data_0[i]);
    }
    t = values[1].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_1 = (float *)mapped_buf.buffer().data();

    t = values[2].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_2 = (float *)mapped_buf.buffer().data();

    t = values[3].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_3 = (float *)mapped_buf.buffer().data();

    t = values[4].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_4 = (float *)mapped_buf.buffer().data();


    t = values[5].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_5 = (float *)mapped_buf.buffer().data();

    t = values[6].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_6 = (float *)mapped_buf.buffer().data();

    t = values[7].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_7 = (float *)mapped_buf.buffer().data();

    t = values[8].as<tensor>().expect("value is not a tensor");
    unmap_buf = t->to_host().expect("not host")->buffer().as_host().expect("not host buffer");
    mapped_buf =std::move(unmap_buf.map(map_access_t::map_read).unwrap());
    float *output_data_8 = (float *)mapped_buf.buffer().data();

    // int max_len=100;
    for (int b = 0; b < 3; b++){
        for (int h = 0; h < 8; h++){
            for (int l = 0; l < decode_round[0]+1; l++){
                for (int d = 0; d < 24; d++){
                    self_k_cache1_[b][h][l][d] = output_data_5[b*(8*24*(decode_round[0]+1))+h*(24*(decode_round[0]+1))+l*24+d];
                    self_k_cache2_[b][h][l][d]= output_data_6[b*(8*24*(decode_round[0]+1))+h*(24*(decode_round[0]+1))+l*24+d];
                    self_v_cache1_[b][h][l][d]= output_data_7[b*(8*24*(decode_round[0]+1))+h*(24*(decode_round[0]+1))+l*24+d];
                    self_v_cache2_[b][h][l][d]= output_data_8[b*(8*24*(decode_round[0]+1))+h*(24*(decode_round[0]+1))+l*24+d];
                }
            }
        }
    }
    // string save_path="dynamic_shape_230/step_"+ std::to_string(decode_round[0])+".txt";
    // ofstream outfile;
    // outfile.open(save_path, ios::trunc);//打开文件
    // //ios::in可替换
    // // unsigned(a)
    // //ios：：app，表示打开文件后，在写入的文件不会覆盖原文件中的内容，也就是原来文件中的数据会得到保存。
    // //ios::trunc,文件里面的内容会清零
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < output_vec.size(); i++){
    //     outfile << setprecision(8) << output_vec[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。
    // string save_path="dynamic_shape_kvcache_230/token_prob"+ std::to_string(decode_round[0])+".txt";
    // ofstream outfile;
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < data_shape; i++){
    //     outfile << setprecision(8) << output_vec[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/attn"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3* 1*16; i++){
    //     outfile << setprecision(8) << output_data_1[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/inner_states0_step"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3* 1*192; i++){
    //     outfile << setprecision(8) << output_data_2[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/inner_states1_step"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3* 1*192; i++){
    //     outfile << setprecision(8) << output_data_3[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/inner_states2_step"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3* 1*192; i++){
    //     outfile << setprecision(8) << output_data_4[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/decoder_self_kcache1"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // //ios::in可替换
    // // unsigned(a)
    // //ios：：app，表示打开文件后，在写入的文件不会覆盖原文件中的内容，也就是原来文件中的数据会得到保存。
    // //ios::trunc,文件里面的内容会清零
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3*8*24*(decode_round[0]+1); i++){
    //     outfile << setprecision(8) << output_data_5[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/decoder_self_kcache2"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3*8*24*(decode_round[0]+1); i++){
    //     outfile << setprecision(8) << output_data_6[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。
    // save_path="dynamic_shape_kvcache_230/decoder_self_vcache1"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3*8*24*(decode_round[0]+1); i++){
    //     outfile << setprecision(8) << output_data_7[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。

    // save_path="dynamic_shape_kvcache_230/decoder_self_vcache2"+ std::to_string(decode_round[0])+".txt";
    // outfile.open(save_path, ios::trunc);//打开文件
    // outfile << fixed;
    // //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    // for (int i = 0; i < 3*8*24*(decode_round[0]+1); i++){
    //     outfile << setprecision(8) << output_data_8[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    // }
    // outfile.close();//关闭文件，保存文件。





    return output_vec;
}

void decoder_ctc_decode(interpreter& interp, vector<float>& vector_encoder_output_0,vector<float>& vector_encoder_output_1,vector<float>& vector_encoder_output_2,vector<float>& vector_encoder_output_4,vector<float>& vector_encoder_output_5, vector<uint8_t>& encoder_mask, sentencepiece::SentencePieceProcessor& sp_ch, char* output_filePath,int & sent_id){

    // clock_t start2,end2;
    // start2=clock();
    // cout << "beam_size:111" << endl;
    // struct timeval t3,t4;
    // double timeuse;
    // gettimeofday(&t3,NULL);    

    // // load en tokenizer
    // sentencepiece::SentencePieceProcessor sp_ch;
    // const auto ch_status = sp_ch.Load(ch_tokenizer_path);
    // if (!ch_status.ok()){
    //     std::cerr << "load " << ch_tokenizer_path << " failed: " << ch_status.ToString() << std::endl;
    //     return ;
    // }

    // cout << "beam_size:222" << endl;
    // interpreter interp;
    // std::ifstream ifs(kmodel, std::ios::binary);
    // interp.load_model(ifs).expect("Invalid kmodel");  
    // cout << "beam_size:3333" << endl;

    // 解码定义变量，及初始化
    int64_t start_symbol = 2;
    int64_t end_symbol = 3;
    int max_len = 200;
    int64_t decode_round = 0; 
   // beam serch 解码相关参数定义及初始化
    int batch=1;
    const int beam_size=3;
    const int vocab_size=12002;
    const int pad_index=1;
    const int unk_index =3;
    const float unk_penalty =0.0;
    const int eos_index = 2;
    // int src_length=100;
    const int head =8;
    const int dim = 24;
    int src_length = encoder_mask.size() ;
    float max_len_a=0;
    int max_len_b=200;
    int self_max_len=1000;
    std::vector<vector<int>> cands_to_ignore;
    std::vector<std::vector<int>> bbsz_offsets;
    std::vector<std::vector<Finalized_Sent>>  finalized;
    finalized.resize(batch);
    vector<int> finished;
    std::vector<int> batch_idxs;
    int num_remaining_sent =batch;
    std::vector<int> reorder_state;
    // cout << "beam_size:" <<beam_size<< endl;
    // 变量初始化


    for (int i = 0; i < batch; i++) {
        vector<int> tmp1;
        for (int j = 0; j <  beam_size; j++) {
            tmp1.push_back(0);
        }
        cands_to_ignore.push_back(tmp1);
    }
    for (int i = 0; i < batch; i++) {
        std::vector<int> tmp;
        for (int j = 0; j < 1; j++) {
            tmp.push_back(i*beam_size);
        }
        bbsz_offsets.push_back(tmp);
    }
    for (int i = 0; i < batch; i++) {
        finished.push_back(0);
    }
    std::vector<int> src_lengths;
    for (int i = 0; i < batch; ++i) {
        src_lengths.push_back(src_length);
    }
    std::vector<std::vector<std::vector<float>>> scores;
    //scores size [batch][beam_size][max_len]。保存batch样本，beam_size条路径的max_len step的概率
    scores.resize(batch);
    for (int i = 0; i < batch; ++i) {
        scores[i].resize(beam_size);  
        for (int j = 0; j < beam_size; ++j){
            scores[i][j].resize(max_len);
        }
    }
    std::vector<std::vector<std::vector<float>>> scores_temp;//scores size [batch][beam_size][max_len]。保存batch样本，beam_size条路径的max_len step的概率
    scores_temp.resize(batch);
    for (int i = 0; i < batch; ++i) {
        scores_temp[i].resize(beam_size);  
        for (int j = 0; j < beam_size; ++j){
            scores_temp[i][j].resize(max_len);
        }
    }
    // int tokens[batch*beam_size][max_len] = {1};
    std::vector<std::vector<int>>  tokens;
    for (int sample_id=0; sample_id<batch*beam_size; sample_id++){
        std::vector<int> tmp ;
        tmp.push_back(eos_index);// eos =2 与fairseq 设置有关
        for (int i = 1; i < max_len; i++){
            tmp.push_back(1);
        }
        tokens.push_back(tmp);
    }
    // cout<< "encoder over ,decoder begin" << endl;
    std::vector<std::vector<int>>  tokens_temp;
    for (int sample_id=0; sample_id<batch*beam_size; sample_id++){
        std::vector<int> tmp ;
        tmp.push_back(2);// eos =2 @@heshaz存疑
        for (int i = 1; i < max_len; i++){
            tmp.push_back(1);
        }
        tokens_temp.push_back(tmp);
    }
    std::vector<float> encoder_output_0;
    std::vector<float> encoder_output_1;
    std::vector<float> encoder_output_2;
    std::vector<float> encoder_output_4;
    std::vector<float> encoder_output_5;
    // std::vector<uint8_t> encoder_output_3;
    std::vector<float> encoder_output_3;
    

    for (int i = 0; i < src_length; i++){
        for (int j = 0; j < beam_size; j++){
            for (int k = 0; k < 192; k++){
                encoder_output_0.push_back(vector_encoder_output_0[i*192+k]);
            }
        }
    }

    for (int b = 0; b < beam_size; b++){
        for (int h = 0; h < head; h++){
            for (int l = 0; l < src_length; l++){
                    for (int d = 0; d < dim; d++){
                            encoder_output_1.push_back(vector_encoder_output_1[ h*(dim*src_length) + l*dim+d]);
                            encoder_output_2.push_back(vector_encoder_output_2[ h*(dim*src_length) + l*dim+d]);
                            encoder_output_4.push_back(vector_encoder_output_4[ h*(dim*src_length) + l*dim+d]);
                            encoder_output_5.push_back(vector_encoder_output_5[ h*(dim*src_length) + l*dim+d]);
                        }
                }
        }
    }
    // cout<< "step:"<<1111881 << endl;
    // for (int i = 0; i < beam_size; i++){
    //     for (int j = 0; j < src_length; j++){
    //         encoder_output_3.push_back(encoder_mask[j]);
    //     }
    // }  
    for (int i = 0; i < beam_size; i++){
        for (int j = 0; j < src_length; j++){
            if (encoder_mask[j]){
                encoder_output_3.push_back(1);
            }
            else{
                encoder_output_3.push_back(0);
            }   
        }
    } 

    vector<vector<vector<vector<float>>>> self_k_cache1_;
    vector<vector<vector<vector<float>>>> self_v_cache1_;
    vector<vector<vector<vector<float>>>> self_k_cache2_;
    vector<vector<vector<vector<float>>>> self_v_cache2_;
    self_k_cache1_.resize(beam_size);		//row size = m
    self_v_cache1_.resize(beam_size);
    self_k_cache2_.resize(beam_size);		//row size = m
    self_v_cache2_.resize(beam_size);
    for(int i=0; i<beam_size; i++) {
        self_k_cache1_[i].resize(head);	
        self_v_cache1_[i].resize(head);	
        self_k_cache2_[i].resize(head);	
        self_v_cache2_[i].resize(head);	
        for(int j=0; j<head; j++) {
            self_k_cache1_[i][j].resize(max_len);	
            self_v_cache1_[i][j].resize(max_len);	
            self_k_cache2_[i][j].resize(max_len);	
            self_v_cache2_[i][j].resize(max_len);	
            for(int k=0; k<max_len; k++) {
                self_k_cache1_[i][j][k].resize(dim);
                self_v_cache1_[i][j][k].resize(dim);
                self_k_cache2_[i][j][k].resize(dim);
                self_v_cache2_[i][j][k].resize(dim);	
            }
        }
    }
    // float self_k_cache1_[beam_size][head][max_len][dim] = {0};
    // float self_v_cache1_[beam_size][head][max_len][dim] = {0};
    // float self_k_cache2_[beam_size][head][max_len][dim] = {0};
    // float self_v_cache2_[beam_size][head][max_len][dim] = {0};

    auto one_sent_start = std::chrono::steady_clock::now();    
    for (int step = 0; step < max_len; step++)
    {
        cout<< "step:"<<step << endl;
        double timeuse1,timeuse2;
        clock_t start3,end3;
        struct timeval t5,t6,t7,t8,t9,t10,t11,t12,t13, t51,t61,t71;
        double timeuse;
        // if (step==1){

        gettimeofday(&t5,NULL);    

        std::vector<std::vector<float>> cand_scores;
        std::vector<std::vector<int>> cand_indices;
        std::vector<std::vector<int>> cand_bbsz_idx;
        std::vector<std::vector<int>> cand_beams;
        
        for (int sample_id = 0; sample_id < batch; sample_id++) {
            std::vector<int> tmp;
            std::vector<float> tmp_f;
            for (int i = 0; i < 2*beam_size ; ++i) {
                tmp.push_back(0);
                tmp_f.push_back(0);
            }
            cand_scores.push_back(tmp_f);    
            cand_indices.push_back(tmp); 
            cand_beams.push_back(tmp);   
            cand_bbsz_idx.push_back(tmp);  
        }
        vector<int64_t> tgt_seq;
        for (int i = 0; i < beam_size; i++){
            // for (int j = 0; j < max_len; j++){
            // cout<<"beam "<<i<< " ";
            for (int j = 0; j < step+1; j++){
                tgt_seq.push_back(tokens[i][j]);
                cout<< tokens[i][j]<< " ";
            }
            // cout<<endl;

        }  

        // beam search 所以需要将复制beam_size 分输入
        std::vector<std::vector<std::vector<float>>> lprobs;//lprobs size [batch][beam_size][vocab_size]
        
        std::vector<int> corr;
        if (reorder_state.size()>0){
            // TODO:
            if (batch_idxs.size()>0){
                for (int i=0; i<batch_idxs.size(); i++){
                    corr.push_back(batch_idxs[i]-i);
                }
                for  (int i=0; i<corr.size(); i++){
                    for (int j=0; j<beam_size; j++){
                        reorder_state[i*beam_size+ j] = reorder_state[i*beam_size+ j]+corr[i]* beam_size;
                    }
                }  
            } 
        }
        // 模型的kv cache输入
        std::vector<float> self_k_cache1;
        std::vector<float> self_v_cache1;
        std::vector<float> self_k_cache2;
        std::vector<float> self_v_cache2;

        for (int b = 0; b < beam_size; b++){
            for (int h = 0; h < head; h++){
                for (int l = 0; l < step; l++){
                        for (int d = 0; d < dim; d++){
                                self_k_cache1.push_back(self_k_cache1_[b][h][l][d]);
                                self_v_cache1.push_back(self_v_cache1_[b][h][l][d]);
                                self_k_cache2.push_back(self_k_cache2_[b][h][l][d]);
                                self_v_cache2.push_back(self_v_cache2_[b][h][l][d]);
                            }
                    }
            }
        }

    //     string save_path="dynamic_shape_kvcache_230/input/self_k_cache1_step"+ std::to_string(step)+".txt";
    //     ofstream outfile;
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*head*step*dim; i++){
    //         outfile << setprecision(8) << self_k_cache1[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。

    //     save_path="dynamic_shape_kvcache_230/input/self_v_cache1_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*head*step*dim; i++){
    //         outfile << setprecision(8) << self_v_cache1[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。

    //     save_path="dynamic_shape_kvcache_230/input/self_k_cache2_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*head*step*dim; i++){
    //         outfile << setprecision(8) << self_k_cache2[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。

    //     save_path="dynamic_shape_kvcache_230/input/self_v_cache2_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*head*step*dim; i++){
    //         outfile << setprecision(8) << self_v_cache2[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。

    //     save_path="dynamic_shape_kvcache_230/input/tgt_token_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*(step+1); i++){
    //         outfile << setprecision(8) << tgt_seq[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。

    //     save_path="dynamic_shape_kvcache_230/input/encoder_kcache1_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*(step+1); i++){
    //         outfile << setprecision(8) << encoder_output_1[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。
    //    save_path="dynamic_shape_kvcache_230/input/encoder_vcache1_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*(step+1); i++){
    //         outfile << setprecision(8) << encoder_output_2[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。
    //     save_path="dynamic_shape_kvcache_230/input/encoder_kcache2_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*(step+1); i++){
    //         outfile << setprecision(8) << encoder_output_4[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。
    //    save_path="dynamic_shape_kvcache_230/input/encoder_vcache2_step"+ std::to_string(step)+".txt";
    //     outfile.open(save_path, ios::trunc);//打开文件
    //     outfile << fixed;
    //     //写入数据，av是存有double类型数据的一个数组，这里不展示具体定义
    //     for (int i = 0; i < beam_size*(step+1); i++){
    //         outfile << setprecision(8) << encoder_output_5[i] << endl;//fixed << setprecision(8)是为了保留小数点后8位进行写入
    //     }
    //     outfile.close();//关闭文件，保存文件。

        // for (int b = 0; b < beam_size; b++){
        //     for (int h = 0; h < head; h++){
        //         for (int l = 0; l < step; l++){
        //                 for (int d = 0; d < dim; d++){
        //                         cout << self_k_cache1_[b][h][l][d]<<" ";
        //                     }
        //             }
        //     }
        // }
        // cout<< endl;
        // for (int b = 0; b < beam_size; b++){
        //     for (int h = 0; h < head; h++){
        //         for (int l = 0; l < step; l++){
        //                 for (int d = 0; d < dim; d++){
        //                         cout << self_v_cache1_[b][h][l][d]<<" ";
        //                     }
        //             }
        //     }
        // }
        // cout<< endl;
        // for (int b = 0; b < beam_size; b++){
        //     for (int h = 0; h < head; h++){
        //         for (int l = 0; l < step; l++){
        //                 for (int d = 0; d < dim; d++){
        //                         cout << self_k_cache2_[b][h][l][d] <<" ";
        //                     }
        //             }
        //     }
        // }
        // cout<<"self_v_cache2_"<< endl;
        // for (int b = 0; b < beam_size; b++){
        //     for (int h = 0; h < head; h++){
        //         for (int l = 0; l < step; l++){
        //                 for (int d = 0; d < dim; d++){
        //                         cout << self_v_cache2_[b][h][l][d]<<" ";
        //                     }
        //             }
        //     }
        // }
        // cout<< endl;

        // inference
        // cout<< "decoder_single_inference one begin" << endl;

        // auto kmodel_start = std::chrono::steady_clock::now();


        std::vector<float> output_vec = decoder_single_inference(interp, self_k_cache1, self_v_cache1 , self_k_cache2 , self_v_cache2 , 
                                                                encoder_output_1,encoder_output_2,encoder_output_4,encoder_output_5, encoder_output_3, 
                                                                tgt_seq, { step }, lprobs,
                                                                self_k_cache1_, self_v_cache1_ , self_k_cache2_ , self_v_cache2_  
                                                                );
        // auto kmodel_end = std::chrono::steady_clock::now();
        // cout << "run decoder kmodel once time: "<< std::chrono::duration_cast<chrono::milliseconds>(kmodel_end - kmodel_start).count()<< " ms" << endl;


        auto beam_one_start = std::chrono::steady_clock::now();

        
 

        // if (step==1){
        //     // end3=clock();
        //     // printf("decoder once totile time=%f\n",(float)(end3-start3)*1000/CLOCKS_PER_SEC);
        //     gettimeofday(&t7,NULL);
        //     timeuse = (t7.tv_sec - t5.tv_sec)*1000 + (double)(t7.tv_usec - t5.tv_usec)/1000.0;
        //     cout<<"kmodel run time = "<<timeuse << "ms"<<endl;  //输出时间（单位：mｓ）
        // }
        for (int sample_id = 0; sample_id < batch; sample_id++){
            std::vector<std::vector<float>> tmp1;
            std::vector<float> score_tmp;
            // std::vector<float> tmp_fu;
            int tgt_len_cur = sizeof(output_vec)/sizeof(output_vec[0])/vocab_size/beam_size/batch;
            for (int i = 0; i < beam_size; i++){
                std::vector<float> tmp;
                std::vector<float> tmp_exp;
                float sum =0;
                // step >= max_len时，终止。
                if (step >= max_len) {
                    for (int j = 0; j < eos_index; j++) {
                        tmp.push_back(INT_MIN);
                        score_tmp.push_back( -( tmp[j] + scores[sample_id][i][step-1]) ) ;
                    }
                    tmp.push_back(-0.5);
                    score_tmp.push_back( -( -0.5 + scores[sample_id][i][step-1]) ) ;
                    for (int j = eos_index + 1; j < vocab_size; i++) {
                        tmp.push_back(INT_MIN);
                        score_tmp.push_back( -( tmp[j] + scores[sample_id][i][step-1]) ) ;
                    }
                    // cout<<"kmodel run time111 = "<<endl;
                }
                
                else if (step ==0)
                {
                    // cout<<"kmodel run time2222 = "<<endl;
                    // step ==0 只执行一次，因为其他的都相同。
                    if (i ==0){
                        for (int j = 0; j < vocab_size; j++)
                        {   
                            // float data = output_vec[j+i*(tgt_len_cur *vocab_size)+(step)*vocab_size]     ; // 100 =max_len???
                            float data = output_vec[j+i*(vocab_size)] ;
                            float data_exp = std::exp(data);
                            // tmp_exp.push_back(data_exp);
                            sum = sum+ data_exp;
                        }
                        float sum_log = std::log(sum);
                        for (int j = 0; j < vocab_size; j++)
                        {   
                            // float data = output_vec[j+i*(tgt_len_cur *vocab_size)+(step)*vocab_size] ;
                            float data = output_vec[j+i*(vocab_size)] ;
                            if ((j == pad_index)||(j == eos_index) ){
                                tmp.push_back(INT_MIN);
                                score_tmp.push_back( -float(INT_MIN)) ;
                            }
                            else if (j==unk_index )
                            {

                                float data_log = data - sum_log;
                                score_tmp.push_back( -( data_log - unk_penalty ) ) ;
                                // score_tmp.push_back( -( lprobs[sample_id][i][j] - unk_penalty ) ) ;
                            
                            }
                            else{
                                float data_log = data - sum_log;
                                score_tmp.push_back( -( data_log ) ) ;
                                // score_tmp.push_back( -( lprobs[sample_id][i][j] ) ) ;
                            }

                        }
                    }
                }
                else {
                    // cout<<"kmodel run time333 = "<<endl;
                    // cout<<"step = "<<step<<endl;
                    for (int j = 0; j < vocab_size; j++)
                    {   
                         // NOTE: 无kv cache 版本输出shape是[beam, tgt_len, vocab_size]，有kv cache 版本输出shape是[beam, 1, vocab_size]
                        // float data = output_vec[j+i*(tgt_len_cur *vocab_size)+(step)*vocab_size]     ; // 100 =max_len???
                        float data = output_vec[j+i*(vocab_size)] ;
                        float data_exp = std::exp(data);
                        // tmp_exp.push_back(data_exp);
                        sum = sum+ data_exp;
                    }

                    float sum_log = std::log(sum);
     
                    for (int j = 0; j < vocab_size; j++)
                    {   
                        // NOTE: 无kv cache 版本输出shape是[beam, tgt_len, vocab_size]，有kv cache 版本输出shape是[beam, 1, vocab_size]
                        // float data = output_vec[j+i*(tgt_len_cur*vocab_size)+(step)*vocab_size] ;
                        float data = output_vec[j+i*(vocab_size)] ;
                        if (j == pad_index){
                            tmp.push_back(INT_MIN);
                            score_tmp.push_back( -( float(INT_MIN )+ scores[sample_id][i][step-1]) ) ;
                            // score_tmp.push_back( -( float(INT_MIN )+ scores[sample_id][i][step-1]) ) ;
                        }
                        else if (j==unk_index )
                        {
                            float data_log = data - sum_log;
                            score_tmp.push_back( -( data_log - unk_penalty + scores[sample_id][i][step-1]) ) ;
                           
                            // score_tmp.push_back( -(  lprobs[sample_id][i][j] - unk_penalty + scores[sample_id][i][step-1]) ) ;
                        }
                        else{
                            float data_log = data - sum_log;
                            score_tmp.push_back( -( data_log + scores[sample_id][i][step-1]) ) ;
                            // score_tmp.push_back( -( lprobs[sample_id][i][j] + scores[sample_id][i][step-1]) ) ;
                        }
                    }
                    // cout<<"scores[sample_id][i][step-1]:"<<scores[sample_id][i][step-1]<<endl;
                }
                // tmp1.push_back(tmp);
            }   

     

            

            // // cout<<"step = "<<step<<endl;
            // if (step==1){
            //     // end3=clock();
            //     // printf("decoder once totile time=%f\n",(float)(end3-start3)*1000/CLOCKS_PER_SEC);
            //     gettimeofday(&t71,NULL);
            //     timeuse2 = (t71.tv_sec - t5.tv_sec)*1000 + (double)(t71.tv_usec - t5.tv_usec)/1000.0;
            //     cout<<"sort_index_k bea time = "<<timeuse2<<endl;  //输出时间（单位：ｓ）
            // }  

            std::vector<int> indices = sort_index_k_v2(score_tmp, 2*beam_size);


            // if (step==1){
            //     // end3=clock();
            //     // printf("decoder once totile time=%f\n",(float)(end3-start3)*1000/CLOCKS_PER_SEC);
            //     gettimeofday(&t71,NULL);
            //     timeuse2 = (t71.tv_sec - t5.tv_sec)*1000 + (double)(t71.tv_usec - t5.tv_usec)/1000.0;
            //     // cout<<"sort_index_k time = "<<timeuse2<<endl;  //输出时间（单位：ｓ）
            // }  
            for (int k = 0; k<2* beam_size; k++) {
                // cout<< "indices[k]:" << k << " "<< indices[k] << endl;
                cand_indices[sample_id][k] = indices[k]%vocab_size;
                cand_beams[sample_id][k] = indices[k]/vocab_size;
                cand_scores[sample_id][k] = -score_tmp[indices[k]];
            }
            for (int s = 0; s < beam_size; s++) {
                scores[sample_id][s][step] = cand_scores[sample_id][s];
            }

        }

        // cand_bbsz_idx 是最大路径所在行索引。（多batch情况满足）
        for (int i = 0; i < batch; i++) {
            for (int k = 0; k < 2*beam_size; k++) {
               cand_bbsz_idx[i][k] =cand_beams[i][k]+ bbsz_offsets[i][0];
            }
        }
        // eos_mask 判断某条路径是否终止。
        vector<vector<int>> eos_mask;
        eos_mask.resize(batch);
        for (int i = 0; i < batch; ++i) {
            eos_mask[i].resize(2* beam_size);
        }
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < 2* beam_size; j++) {
                eos_mask[i][j] = cand_indices[i][j] == eos_index;
                if (j<beam_size){
                    if (cands_to_ignore[i][j]){
                        eos_mask[i][j] = 0;
                    }                   
                }
            }    
        }
        int eos_count =0;   
        std::vector<int> eos_bbsz_idx;
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < beam_size; j++) {
                if (eos_mask[i][j]) {
                eos_bbsz_idx.push_back(cand_bbsz_idx[i][j]);
                eos_count++ ;
                } 
            }
         } 
         // 有已完成推理的句子，进入统计
        std::vector<int> finalized_sents;
        if (eos_count >0){
            std::vector<float> eos_scores;
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j <  beam_size; j++) {
                    if (eos_mask[i][j]) {
                    eos_scores.push_back(cand_scores[i][j]);
                    } 
                }
            }
            finalized_sents = finalize_hypos(step,eos_bbsz_idx,eos_scores,tokens,
                            scores,finalized,finished,beam_size,max_len);
            // cout<<"step:"<<step<<","<<finalized.size()<<"," <<finalized[0].size()<< endl;
            num_remaining_sent -= finalized_sents.size();
        } 
        
        if ((num_remaining_sent==0) || step >=max_len){
            break;
        }

        
        // 有已完成推理的句子，进入修改参数
        if (finalized_sents.size()>0){
            int new_bsz = batch - finalized_sents.size();
            vector<int> batch_mask;
            // vector<int> batch_idxs;
            vector<vector<int>> eos_mask_tmp;
            // vector<vector<int>> eos_mask;
            vector<vector<int>> cand_beams_tmp;
            // vector<vector<int>> cand_beams;
            vector<vector<int>> bbsz_offsets_tmp;
            vector<vector<float>> cand_scores_tmp;
            vector<vector<int>> cand_indices_tmp;
            vector<vector<int>> cands_to_ignore_tmp;
            vector<int> src_lengths_tmp;
            // vector<vector<int>> bbsz_offsets;eos_mask_tmp
            vector<vector<vector<float>>> scores_tmp;
            vector<vector<int>> tokens_tmp;
            vector<vector<int>> cand_bbsz_idx_tmp;
            // vector<vector<int>> cand_bbsz_idx;
            for (int j = 0; j < batch; j++) {
                batch_mask.push_back(1);
            }
            for (int j :finalized_sents) {
                batch_mask[j] =  0; //false
            }
            for (int j=0;j< batch_mask.size();j++) {
                if (batch_mask[j]){
                    batch_idxs.push_back(j);
                }
            }

            // cand_beams_tmp = cand_beams;
            for (int j :batch_idxs) {
                // cand_beams.push_back(cand_beams_tmp[j]);
                eos_mask_tmp.push_back(eos_mask[j]);
                cand_beams_tmp.push_back(cand_beams[j]);
                cand_scores_tmp.push_back(cand_scores[j]);
                cand_indices_tmp.push_back(cand_indices[j]);
                src_lengths_tmp.push_back(src_lengths[j]);
                cands_to_ignore_tmp.push_back(cands_to_ignore[j]);

            }  
            eos_mask = eos_mask_tmp;
            cand_beams = cand_beams_tmp;
            cand_scores=  cand_scores_tmp;
            cand_indices = cand_indices_tmp;
            src_lengths = src_lengths_tmp;
            cands_to_ignore = cands_to_ignore_tmp;
            // bbsz_offsets_tmp = bbsz_offsets;
            for (int i=0 ; i<new_bsz;i++) {
                vector<int> tmp; 
                tmp.push_back(bbsz_offsets[i][0]);
                bbsz_offsets_tmp.push_back(tmp) ;
            }
            bbsz_offsets =  bbsz_offsets_tmp;
            for (int i=0; i<cand_beams.size();i++){
                vector<int> tmp; 
                for (int j=0; j<cand_beams[0].size();j++){
                    tmp.push_back(cand_beams[i][j]+bbsz_offsets[i][0]);
                    // cand_bbsz_idx_tmp[i][j] = cand_beams[i][j]+bbsz_offsets[i][0];
                }  
                cand_bbsz_idx_tmp.push_back(tmp);       
            }
            cand_bbsz_idx =cand_bbsz_idx_tmp;
            for (int j :batch_idxs) {
                scores_tmp.push_back(scores[j]);
                for (int i=0; i<beam_size;i++){
                    tokens_tmp.push_back(tokens[j*beam_size+i]);
                }
            }
            batch = new_bsz;
            scores =scores_tmp;
            tokens= tokens_tmp;
        }
        else{
            vector<int> batch_idxs;
        }



        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < beam_size; j++) {
                eos_mask[i][j] = !((!cands_to_ignore[i][j]) & (!eos_mask[i][j]));
            }
        }
        int cand_size =2*beam_size;
        std::vector<vector<float>> active_mask;
        vector<int> cand_offsets;
        for (int i=0;i<2*beam_size;i++){
            cand_offsets.push_back(i);
        }
        for (int sample_id  = 0; sample_id  < batch; sample_id ++) {
            vector<float> tmp;
            for (int i = 0; i < 2*beam_size; i++) {
                if (eos_mask[sample_id][i]) {
                    // active_mask[sample_id][i] = cand_size + cand_offsets[i];  
                    tmp.push_back(cand_size + cand_offsets[i])    ;        
                } else {
                    // active_mask[sample_id][i] = cand_offsets[i];
                    tmp.push_back(cand_offsets[i]);
                } 
            }
            active_mask.push_back(tmp)   ;        
        }
        std::vector<vector<float>> new_cands_to_ignore;
        // std::vector<vector<int>> cands_to_ignore;
        std::vector<vector<int>> active_hypos;

        for (int sample_id  = 0; sample_id  < batch; sample_id ++) {
            std::vector<float> values ;
            std::vector<int> indices;
            // std::vector<int> indices_sort= sort_index(active_mask[sample_id]);
            std::vector<int> indices_sort = sort_index_k_v2(active_mask[sample_id], beam_size);

            // std::vector<int> indices_sort = sort_index_k(active_mask[sample_id], 2*beam_size);
            // topk(active_mask[sample_id], beam_size, values, indices);
            for (int i = 0; i < beam_size; i++) {
                values.push_back(active_mask[sample_id][indices_sort[i]]);
                indices.push_back(indices_sort[i]);
            }
            new_cands_to_ignore.push_back(values);
            active_hypos.push_back(indices);
            // std::vector<bool> tmp ;
            for (int i = 0; i < beam_size; i++) {          
                // tmp.push_back(values[i]>cand_size);
                cands_to_ignore[sample_id][i] = values[i]>cand_size;
            }      
        }
        vector<int> active_bbsz_idx;
        vector<float> active_scores;
        for (int i = 0; i < active_hypos.size(); i++) {
            for (int j = 0; j < active_hypos[0].size(); j++) {
                active_bbsz_idx.push_back(cand_bbsz_idx[i][active_hypos[i][j]]);
                active_scores.push_back(cand_scores[i][active_hypos[i][j]]);
            } 
        }
        // 根据当前step的结果， 更新token。active_bbsz_idx保存的是保留下来的句子的索引，可重复。
        for (int sample_id  = 0; sample_id  < batch; sample_id ++) {
            for (int i = 0; i < beam_size; i++) {
                for (int j = 0; j < step+1; j++) {
                    // tokens[i][j] = tokens[active_bbsz_idx[i*beam_size+i]][j] ;
                    tokens_temp[i+sample_id*beam_size][j] = tokens[active_bbsz_idx[i+sample_id*beam_size]][j];
                }
            }
        }
        for (int i = 0; i < beam_size; i++) {
            for (int j = 0; j < step+1; j++) {
                // tokens[i][j] = tokens[active_bbsz_idx[i*beam_size+i]][j] ;
                tokens[i][j] = tokens_temp[i][j];
            }
        }

    std::vector<std::vector<std::vector<std::vector<float>>>>  k1_temp;
    std::vector<std::vector<std::vector<std::vector<float>>>>  k2_temp;
    std::vector<std::vector<std::vector<std::vector<float>>>>  v1_temp;
    std::vector<std::vector<std::vector<std::vector<float>>>>  v2_temp;

    k1_temp.resize(beam_size);		//row size = m
    k2_temp.resize(beam_size);
    v1_temp.resize(beam_size);		//row size = m
    v2_temp.resize(beam_size);
    for(int i=0; i<beam_size; i++) {
        k1_temp[i].resize(head);	
        k2_temp[i].resize(head);	
        v1_temp[i].resize(head);	
        v2_temp[i].resize(head);	
        for(int j=0; j<head; j++) {
            k1_temp[i][j].resize(step+1);	
            k2_temp[i][j].resize(step+1);	
            v1_temp[i][j].resize(step+1);	
            v2_temp[i][j].resize(step+1);	
            for(int k=0; k<step+1; k++) {
                k1_temp[i][j][k].resize(dim);
                k2_temp[i][j][k].resize(dim);
                v1_temp[i][j][k].resize(dim);
                v2_temp[i][j][k].resize(dim);	
            }
        }
    }
        // beam search中更新kv cache
        for (int sample_id  = 0; sample_id  < batch; sample_id ++) {
            for (int i = 0; i < beam_size; i++) {
                for (int k = 0; k < head; k++) {
                    for (int j = 0; j < step+1; j++) {
                        for (int d = 0; d < dim; d++) {
                            // tokens[i][j] = tokens[active_bbsz_idx[i*beam_size+i]][j] ;
                            // cout<< "hhhhffh" <<active_bbsz_idx[i+sample_id*beam_size] << " "<< 2  << endl;
                            k1_temp[i+sample_id*beam_size][k][j][d] = self_k_cache1_[active_bbsz_idx[i+sample_id*beam_size]][k][j][d] ;
                            // cout<< "hhhhffh" <<active_bbsz_idx[i+sample_id*beam_size] << " "<< 2  << endl;
                            v1_temp[i+sample_id*beam_size][k][j][d] = self_v_cache1_[active_bbsz_idx[i+sample_id*beam_size]][k][j][d] ;
                            // cout<< "hhhhffh" <<active_bbsz_idx[i+sample_id*beam_size] << " "<< 2  << endl;
                            k2_temp[i+sample_id*beam_size][k][j][d] = self_k_cache2_[active_bbsz_idx[i+sample_id*beam_size]][k][j][d] ;
                            // cout<< "hhhhffh" <<active_bbsz_idx[i+sample_id*beam_size] << " "<< 2  << endl;
                            v2_temp[i+sample_id*beam_size][k][j][d] = self_v_cache2_[active_bbsz_idx[i+sample_id*beam_size]][k][j][d] ;
                        }
                    }
                }
            }
        }
        // cout<< "hhhhwehhhh" <<1 << " "<< 2  << endl;
        for (int sample_id  = 0; sample_id  < batch; sample_id ++) {
            for (int i = 0; i < beam_size; i++) {
                for (int k = 0; k < head; k++) {
                    for (int j = 0; j < step+1; j++) {
                        for (int d = 0; d < dim; d++) {
                            // tokens[i][j] = tokens[active_bbsz_idx[i*beam_size+i]][j] ;
                            self_k_cache1_[i+sample_id*beam_size][k][j][d] = k1_temp[i+sample_id*beam_size][k][j][d] ;
                            self_v_cache1_[i+sample_id*beam_size][k][j][d] = v1_temp[i+sample_id*beam_size][k][j][d] ;
                            self_k_cache2_[i+sample_id*beam_size][k][j][d] = k2_temp[i+sample_id*beam_size][k][j][d] ;
                            self_v_cache2_[i+sample_id*beam_size][k][j][d] = v2_temp[i+sample_id*beam_size][k][j][d] ;
                        }
                    }
                }
            }
        }
        // cout<< "hhhhhhhhhh" <<1 << " "<< 2  << endl;
        // if (step==1){

        //     gettimeofday(&t11,NULL);
        //     timeuse = (t11.tv_sec - t5.tv_sec)*1000 + (double)(t11.tv_usec - t5.tv_usec)/1000.0;
        //     cout<<"t11 time = "<<timeuse<<endl;  //输出时间（单位：ｓ）
        // }
      // std::vector<vector<int>> active_scores;
        std::vector<vector<int>> select_index;
        for (int i = 0; i < batch; i++) {
            std::vector<int> tmp ;
            for (int j = 0; j < beam_size; j++) {
                tmp.push_back(cand_indices[i][active_hypos[i][j]]);
            } 
            select_index.push_back(tmp);
        }

        for (int i = 0; i < active_hypos.size(); i++) {
            for (int j = 0; j < active_hypos[0].size(); j++) {
                // tokens[i*beam_size+j][step+1] = select_index[i][active_hypos[i][j]];
                tokens[i*beam_size+j][step+1] = select_index[i][j];
            } 
        } 
        // scores size （batch，beam_size，max_len）,python 中shape为（batch*beam_size，max_len）
        if (step>0){
            // 根据当前step的结果， 更新token。active_bbsz_idx保存的是保留下来的句子的索引，可重复。
            for (int sample_id = 0; sample_id < batch; sample_id++) {
                for (int i = 0; i < beam_size; i++) {
                    for (int j = 0; j < step; j++) {
                        // scores[sample_id][i][j] = scores[sample_id][active_bbsz_idx[sample_id*beam_size+i]][j] ;
                        scores_temp[sample_id][i][j] = scores[sample_id][active_bbsz_idx[sample_id*beam_size+i]-sample_id*beam_size][j] ;
                    }
                }
            }
            for (int sample_id = 0; sample_id < batch; sample_id++) {
                for (int i = 0; i < beam_size; i++) {
                    for (int j = 0; j < step; j++) {
                        // scores[sample_id][i][j] = scores[sample_id][active_bbsz_idx[sample_id*beam_size+i]][j] ;
                        scores[sample_id][i][j] = scores_temp[sample_id][i][j];
                    }
                }
            }   
        } 
        std::vector<vector<float>> select_index_2;
        for (int i = 0; i < batch; i++) {
            std::vector<float> tmp ;
            for (int j = 0; j < beam_size; j++) {
                tmp.push_back(cand_scores[i][active_hypos[i][j]]);
            } 
            select_index_2.push_back(tmp);
        }       
        // scores size 是[batch][beam_size][max_len]
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < beam_size; j++) {
                // scores[i][j][step] = select_index_2[i][active_hypos[i][j]];
                scores[i][j][step] = select_index_2[i][j];
            } 
        }
        reorder_state = active_bbsz_idx;
        
        // if (step==1){
        //     // end3=clock();
        //     // printf("decoder once totile time=%f\n",(float)(end3-start3)*1000/CLOCKS_PER_SEC);
        //     gettimeofday(&t6,NULL);
        //     timeuse = (t6.tv_sec - t5.tv_sec)*1000 + (double)(t6.tv_usec - t5.tv_usec)/1000.0;
        //     cout<<"decoder beam search once totile time = "<<timeuse<< "ms"<<endl;  //输出时间（单位：ｓ）
        // }

        auto beam_one_end = std::chrono::steady_clock::now();
        cout << "decode beam once time: "<< std::chrono::duration_cast<chrono::milliseconds>(beam_one_end - beam_one_start).count()<< " ms" << endl;

    }
    auto one_sent_end = std::chrono::steady_clock::now();
    cout << "decode beam once time: "<< std::chrono::duration_cast<chrono::milliseconds>(one_sent_end - one_sent_start).count()<< " ms" << endl;
    // cout<< "decoder run over" << endl; 
    std::vector<std::vector<Finalized_Sent>> finalized_new;
    // batch 会改变，所以sample_id<finalized.size()
    for (int sample_id=0; sample_id<finalized.size(); sample_id++){
        std::vector<float> score_tmp ;
        for (int i = 0; i < beam_size; i++) {
            // 取负数是因为，要降序排列
            // cout<< 2 << endl;
            // cout<< finalized.size() <<","<< finalized[0].size() << endl;
            score_tmp.push_back(-finalized[sample_id][i].scores_tmp);
        }
        // cout<< 3 << endl;
        // std::vector<int> indices = sort_index(score_tmp);
        std::vector<int> indices = sort_index_k_v2(score_tmp, beam_size);

        std::vector<Finalized_Sent> tmp ;
        for (int i = 0; i < beam_size; i++) {
            tmp.push_back(finalized[sample_id][indices[i]]);
            // cout<< 4 << endl;
        }
        // cout<< 5 << endl;
        finalized_new.push_back(tmp);
    }

    // for (int sample_id=0; sample_id<finalized.size(); sample_id++){
    //     for (int i = 0; i < beam_size; i++) {
    //         cout<<"beam_size:"<< i <<endl;
    //         for (int j = 0; j < finalized_new[sample_id][i].token.size(); j++) {
    //             cout<< finalized_new[sample_id][i].token[j]<< " ";
    //         }
    //         cout<< finalized_new[sample_id][i].scores_tmp;
    //         cout<<endl;
    //     }  
    //     cout<< finalized_new[sample_id][0].scores_tmp - finalized_new[sample_id][1].scores_tmp<<endl;
    // }

    // decoder的输出解码，并保存成txt。支持多batch
    // vector<vector<int>> output;
    for (int sample_id=0; sample_id<finalized.size(); sample_id++){
        vector<int> tgt_seq_origin;
        string translation_text;
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < finalized_new[sample_id][i].token.size(); j++) {
                tgt_seq_origin.push_back(finalized_new[sample_id][i].token[j]);
                    // detokenize
                sp_ch.Decode(tgt_seq_origin, &translation_text);
            }   
        }
        // 保存输出到txt文件。output_filePath  "output1/pred_text_zh2en_beam_510.txt"
        ofstream out_txt_file;
        out_txt_file.open(output_filePath, std::ios::app);
        // out_txt_file << fixed;
        out_txt_file << translation_text << endl;
        cout<< "save over" << endl;
        out_txt_file.close();
        // cout<< "save over1" << endl;
    }
    // cout<< "save over2" << endl;
    return;

}


// 导出onnx模型，采用定长输入。kmodel模型推理时根据step取出当前步probs。做beam search 计算。
int main(int argc, char *argv[]) {

    // struct timeval t1,t2;
    // double timeuse;
    // gettimeofday(&t1,NULL); 
    auto start = std::chrono::steady_clock::now();
    string en_tokenizer_path = argv[1]; 
    string ch_tokenizer_path = argv[3]; 
    sentencepiece::SentencePieceProcessor sp_en;
    const auto en_status = sp_en.Load(en_tokenizer_path);
    if (!en_status.ok()) {
        std::cerr << en_status.ToString() << std::endl;
    }
    // end11=clock();
    // printf("load sp_en totile time=%f\n",(float)(end11-start11)*1000/CLOCKS_PER_SEC);
    auto stop = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "load sp_en totile time=%f\n" << duration << " ms, fps = " << 1000 / duration << std::endl;

    // load ch tokenizer
    start = std::chrono::steady_clock::now();
    sentencepiece::SentencePieceProcessor sp_ch;
    const auto ch_status = sp_ch.Load(ch_tokenizer_path);
    if (!ch_status.ok()){
        std::cerr << "load " << ch_tokenizer_path << " failed: " << ch_status.ToString() << std::endl;
        // return ;
    }
    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "load sp_ch totile time=%f\n" << duration << " ms, fps = " << 1000 / duration << std::endl;

    char *output_filePath = argv[6];
    // encoder preprocess
    char *filePath = argv[5];
    ifstream file;
    file.open(filePath,ios::in);
    //连续以行为单位，读取 in.txt 文件中的数据
    std::string input_text;
    int sent_id=0;
    char* encoder_kmodel_path = argv[2];
    char* decoder_kmodel_path = argv[4];

    start = std::chrono::steady_clock::now();

    interpreter interp_en;
    std::ifstream ifs0(encoder_kmodel_path, std::ios::binary);
    interp_en.load_model(ifs0).expect("Invalid kmodel");  
    
    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "load encoder: " << duration << " ms, fps = " << 1000 / duration << std::endl;    

    start = std::chrono::steady_clock::now();

    interpreter interp_de;
    std::ifstream ifs1(decoder_kmodel_path, std::ios::binary);
    interp_de.load_model(ifs1).expect("Invalid kmodel");  

    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "load decoder: " << duration << " ms, fps = " << 1000 / duration << std::endl;   

    while (getline(file,input_text)) {
        struct timeval t1,t2;
        double timeuse1;
        gettimeofday(&t1,NULL);

        vector<int64_t> source_seq;
        vector<uint8_t> source_mask;
        // clock_t start1,end1;
        // start1=clock();

        struct timeval t11,t21;
        double timeuse2;
        gettimeofday(&t11,NULL);    

        // input_text="Why has President Trump given up so much?";
        encoder_preprocess(sp_en, input_text, source_seq, source_mask); //source_seq/source_mask：（1，source_len）,


        gettimeofday(&t21,NULL);
        timeuse2 = (t21.tv_sec - t11.tv_sec)*1000 + (double)(t21.tv_usec - t11.tv_usec)/1000.0;
        cout<<"prepocess totile time = "<<timeuse2<<endl;  //输出时间（单位：ｓ）

        // encoder inference
        
        //float* encoder_output_data;
        vector<float> vector_encoder_output_0;
        vector<float> vector_encoder_output_1;
        vector<float> vector_encoder_output_2;
        vector<float> vector_encoder_output_4;
        vector<float> vector_encoder_output_5;
        // cout << "encoder in " << endl ;

        // clock_t start2,end2;
        // start2=clock();
        
        struct timeval t12,t22;
        double timeuse3;
        gettimeofday(&t12,NULL);    


        encoder_inference(interp_en, source_seq,  vector_encoder_output_0, vector_encoder_output_1, vector_encoder_output_2, vector_encoder_output_4,vector_encoder_output_5);
        
        // end2=clock();
        // printf("encoder totile time=%f\n",(float)(end2-start2)*1000/CLOCKS_PER_SEC);

        gettimeofday(&t22,NULL);
        timeuse3 = (t22.tv_sec - t12.tv_sec)*1000 + (double)(t22.tv_usec - t12.tv_usec)/1000.0;
        cout<<"encoder totile time = "<<timeuse3<<endl;  //输出时间（单位：ｓ）
        // cout << "encoder out " << endl ;
        /// Step5: 机器翻译_Decoder

        // decoder inference and postprocess
        // string ch_tokenizer_path = argv[3]; 
        string translation_text;
        // char* decoder_kmodel_path = argv[4];
        // cout << "decoder in " << endl ;
        start = std::chrono::steady_clock::now();
        decoder_ctc_decode(interp_de, vector_encoder_output_0, vector_encoder_output_1, vector_encoder_output_2, vector_encoder_output_4,vector_encoder_output_5,source_mask, sp_ch, output_filePath,sent_id);

        stop = std::chrono::steady_clock::now();
        duration = std::chrono::duration<double, std::milli>(stop - start).count();
        std::cout << "decoder totile time : " << duration << " ms, fps = " << 1000 / duration << std::endl;
        
        sent_id+=1;

        time_t now_end = time(NULL);
        tm* tm_t_end = localtime(&now_end);
        std::stringstream end_time;
        end_time << "year:" << tm_t_end->tm_year + 1900 << " month:" << tm_t_end->tm_mon + 1 << " day:" << tm_t_end->tm_mday
            << " hour:" << tm_t_end->tm_hour << " minute:" << tm_t_end->tm_min << " second:" << tm_t_end->tm_sec;
    
        std::cout << end_time.str()<< endl;

        gettimeofday(&t2,NULL);
        timeuse1 = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;

        cout<<"one sentence total time = "<<timeuse1<<endl;  //输出时间（单位：ｓ）


    }
    
    file.close();
    return 0;
}
