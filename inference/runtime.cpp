#include "runtime.hpp"
#include "runtime.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace lw::inference {
namespace {
float sigmoid(float x) {if(x>=0)return 1.0f/(1.0f+std::exp(-x));float e=std::exp(x);return e/(1.0f+e);}
float dot(const float* a,const float* b,int n) {float value=0;for(int i=0;i<n;++i)value+=a[i]*b[i];return value;}
bool supported(int inputs,int hidden) {return (inputs==22&&hidden==48)||(inputs==38&&hidden==64);}
void valid_ids(const std::uint64_t* ids,int count) {
    if(count<0||count>4096||(!ids&&count))throw std::invalid_argument("invalid identity count");
    std::set<std::uint64_t> unique;
    for(int j=0;j<count;++j)if(ids[j]==0||!unique.insert(ids[j]).second)throw std::invalid_argument("invalid or duplicate identity");
}
}
std::size_t Policy::weight_count(int inputs,int hidden) {
    if(!supported(inputs,hidden))throw std::invalid_argument("unsupported architecture");
    return std::size_t(hidden)*inputs+6*std::size_t(hidden)*hidden+16*hidden+9;
}
Policy::Policy(int i,int h,const std::vector<float>& w):inputs(i),hidden_size(h),weights(w) {
    if(weights.size()!=weight_count(i,h))throw std::invalid_argument("weight count mismatch");
    for(float value:weights)if(!std::isfinite(value)||std::abs(value)>1000)throw std::invalid_argument("invalid model weight");
}
Policy Policy::load(std::istream& stream) {
    if(std::endian::native!=std::endian::little||sizeof(float)!=4||!std::numeric_limits<float>::is_iec559)
        throw std::runtime_error("runtime requires little-endian IEEE float32");
    char magic[8];std::array<std::uint32_t,5> header{};
    stream.read(magic,8);stream.read(reinterpret_cast<char*>(header.data()),sizeof(header));
    const char expected[8]={'L','W','G','R','U','0','1','\0'};
    if(!stream||std::memcmp(magic,expected,8)||header[0]!=1||header[3]!=8||header[1]>1000||header[2]>1000)
        throw std::invalid_argument("invalid model header");
    auto count=weight_count(int(header[1]),int(header[2]));
    if(header[4]!=count)throw std::invalid_argument("invalid model size");
    std::vector<float> weights(count);stream.read(reinterpret_cast<char*>(weights.data()),std::streamsize(count*sizeof(float)));
    if(!stream||stream.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("truncated or trailing model data");
    return Policy(int(header[1]),int(header[2]),weights);
}
std::vector<float> Policy::evaluate(const float* input,const std::vector<float>& prior,float* logits,float& value) const {
    int h=hidden_size;const float* w=weights.data();
    const float* ew=w;w+=h*inputs;const float* eb=w;w+=h;
    const float* wi=w;w+=3*h*h;const float* wh=w;w+=3*h*h;
    const float* bi=w;w+=3*h;const float* bh=w;w+=3*h;
    const float* aw=w;w+=actions*h;const float* ab=w;w+=actions;
    const float* cw=w;w+=h;float cb=*w;
    std::vector<float> encoded(h),result(h);
    for(int k=0;k<h;++k)encoded[k]=std::tanh(dot(ew+k*inputs,input,inputs)+eb[k]);
    for(int k=0;k<h;++k) {
        float ir=dot(wi+k*h,encoded.data(),h)+bi[k],hr=dot(wh+k*h,prior.data(),h)+bh[k];
        float iz=dot(wi+(h+k)*h,encoded.data(),h)+bi[h+k],hz=dot(wh+(h+k)*h,prior.data(),h)+bh[h+k];
        float in=dot(wi+(2*h+k)*h,encoded.data(),h)+bi[2*h+k],hn=dot(wh+(2*h+k)*h,prior.data(),h)+bh[2*h+k];
        float reset=sigmoid(ir+hr),update=sigmoid(iz+hz);
        float candidate=std::tanh(in+reset*hn);
        result[k]=(1-update)*candidate+update*prior[k];
    }
    for(int k=0;k<actions;++k)logits[k]=dot(aw+k*h,result.data(),h)+ab[k];
    value=dot(cw,result.data(),h)+cb;
    return result;
}
void Policy::decide(const std::uint64_t* ids,const float* features,const std::uint8_t* masks,
                    int count,std::int32_t* choices,float* logits,float* values) {
    if(count<0||count>256)throw std::invalid_argument("batch limit is 256");
    valid_ids(ids,count);
    if(count&&(!features||!masks||!choices||!logits||!values))throw std::invalid_argument("null batch buffer");
    std::size_t new_ids=0;
    for(int j=0;j<count;++j) {
        if(!memory.contains(ids[j]))++new_ids;
        bool enabled=false;
        for(int k=0;k<inputs;++k)if(!std::isfinite(features[j*inputs+k])||std::abs(features[j*inputs+k])>16)
            throw std::invalid_argument("invalid feature");
        for(int k=0;k<actions;++k){if(masks[j*actions+k]>1)throw std::invalid_argument("invalid action mask");enabled|=masks[j*actions+k]!=0;}
        if(!enabled)throw std::invalid_argument("empty action mask");
    }
    if(memory.size()+new_ids>4096)throw std::invalid_argument("controller identity capacity reached");
    std::vector<std::vector<float>> pending;pending.reserve(count);
    for(int j=0;j<count;++j) {
        auto found=memory.find(ids[j]);std::vector<float> zero;
        if(found==memory.end())zero.assign(hidden_size,0);
        auto& prior=found==memory.end()?zero:found->second;
        pending.push_back(evaluate(features+j*inputs,prior,logits+j*actions,values[j]));
        int best=-1;
        for(int k=0;k<actions;++k)if(masks[j*actions+k]&&(best<0||logits[j*actions+k]>logits[j*actions+best]))best=k;
        choices[j]=best;
    }
    auto updated=memory;
    for(int j=0;j<count;++j)updated[ids[j]]=std::move(pending[j]);
    memory.swap(updated);
}
void Policy::restore(const std::uint64_t* ids,const float* hidden,int count) {
    valid_ids(ids,count);if(count&&!hidden)throw std::invalid_argument("null memory buffer");
    std::map<std::uint64_t,std::vector<float>> restored;
    for(int j=0;j<count;++j) {
        std::vector<float> state(hidden+j*hidden_size,hidden+(j+1)*hidden_size);
        for(float value:state)if(!std::isfinite(value)||std::abs(value)>1.00001f)throw std::invalid_argument("invalid recurrent state");
        restored.emplace(ids[j],std::move(state));
    }
    memory=std::move(restored);
}
}

namespace {
thread_local std::string policy_error;
std::filesystem::path utf8_path(const char* path) {std::string bytes(path);return std::filesystem::path(std::u8string(bytes.begin(),bytes.end()));}
lw::inference::Policy& policy(void* handle){if(!handle)throw std::invalid_argument("null policy");return *static_cast<lw::inference::Policy*>(handle);}
template<class F> int32_t call(F f){try{policy_error.clear();return f();}catch(const std::exception& e){policy_error=e.what();return -1;}catch(...){policy_error="native inference exception";return -1;}}
}
int32_t lw_policy_abi(void){return 1;}
void* lw_policy_load(const char* path){try{policy_error.clear();if(!path)throw std::invalid_argument("null model path");std::ifstream stream(utf8_path(path),std::ios::binary);if(!stream)throw std::invalid_argument("cannot open native model");return new lw::inference::Policy(lw::inference::Policy::load(stream));}catch(const std::exception& e){policy_error=e.what();return nullptr;}catch(...){policy_error="native inference exception";return nullptr;}}
void lw_policy_destroy(void* handle){delete static_cast<lw::inference::Policy*>(handle);}
int32_t lw_policy_inputs(void* handle){return call([&]{return policy(handle).inputs;});}
int32_t lw_policy_hidden_size(void* handle){return call([&]{return policy(handle).hidden_size;});}
int32_t lw_policy_decide(void* handle,const uint64_t* ids,const float* features,const uint8_t* masks,int32_t count,int32_t* choices,float* logits,float* values){return call([&]{policy(handle).decide(ids,features,masks,count,choices,logits,values);return 0;});}
int32_t lw_policy_get_hidden(void* handle,uint64_t id,float* output){return call([&]{auto& p=policy(handle);if(!output||id==0)throw std::invalid_argument("invalid memory query");auto found=p.memory.find(id);if(found==p.memory.end())std::fill(output,output+p.hidden_size,0);else std::copy(found->second.begin(),found->second.end(),output);return 0;});}
int32_t lw_policy_set_hidden(void* handle,const uint64_t* ids,const float* hidden,int32_t count){return call([&]{policy(handle).restore(ids,hidden,count);return 0;});}
const char* lw_policy_error(void){return policy_error.c_str();}
