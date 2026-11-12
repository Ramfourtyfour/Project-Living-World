#include "social_runtime.hpp"
#include "social_runtime.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace lw::social_inference {
namespace {
float dot(const float* a,const float* b,int n) {float result=0;for(int i=0;i<n;++i)result+=a[i]*b[i];return result;}
float sigmoid(float x) {if(x>=0)return 1/(1+std::exp(-x));float e=std::exp(x);return e/(1+e);}
void identities(const std::uint64_t* ids,int count) {
    if(count<0||count>4096||(!ids&&count))throw std::invalid_argument("invalid identity batch");
    std::set<std::uint64_t> unique;
    for(int i=0;i<count;++i)if(!ids[i]||!unique.insert(ids[i]).second)throw std::invalid_argument("invalid or duplicate identity");
}
}
Policy::Policy(const std::vector<float>& values):weights(values) {
    if(weights.size()!=weight_count)throw std::invalid_argument("invalid social weight count");
    for(float v:weights)if(!std::isfinite(v)||std::abs(v)>1000)throw std::invalid_argument("invalid social weight");
}
Policy Policy::load(std::istream& stream) {
    if(std::endian::native!=std::endian::little||sizeof(float)!=4||!std::numeric_limits<float>::is_iec559)throw std::runtime_error("requires little-endian IEEE float32");
    char magic[8];std::array<std::uint32_t,6> header{};
    stream.read(magic,8);stream.read(reinterpret_cast<char*>(header.data()),sizeof(header));
    const char expected[8]={'L','W','S','L','A','T',char('0'+version),'\0'};
    if(!stream||std::memcmp(magic,expected,8)||header!=std::array<std::uint32_t,6>{version,inputs,candidate_inputs,hidden_size,max_candidates,weight_count})throw std::invalid_argument("invalid social model header");
    std::vector<float> weights(weight_count);stream.read(reinterpret_cast<char*>(weights.data()),weight_count*sizeof(float));
    if(!stream||stream.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("truncated or trailing social model data");
    return Policy(weights);
}
std::vector<float> Policy::evaluate(const float* input,const std::vector<float>& prior,const float* candidates,int count,float* logits,float& value) const {
    constexpr int h=hidden_size,c=candidate_inputs;
    const float* w=weights.data();const float* ew=w;w+=h*inputs;const float* eb=w;w+=h;
    const float* wi=w;w+=3*h*h;const float* wh=w;w+=3*h*h;const float* bi=w;w+=3*h;const float* bh=w;w+=3*h;
    const float* aw=w;w+=8*h;const float* ab=w;w+=8;
    const float* sw=w;w+=h*(h+c);const float* sb=w;w+=h;const float* ow=w;w+=h;float ob=*w++;
    const float* cw=w;w+=h;float cb=*w;
    std::array<float,h> encoded{},base{};std::array<float,8> survival{};std::vector<float> result(h);
    for(int k=0;k<h;++k)encoded[k]=std::tanh(dot(ew+k*inputs,input,inputs)+eb[k]);
    for(int k=0;k<h;++k) {
        float ir=dot(wi+k*h,encoded.data(),h)+bi[k],hr=dot(wh+k*h,prior.data(),h)+bh[k];
        float iz=dot(wi+(h+k)*h,encoded.data(),h)+bi[h+k],hz=dot(wh+(h+k)*h,prior.data(),h)+bh[h+k];
        float in=dot(wi+(2*h+k)*h,encoded.data(),h)+bi[2*h+k],hn=dot(wh+(2*h+k)*h,prior.data(),h)+bh[2*h+k];
        float reset=sigmoid(ir+hr),update=sigmoid(iz+hz),candidate=std::tanh(in+reset*hn);
        result[k]=(1-update)*candidate+update*prior[k];
    }
    for(int k=0;k<8;++k)survival[k]=dot(aw+k*h,result.data(),h)+ab[k];
    for(int k=0;k<h;++k)base[k]=dot(sw+k*(h+c),result.data(),h);
    for(int j=0;j<count;++j) {
        const float* features=candidates+j*c;std::array<float,h> cell{};
        for(int k=0;k<h;++k)cell[k]=std::tanh(base[k]+dot(sw+k*(h+c)+h,features,c)+sb[k]);
        float score=dot(ow,cell.data(),h)+ob;
        for(int k=0;k<8;++k)score+=survival[k]*features[k];
        logits[j]=score;
    }
    value=dot(cw,result.data(),h)+cb;return result;
}
void Policy::decide(const std::uint64_t* ids,const float* features,int count,const std::int32_t* offsets,
                    const float* candidates,const std::uint8_t* masks,int candidate_count,
                    std::int32_t* choices,float* logits,float* values) {
    if(count<0||count>256||candidate_count<0||candidate_count>count*max_candidates)throw std::invalid_argument("invalid social batch size");
    identities(ids,count);
    if(!count)return;
    if(!features||!offsets||!candidates||!masks||!choices||!logits||!values)throw std::invalid_argument("null social batch buffer");
    if(offsets[0]!=0||offsets[count]!=candidate_count)throw std::invalid_argument("invalid candidate offsets");
    std::size_t added=0;
    for(int j=0;j<count;++j) {
        if(!memory.contains(ids[j]))++added;
        if(offsets[j]<0||offsets[j+1]<=offsets[j]||offsets[j+1]>candidate_count||offsets[j+1]-offsets[j]>max_candidates)throw std::invalid_argument("invalid candidate slice");
        bool enabled=false;
        for(int k=0;k<inputs;++k)if(!std::isfinite(features[j*inputs+k])||std::abs(features[j*inputs+k])>16)throw std::invalid_argument("invalid social context feature");
        for(int k=offsets[j];k<offsets[j+1];++k) {
            if(masks[k]>1)throw std::invalid_argument("invalid social mask");
            enabled|=masks[k]!=0;
            for(int f=0;f<candidate_inputs;++f)if(!std::isfinite(candidates[k*candidate_inputs+f])||std::abs(candidates[k*candidate_inputs+f])>16)throw std::invalid_argument("invalid proposal feature");
        }
        if(!enabled)throw std::invalid_argument("empty social action mask");
    }
    if(memory.size()+added>4096)throw std::invalid_argument("social identity capacity reached");
    auto updated=memory;
    for(int j=0;j<count;++j) {
        auto found=memory.find(ids[j]);std::vector<float> zero;
        if(found==memory.end())zero.assign(hidden_size,0);
        auto& prior=found==memory.end()?zero:found->second;
        updated[ids[j]]=evaluate(features+j*inputs,prior,candidates+offsets[j]*candidate_inputs,offsets[j+1]-offsets[j],logits+offsets[j],values[j]);
        int best=-1;
        for(int k=offsets[j];k<offsets[j+1];++k)if(masks[k]&&(best<0||logits[k]>logits[best]))best=k;
        choices[j]=best-offsets[j];
    }
    memory.swap(updated);
}
void Policy::restore(const std::uint64_t* ids,const float* hidden,int count) {
    identities(ids,count);if(count&&!hidden)throw std::invalid_argument("null social memory");
    std::map<std::uint64_t,std::vector<float>> result;
    for(int j=0;j<count;++j) {
        std::vector<float> h(hidden+j*hidden_size,hidden+(j+1)*hidden_size);
        for(float v:h)if(!std::isfinite(v)||std::abs(v)>1.00001f)throw std::invalid_argument("invalid social hidden value");
        result[ids[j]]=std::move(h);
    }
    memory.swap(result);
}
}

namespace {
thread_local std::string error;
lw::social_inference::Policy& policy(void* handle) {if(!handle)throw std::invalid_argument("null social policy");return *static_cast<lw::social_inference::Policy*>(handle);}
template<class F> int32_t call(F function) {try{error.clear();function();return 0;}catch(const std::exception& e){error=e.what();return -1;}catch(...){error="native social exception";return -1;}}
}
int32_t lw_social_abi(void){return lw::social_inference::Policy::version;}
void* lw_social_load(const char* path){try{error.clear();if(!path)throw std::invalid_argument("null social model path");std::string bytes(path);std::ifstream stream(std::filesystem::path(std::u8string(bytes.begin(),bytes.end())),std::ios::binary);if(!stream)throw std::invalid_argument("cannot open social model");return new lw::social_inference::Policy(lw::social_inference::Policy::load(stream));}catch(const std::exception& e){error=e.what();return nullptr;}catch(...){error="native social exception";return nullptr;}}
void lw_social_destroy(void* handle){delete static_cast<lw::social_inference::Policy*>(handle);}
int32_t lw_social_decide(void* handle,const uint64_t* ids,const float* features,int32_t count,const int32_t* offsets,const float* candidates,const uint8_t* masks,int32_t candidate_count,int32_t* choices,float* logits,float* values){return call([&]{policy(handle).decide(ids,features,count,offsets,candidates,masks,candidate_count,choices,logits,values);});}
int32_t lw_social_get_hidden(void* handle,uint64_t id,float* output){return call([&]{auto& p=policy(handle);if(!id||!output)throw std::invalid_argument("invalid social memory query");auto found=p.memory.find(id);if(found==p.memory.end())std::fill(output,output+64,0);else std::copy(found->second.begin(),found->second.end(),output);});}
int32_t lw_social_set_hidden(void* handle,const uint64_t* ids,const float* hidden,int32_t count){return call([&]{policy(handle).restore(ids,hidden,count);});}
const char* lw_social_error(void){return error.c_str();}
