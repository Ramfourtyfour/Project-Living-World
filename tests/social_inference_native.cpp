#include "social_runtime.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using lw::social_inference::Policy;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F function,const char* message){bool failed=false;try{function();}catch(const std::invalid_argument&){failed=true;}check(failed,message);}
int main(){try{
    std::vector<float> weights(Policy::weight_count);
    for(std::size_t i=0;i<weights.size();++i)weights[i]=float(int(i%31)-15)*0.003f;
    Policy p(weights);std::array<std::uint64_t,2> ids={7,9};
    std::array<float,128> x{};x[0]=0.3f;x[64]=0.7f;
    std::array<float,5*Policy::candidate_inputs> candidates{};for(int k=0;k<5;++k)candidates[k*Policy::candidate_inputs+k]=1;
    std::array<std::int32_t,3> offsets={0,3,5};std::array<std::uint8_t,5> mask={0,1,1,1,1};
    std::array<std::int32_t,2> choices{};std::array<float,5> scores{};std::array<float,2> values{};
    p.decide(ids.data(),x.data(),2,offsets.data(),candidates.data(),mask.data(),5,choices.data(),scores.data(),values.data());
    check(choices[0]>=1&&choices[0]<3&&choices[1]>=0&&choices[1]<2,"masked/variable choice invalid");
    check(p.memory.size()==2&&p.memory[7]!=p.memory[9],"individual memories not independent");
    auto original=p.memory;auto empty_mask=mask;empty_mask.fill(0);
    rejects([&]{p.decide(ids.data(),x.data(),2,offsets.data(),candidates.data(),empty_mask.data(),5,choices.data(),scores.data(),values.data());},"empty mask accepted");
    check(p.memory==original,"invalid mask changed memory");
    auto bad_offsets=offsets;bad_offsets[1]=-1;
    rejects([&]{p.decide(ids.data(),x.data(),2,bad_offsets.data(),candidates.data(),mask.data(),5,choices.data(),scores.data(),values.data());},"bad offsets accepted");
    auto poison=candidates;poison.back()=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{p.decide(ids.data(),x.data(),2,offsets.data(),poison.data(),mask.data(),5,choices.data(),scores.data(),values.data());},"NaN proposal accepted");
    check(p.memory==original,"failed batch partially committed memory");
    Policy solo=p,together=p;std::int32_t one_offsets[2]={0,3};
    solo.decide(ids.data(),x.data(),1,one_offsets,candidates.data(),mask.data(),3,choices.data(),scores.data(),values.data());
    together.decide(ids.data(),x.data(),2,offsets.data(),candidates.data(),mask.data(),5,choices.data(),scores.data(),values.data());
    check(solo.memory[7]==together.memory[7],"another actor changed this actor's inference");
    check(solo.memory[9]==original[9],"subset batch updated absent actor");
    std::array<float,64> invalid{};invalid[0]=2;
    rejects([&]{p.restore(ids.data(),invalid.data(),1);},"out-of-range memory accepted");check(p.memory==original,"failed restore changed memory");
    p.restore(nullptr,nullptr,0);check(p.memory.empty(),"empty restore did not clear memory");
    p.decide(nullptr,nullptr,0,nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr);check(p.memory.empty(),"empty batch mutated memory");
    std::ostringstream out(std::ios::binary);const char magic[8]={'L','W','S','L','A','T',char('0'+Policy::version),'\0'};
    std::array<std::uint32_t,6> header={Policy::version,64,Policy::candidate_inputs,64,128,Policy::weight_count};
    out.write(magic,8);out.write(reinterpret_cast<const char*>(header.data()),sizeof(header));out.write(reinterpret_cast<const char*>(weights.data()),weights.size()*sizeof(float));
    auto bytes=out.str();std::istringstream in(bytes,std::ios::binary);auto loaded=Policy::load(in);check(loaded.weights==weights,"social model round trip changed weights");
    auto trailing=bytes+"x";std::istringstream junk(trailing,std::ios::binary);rejects([&]{Policy::load(junk);},"trailing model data accepted");
    auto broken=bytes.substr(0,bytes.size()-1);std::istringstream truncated(broken,std::ios::binary);rejects([&]{Policy::load(truncated);},"truncated model accepted");
    std::cout<<"Native social inference checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
