#include "runtime.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    using lw::inference::Policy;std::vector<float> w(Policy::weight_count(22,48),0);
    auto actor_bias=w.size()-48-1-8;w[actor_bias+1]=1;
    Policy p(22,48,w);std::array<std::uint64_t,2> ids{41,42};std::array<float,44> features{};
    std::array<std::uint8_t,16> mask;mask.fill(1);std::array<std::int32_t,2> choices{};std::array<float,16> logits{};std::array<float,2> values{};
    p.decide(ids.data(),features.data(),mask.data(),2,choices.data(),logits.data(),values.data());
    check(choices[0]==1&&choices[1]==1,"linear head mismatch");
    mask.fill(0);mask[0]=1;mask[8]=1;p.decide(ids.data(),features.data(),mask.data(),2,choices.data(),logits.data(),values.data());
    check(choices[0]==0&&choices[1]==0,"mask ignored");
    auto before=p.memory;ids[1]=ids[0];bool duplicate=false;
    try{p.decide(ids.data(),features.data(),mask.data(),2,choices.data(),logits.data(),values.data());}catch(const std::invalid_argument&){duplicate=true;}
    check(duplicate&&p.memory==before,"duplicate batch mutated memory");
    bool bad=false;std::istringstream stream("invalid");try{Policy::load(stream);}catch(const std::invalid_argument&){bad=true;}
    check(bad,"bad model accepted");
    std::cout<<"Native inference checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
