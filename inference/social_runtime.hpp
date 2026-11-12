#pragma once
#include <cstdint>
#include <istream>
#include <map>
#include <vector>

namespace lw::social_inference {
#ifndef LW_SOCIAL_VERSION
#define LW_SOCIAL_VERSION 1
#endif
class Policy {
public:
    static constexpr int version=LW_SOCIAL_VERSION,inputs=64,candidate_inputs=version==1?52:64,hidden_size=64,max_candidates=128,weight_count=33930+64*candidate_inputs;
    static_assert(version==1||version==2);
    std::vector<float> weights;
    std::map<std::uint64_t,std::vector<float>> memory;
    explicit Policy(const std::vector<float>& weights);
    static Policy load(std::istream& stream);
    void decide(const std::uint64_t* ids,const float* features,int count,const std::int32_t* offsets,
                const float* candidates,const std::uint8_t* masks,int candidate_count,
                std::int32_t* choices,float* logits,float* values);
    void restore(const std::uint64_t* ids,const float* hidden,int count);
private:
    std::vector<float> evaluate(const float* features,const std::vector<float>& prior,const float* candidates,
                                int count,float* logits,float& value) const;
};
}
