#pragma once
#include <cstdint>
#include <istream>
#include <map>
#include <vector>

namespace lw::inference {
class Policy {
public:
    int inputs=0,hidden_size=0,actions=8;
    std::vector<float> weights;
    std::map<std::uint64_t,std::vector<float>> memory;
    Policy(int inputs,int hidden,const std::vector<float>& weights);
    static Policy load(std::istream& stream);
    static std::size_t weight_count(int inputs,int hidden);
    void decide(const std::uint64_t* ids,const float* features,const std::uint8_t* masks,
                int count,std::int32_t* choices,float* logits,float* values);
    void restore(const std::uint64_t* ids,const float* hidden,int count);
private:
    std::vector<float> evaluate(const float* features,const std::vector<float>& prior,
                                float* logits,float& value) const;
};
}
