#pragma once
#include <stdint.h>
#ifdef _WIN32
#define LW_INFER_API __declspec(dllexport)
#else
#define LW_INFER_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
LW_INFER_API int32_t lw_policy_abi(void);
LW_INFER_API void* lw_policy_load(const char* path);
LW_INFER_API void lw_policy_destroy(void* policy);
LW_INFER_API int32_t lw_policy_inputs(void* policy);
LW_INFER_API int32_t lw_policy_hidden_size(void* policy);
LW_INFER_API int32_t lw_policy_decide(void* policy,const uint64_t* ids,const float* features,
                                    const uint8_t* masks,int32_t count,int32_t* choices,float* logits,float* values);
LW_INFER_API int32_t lw_policy_get_hidden(void* policy,uint64_t id,float* output);
LW_INFER_API int32_t lw_policy_set_hidden(void* policy,const uint64_t* ids,const float* hidden,int32_t count);
LW_INFER_API const char* lw_policy_error(void);
#ifdef __cplusplus
}
#endif
