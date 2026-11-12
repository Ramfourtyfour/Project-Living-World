#pragma once
#include <stdint.h>
#ifdef _WIN32
#define LWS_API __declspec(dllexport)
#else
#define LWS_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
LWS_API int32_t lw_social_abi(void);
LWS_API void* lw_social_load(const char* utf8_path);
LWS_API void lw_social_destroy(void* policy);
LWS_API int32_t lw_social_decide(void* policy,const uint64_t* ids,const float* features,int32_t count,
                                const int32_t* offsets,const float* candidates,const uint8_t* masks,
                                int32_t candidate_count,int32_t* choices,float* logits,float* values);
LWS_API int32_t lw_social_get_hidden(void* policy,uint64_t id,float* output);
LWS_API int32_t lw_social_set_hidden(void* policy,const uint64_t* ids,const float* hidden,int32_t count);
LWS_API const char* lw_social_error(void);
#ifdef __cplusplus
}
#endif
