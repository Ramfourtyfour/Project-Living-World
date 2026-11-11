#pragma once
#include <stdint.h>
#ifdef _WIN32
#define LW_API __declspec(dllexport)
#else
#define LW_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct LWAction {
    uint64_t actor, request;
    int32_t kind;
    uint64_t target;
    int32_t x,y,resource,quantity,ask_resource,ask_quantity;
} LWAction;
LW_API void* lw_create(uint64_t seed,int32_t population);
LW_API void* lw_create_scenario(uint64_t seed,int32_t population,int32_t scenario);
LW_API void* lw_load(const char* snapshot);
LW_API void lw_destroy(void* world);
LW_API int32_t lw_step(void* world,const LWAction* actions,int32_t count);
// Returned strings are thread-local, valid until the next string-returning API call.
LW_API const char* lw_observe(void* world,uint64_t actor);
LW_API const char* lw_debug(void* world);
LW_API const char* lw_save(void* world);
LW_API const char* lw_hash(void* world);
LW_API const char* lw_results(void* world);
LW_API const char* lw_error(void);
LW_API int32_t lw_abi_version(void);
LW_API int32_t lw_action_size(void);
#ifdef __cplusplus
}
#endif
