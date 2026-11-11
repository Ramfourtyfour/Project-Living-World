#include "api.h"
#include "world.hpp"
#include <exception>
#include <stdexcept>
#include <string>
namespace {
thread_local std::string output,error;
lw::World& world(void* p){if(!p)throw std::invalid_argument("null world");return *static_cast<lw::World*>(p);}
template<class F> const char* text(F f){try{error.clear();output=f();return output.c_str();}catch(const std::exception& e){error=e.what();return nullptr;}catch(...){error="native exception";return nullptr;}}
}
void* lw_create(uint64_t seed,int32_t population){try{error.clear();return new lw::World(seed,population);}catch(const std::exception& e){error=e.what();return nullptr;}catch(...){error="native exception";return nullptr;}}
void* lw_create_scenario(uint64_t seed,int32_t population,int32_t scenario){try{error.clear();return new lw::World(seed,population,scenario);}catch(const std::exception& e){error=e.what();return nullptr;}catch(...){error="native exception";return nullptr;}}
void* lw_load(const char* snapshot){try{error.clear();if(!snapshot)throw std::invalid_argument("null snapshot");return new lw::World(lw::World::load(snapshot));}catch(const std::exception& e){error=e.what();return nullptr;}catch(...){error="native exception";return nullptr;}}
void lw_destroy(void* p){delete static_cast<lw::World*>(p);}
int32_t lw_step(void* p,const LWAction* actions,int32_t count){try{
    error.clear();if(count<0||count>256||(!actions&&count))throw std::invalid_argument("invalid action batch");
    std::vector<lw::Action> batch;batch.reserve(count);
    for(int i=0;i<count;++i){auto& q=actions[i];batch.push_back({q.actor,q.request,q.kind,q.target,q.x,q.y,q.resource,q.quantity,q.ask_resource,q.ask_quantity});}
    world(p).step(batch);return 0;
}catch(const std::exception& e){error=e.what();return -1;}catch(...){error="native exception";return -1;}}
const char* lw_observe(void* p,uint64_t actor){return text([&]{return world(p).observation(actor);});}
const char* lw_debug(void* p){return text([&]{return world(p).debug_json();});}
const char* lw_save(void* p){return text([&]{return world(p).save();});}
const char* lw_hash(void* p){return text([&]{return world(p).hash();});}
const char* lw_results(void* p){return text([&]{return world(p).results_json();});}
const char* lw_error(void){return error.c_str();}
int32_t lw_abi_version(void){return 1;}
int32_t lw_action_size(void){return sizeof(LWAction);}
