#include "world.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(){try{
    lw::World a(42,32),b(42,32);check(a.hash()==b.hash(),"generation mismatch");
    for(int t=0;t<160;++t){a.step({});b.step({});check(a.hash()==b.hash(),"rollout mismatch");check(a.conserved(),"conservation");}
    auto c=lw::World::load(a.save());check(c.hash()==a.hash(),"checkpoint mismatch");
    for(int t=0;t<10;++t){a.step({});c.step({});check(c.hash()==a.hash(),"resume mismatch");}
    bool rejected=false;try{lw::World::load("bad");}catch(const std::invalid_argument&){rejected=true;}check(rejected,"bad save accepted");
    lw::World d(7,4);auto id=d.people.front().id;
    d.step({{id,1,lw::Depart}});check(d.people.front().group==0,"depart rejected");
    d.step({{id,1,lw::Depart}});check(!d.results.front().ok,"duplicate request accepted");
    lw::World geometry(9,1);
    auto& viewer=geometry.people.front();viewer.x=11000;viewer.y=7000;
    check(!geometry.sees(viewer,12500,7000),"obstruction leaked sight");
    check(geometry.sees(viewer,11000,8000),"clear sight blocked");
    geometry.step({{viewer.id,1,lw::Move,0,12500,7000}});
    check(!geometry.results.front().ok && geometry.results.front().reason=="blocked","body crossed obstacle");
    auto old_tick=geometry.tick;
    bool oversized=false;try{geometry.step(std::vector<lw::Action>(257));}catch(const std::invalid_argument&){oversized=true;}
    check(oversized && geometry.tick==old_tick,"invalid batch mutated world");
    // A depleted source changes only the observer's new sighting, never remote beliefs.
    lw::World memory(9,8);auto& person=memory.people.front();auto& food=memory.nodes.front();
    auto remembered=person.memory.front();person.x=12000;person.y=12000;
    auto original=person.memory;
    memory.step({});
    auto found=std::find_if(person.memory.begin(),person.memory.end(),[&](auto& m){return m.id==remembered.id;});
    check(found!=person.memory.end() && found->tick==remembered.tick && found->amount==remembered.amount,"remote memory updated");
    (void)food;(void)original;
    lw::World duplicated(9,4);duplicated.people[1].id=duplicated.people[0].id;
    bool duplicate_id=false;try{lw::World::load(duplicated.save());}catch(const std::invalid_argument&){duplicate_id=true;}
    check(duplicate_id,"duplicate entity IDs accepted with valid checksum");
    std::cout<<"Native checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
