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
    lw::World senses(42,2,0,3);auto& eye=senses.people[0];auto& body=senses.people[1];
    senses.cover.clear();eye.x=4000;eye.y=4000;eye.facing_x=1000;eye.facing_y=0;
    check(senses.sees(eye,5000,4000),"front sight missing");
    check(!senses.sees(eye,3000,4000)&&!senses.sees(eye,4100,5000),"outside field of view leaked");
    senses.tick=600;check(!senses.sees(eye,5000,4000)&&senses.sees(eye,4400,4000),"illumination not enforced");
    senses.tick=0;senses.cover.push_back({senses.next_id++,5000,4000,200,60});
    check(!senses.sees(eye,5000,4000),"concealment not enforced");
    eye.x=4500;body.x=5000;body.y=4000;body.crouching=false;
    check(senses.sees_person(eye,body),"standing visible person missing");
    body.crouching=true;check(!senses.sees_person(eye,body),"crouching cover not enforced");
    // Hidden gathering quantities produce the same anonymous audible clue.
    lw::World audio(42,2,0,3);audio.cover.clear();
    auto& listener=audio.people[0];auto& worker=audio.people[1];auto& source=audio.nodes[0];
    listener.x=source.x-400;listener.y=source.y;listener.facing_x=-1000;listener.facing_y=0;
    worker.x=source.x;worker.y=source.y;worker.facing_x=1000;worker.facing_y=0;
    listener.person_memory.clear();listener.memory.clear();
    auto alternate=audio;
    audio.step({{worker.id,1,lw::Gather,source.id,0,0,0,1}});
    alternate.step({{worker.id,1,lw::Gather,source.id,0,0,0,2}});
    check(audio.results.front().ok&&alternate.results.front().ok,"hearing test gather failed");
    check(listener.heard.size()==1&&listener.heard[0].kind=="work","unseen sound absent");
    check(audio.observation(listener.id)==alternate.observation(listener.id),"sound leaked hidden quantity/state");
    check(listener.person_memory.empty(),"sound invented a visual person sighting");
    auto heard_save=audio.save();auto heard_copy=lw::World::load(heard_save);
    check(heard_copy.save()==heard_save,"hearing/heading checkpoint mismatch");
    audio.step({});heard_copy.step({});check(audio.save()==heard_copy.save(),"hearing resume diverged");
    for(int i=0;i<41;++i)audio.step({});
    check(listener.heard.empty(),"sound memory never expires");
    // Body motion creates quieter, shorter movement while crouched.
    lw::World stealth(42,2,0,3);stealth.cover.clear();
    stealth.people[0].x=4000;stealth.people[0].y=4000;stealth.people[0].facing_x=-1000;stealth.people[0].facing_y=0;
    stealth.people[1].x=4300;stealth.people[1].y=4000;auto noisy=stealth;
    stealth.step({{stealth.people[1].id,1,lw::Crouch,0,0,0,0,1}});
    stealth.step({{stealth.people[1].id,2,lw::Move,0,5000,4000}});
    noisy.step({{noisy.people[1].id,1,lw::Move,0,4100,4000}});
    check(stealth.people[1].x==4400&&stealth.people[0].heard.empty(),"stealth speed/noise wrong");
    check(noisy.people[0].heard.size()==1,"ordinary footsteps missing");
    // A call can be heard through obstruction without granting visual identity.
    lw::World wall(42,2,0,3);wall.cover.clear();
    wall.people[0].x=11050;wall.people[0].y=7000;wall.people[0].facing_x=1000;wall.people[0].facing_y=0;
    wall.people[1].x=12950;wall.people[1].y=7000;
    check(!wall.sees_person(wall.people[0],wall.people[1]),"wall leaked body");
    wall.step({{wall.people[1].id,1,lw::Call}});
    check(wall.people[0].heard.size()==1&&wall.people[0].heard[0].kind=="call","muffled call missing");
    auto poisoned=wall;poisoned.people[0].facing_x=0;poisoned.people[0].facing_y=0;
    bool bad_perception=false;try{lw::World::load(poisoned.save());}catch(const std::invalid_argument&){bad_perception=true;}
    check(bad_perception,"invalid heading accepted with valid checksum");
    poisoned=wall;poisoned.people[0].heard[0].bearing=8;bad_perception=false;
    try{lw::World::load(poisoned.save());}catch(const std::invalid_argument&){bad_perception=true;}
    check(bad_perception,"invalid hearing accepted with valid checksum");
    lw::World edge(42,1,0,3);edge.tick=600;edge.cover.clear();
    auto& walker=edge.people[0];walker.x=11050;walker.y=7000;walker.facing_x=1000;walker.facing_y=0;
    check(!edge.sees(walker,12000,7000)&&edge.sees_obstacle(walker,edge.obstacles[0]),"near rock edge invisible at night");
    edge.step({});check(walker.landmark_memory.size()==1,"visible landmark not remembered");
    auto landmark=walker.landmark_memory[0];
    edge.step({{walker.id,1,lw::Turn,0,9000,7000}});
    check(walker.landmark_memory[0].tick==landmark.tick,"remote landmark memory refreshed");
    check(lw::World::load(edge.save()).save()==edge.save(),"landmark memory checkpoint mismatch");
    std::cout<<"Native checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
