#include "world.hpp"
#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace lw {
namespace {
std::int64_t distance2(int ax,int ay,int bx,int by) {
    auto dx=std::int64_t(ax)-bx,dy=std::int64_t(ay)-by;return dx*dx+dy*dy;
}
bool location(int x,int y) {return x>=0&&x<=extent&&y>=0&&y<=extent;}
bool sound_kind(const std::string& kind) {return kind=="footsteps"||kind=="work"||kind=="voice"||kind=="call";}
}

int World::illumination() const {
    if(rules_version<3)return 100;
    auto phase=tick%1200;
    if(phase<=400)return 100;
    if(phase<500)return 100-int((phase-400)*85/100);
    if(phase<=1000)return 15;
    return 15+int((phase-1000)*85/200);
}

int World::cover_density(int x,int y) const {
    int density=0;
    for(const auto& patch:cover)if(distance2(x,y,patch.x,patch.y)<=std::int64_t(patch.radius)*patch.radius)density=std::max(density,patch.density);
    return density;
}

bool World::occluded(int ax,int ay,int x,int y) const {
    for(const auto& b:obstacles) {
        auto dx=std::int64_t(x)-ax,dy=std::int64_t(y)-ay,len=dx*dx+dy*dy;
        auto dot=(std::int64_t(b.x)-ax)*dx+(std::int64_t(b.y)-ay)*dy;
        if(len&&dot>0&&dot<len) {
            auto cross=dx*(std::int64_t(b.y)-ay)-dy*(std::int64_t(b.x)-ax);
            if(cross*cross<=std::int64_t(b.radius)*b.radius*len)return true;
        }
    }
    return false;
}

bool World::sees_person(const Individual& observer,const Individual& target) const {
    int concealment=rules_version>=3&&target.crouching&&cover_density(target.x,target.y)?25:0;
    return sees(observer,target.x,target.y,concealment);
}

bool World::sees_obstacle(const Individual& observer,const Obstacle& obstacle) const {
    if(rules_version<3)return sees(observer,obstacle.x,obstacle.y);
    double dx=double(observer.x)-obstacle.x,dy=double(observer.y)-obstacle.y,length=std::hypot(dx,dy);
    if(length<=obstacle.radius)return sees(observer,observer.x,observer.y);
    return sees(observer,obstacle.x+int(std::round(dx*(obstacle.radius+1)/length)),obstacle.y+int(std::round(dy*(obstacle.radius+1)/length)));
}

void World::face(Individual& actor,int x,int y) {
    double dx=double(x)-actor.x,dy=double(y)-actor.y,length=std::hypot(dx,dy);
    if(length) {actor.facing_x=int(std::round(dx*1000/length));actor.facing_y=int(std::round(dy*1000/length));}
}

void World::initialize_perception(bool generate_cover) {
    constexpr int directions[8][2]={{1000,0},{707,707},{0,1000},{-707,707},{-1000,0},{-707,-707},{0,-1000},{707,-707}};
    for(auto& actor:people) {auto index=actor.style%8;actor.facing_x=directions[index][0];actor.facing_y=directions[index][1];}
    for(auto& obstacle:obstacles)if(!obstacle.id)obstacle.id=next_id++;
    if(generate_cover) {
        constexpr int centers[8][2]={{6000,4500},{18000,4500},{6000,19500},{18000,19500},{7000,9000},{17000,15000},{14000,10000},{10000,14000}};
        for(const auto& center:centers)cover.push_back({next_id++,center[0]+int(random()%501)-250,center[1]+int(random()%501)-250,350+int(random()%301),35+int(random()%31)});
    }
}

void World::upgrade_perception() {
    if(rules_version>=3)return;
    World upgraded=*this;
    upgraded.upgrade_rules();upgraded.rules_version=3;
    upgraded.initialize_perception(false); // No invented terrain, items or old sightings.
    upgraded.refresh_perception();
    *this=std::move(upgraded);
}

void World::make_sound(const Individual& actor,const std::string& kind,int radius) {
    if(rules_version>=3)pending_sounds.push_back({actor.id,actor.x,actor.y,radius,kind});
}

void World::refresh_perception() {
    for(auto& observer:people)if(observer.health>0) {
        for(const auto& obstacle:obstacles)if(sees_obstacle(observer,obstacle)) {
            auto memory=std::find_if(observer.landmark_memory.begin(),observer.landmark_memory.end(),[&](const auto& m){return m.id==obstacle.id;});
            LandmarkMemory sighting{obstacle.id,obstacle.x,obstacle.y,obstacle.radius,tick};
            if(memory==observer.landmark_memory.end())observer.landmark_memory.push_back(sighting);else *memory=sighting;
        }
        std::stable_sort(observer.landmark_memory.begin(),observer.landmark_memory.end(),[](const auto& a,const auto& b){return a.tick>b.tick;});
        if(observer.landmark_memory.size()>32)observer.landmark_memory.resize(32);
        for(const auto& target:people)if(target.id!=observer.id&&sees_person(observer,target)) {
            auto memory=std::find_if(observer.person_memory.begin(),observer.person_memory.end(),[&](const auto& m){return m.id==target.id;});
            PersonMemory sighting{target.id,target.x,target.y,target.group,target.health>0,tick};
            if(memory==observer.person_memory.end())observer.person_memory.push_back(sighting);else *memory=sighting;
        }
        std::stable_sort(observer.person_memory.begin(),observer.person_memory.end(),[](const auto& a,const auto& b){return a.tick>b.tick;});
        if(observer.person_memory.size()>32)observer.person_memory.resize(32);
        std::erase_if(observer.heard,[&](const auto& cue){return tick-cue.tick>40;});
        for(const auto& sound:pending_sounds) {
            if(sound.actor==observer.id)continue;
            bool muffled=occluded(observer.x,observer.y,sound.x,sound.y);
            int radius=muffled?sound.radius/2:sound.radius;
            auto distance=distance2(observer.x,observer.y,sound.x,sound.y);
            if(distance>std::int64_t(radius)*radius)continue;
            auto angle=std::atan2(double(sound.y-observer.y),double(sound.x-observer.x));
            int bearing=(int(std::round(angle/(std::acos(-1.0)/4)))+8)%8;
            int band=distance<500*500?0:distance<1500*1500?1:2;
            constexpr int confidence_by_band[3]={85,65,45};
            int confidence=confidence_by_band[band]-(muffled?10:0);
            observer.heard.push_back({observer.next_cue++,tick,observer.x,observer.y,bearing,band,confidence,sound.kind});
            if(observer.heard.size()>16)observer.heard.pop_front();
        }
    }
    // Dead people do not acquire new information, but their old memories also expire.
    for(auto& observer:people)if(observer.health<=0)std::erase_if(observer.heard,[&](const auto& cue){return tick-cue.tick>40;});
    pending_sounds.clear();
}

std::string World::perception_observation(const Individual& observer) const {
    std::ostringstream out;
    out<<",\"senses\":{\"ambient_light\":"<<(observer.health>0?std::to_string(illumination()):"null")<<",\"facing\":["<<observer.facing_x<<','<<observer.facing_y
       <<"],\"crouching\":"<<(observer.crouching?"true":"false")<<",\"cover_density\":"<<(observer.health>0?std::to_string(cover_density(observer.x,observer.y)):"null")<<",\"field_of_view_degrees\":120},\"cover\":[";
    bool comma=false;
    for(const auto& patch:cover)if(sees(observer,patch.x,patch.y)) {
        if(comma)out<<',';
        comma=true;
        out<<"{\"id\":"<<patch.id<<",\"x\":"<<patch.x<<",\"y\":"<<patch.y<<",\"radius\":"<<patch.radius<<",\"density\":"<<patch.density<<'}';
    }
    out<<"],\"person_memory\":[";comma=false;
    for(const auto& memory:observer.person_memory) {
        if(comma)out<<',';
        comma=true;
        out<<"{\"id\":"<<memory.id<<",\"x\":"<<memory.x<<",\"y\":"<<memory.y<<",\"group_at_observation\":"<<memory.group
           <<",\"alive_at_observation\":"<<(memory.alive?"true":"false")<<",\"observed_tick\":"<<memory.tick<<",\"age\":"<<tick-memory.tick
           <<",\"source\":\"sight\",\"confidence\":"<<std::max(10,100-int(std::min<std::uint64_t>(90,tick-memory.tick)))<<'}';
    }
    out<<"],\"landmark_memory\":[";comma=false;
    for(const auto& memory:observer.landmark_memory) {
        if(comma)out<<',';
        comma=true;
        out<<"{\"id\":"<<memory.id<<",\"x\":"<<memory.x<<",\"y\":"<<memory.y<<",\"radius\":"<<memory.radius<<",\"observed_tick\":"<<memory.tick
           <<",\"age\":"<<tick-memory.tick<<",\"source\":\"sight\",\"confidence\":"<<std::max(10,100-int(std::min<std::uint64_t>(90,tick-memory.tick)))<<'}';
    }
    out<<"],\"heard\":[";comma=false;
    constexpr int bounds[3][2]={{0,500},{500,1500},{1500,4000}};
    for(const auto& cue:observer.heard) {
        if(comma)out<<',';
        comma=true;
        out<<"{\"local_id\":"<<cue.id<<",\"observed_tick\":"<<cue.tick<<",\"age\":"<<tick-cue.tick<<",\"source\":\"hearing\",\"kind\":\""<<cue.kind
           <<"\",\"listener_origin\":["<<cue.origin_x<<','<<cue.origin_y<<"],\"bearing_octant\":"<<cue.bearing<<",\"bearing_half_width_degrees\":23,\"distance_interval_cm\":["
           <<bounds[cue.distance_band][0]<<','<<bounds[cue.distance_band][1]<<"],\"confidence\":"<<std::max(10,cue.confidence-int(tick-cue.tick))<<'}';
    }
    out<<']';return out.str();
}

std::string World::perception_debug() const {
    std::ostringstream out;out<<",\"ambient_light\":"<<illumination()<<",\"cover\":[";bool comma=false;
    for(const auto& patch:cover) {if(comma)out<<',';comma=true;out<<"{\"id\":"<<patch.id<<",\"x\":"<<patch.x<<",\"y\":"<<patch.y<<",\"radius\":"<<patch.radius<<",\"density\":"<<patch.density<<'}';}
    out<<']';return out.str();
}

void World::save_perception(std::ostream& out) const {
    out<<obstacles.size()<<'\n';for(const auto& obstacle:obstacles)out<<obstacle.id<<'\n';
    out<<cover.size()<<'\n';
    for(const auto& patch:cover)out<<patch.id<<' '<<patch.x<<' '<<patch.y<<' '<<patch.radius<<' '<<patch.density<<'\n';
    out<<people.size()<<'\n';
    for(const auto& person:people) {
        out<<person.id<<' '<<person.facing_x<<' '<<person.facing_y<<' '<<person.crouching<<' '<<person.next_cue<<'\n'<<person.person_memory.size()<<'\n';
        for(const auto& memory:person.person_memory)out<<memory.id<<' '<<memory.x<<' '<<memory.y<<' '<<memory.group<<' '<<memory.alive<<' '<<memory.tick<<'\n';
        out<<person.landmark_memory.size()<<'\n';
        for(const auto& memory:person.landmark_memory)out<<memory.id<<' '<<memory.x<<' '<<memory.y<<' '<<memory.radius<<' '<<memory.tick<<'\n';
        out<<person.heard.size()<<'\n';
        for(const auto& cue:person.heard)out<<cue.id<<' '<<cue.tick<<' '<<cue.origin_x<<' '<<cue.origin_y<<' '<<cue.bearing<<' '<<cue.distance_band<<' '<<cue.confidence<<' '<<cue.kind<<'\n';
    }
}

void World::load_perception(std::istream& in) {
    auto require=[&](bool condition){if(!condition||!in)throw std::invalid_argument("invalid perception snapshot");};
    auto count=[&](int limit){int n=-1;in>>n;require(n>=0&&n<=limit);return n;};
    std::set<Id> ids;
    for(const auto& person:people)ids.insert(person.id);
    for(const auto& node:nodes)ids.insert(node.id);
    for(const auto& cache:caches)ids.insert(cache.id);
    for(const auto& offer:offers)ids.insert(offer.id);
    for(const auto& event:events)ids.insert(event.id);
    for(const auto& request:requests)ids.insert(request.id);
    int n=count(10000);require(std::size_t(n)==obstacles.size());
    for(auto& obstacle:obstacles) {in>>obstacle.id;require(obstacle.id>0&&obstacle.id<next_id&&ids.insert(obstacle.id).second);}
    n=count(256);
    for(int i=0;i<n;++i) {
        Cover patch{};in>>patch.id>>patch.x>>patch.y>>patch.radius>>patch.density;
        require(patch.id>0&&patch.id<next_id&&ids.insert(patch.id).second&&patch.radius>=100&&patch.radius<=1000&&patch.x>=patch.radius&&patch.x<=extent-patch.radius&&patch.y>=patch.radius&&patch.y<=extent-patch.radius&&patch.density>=1&&patch.density<=90);
        cover.push_back(patch);
    }
    n=count(64);require(std::size_t(n)==people.size());std::set<Id> seen;
    for(int i=0;i<n;++i) {
        Id id;in>>id;auto person=individual(id);require(person&&seen.insert(id).second);
        in>>person->facing_x>>person->facing_y>>person->crouching>>person->next_cue;
        require(person->facing_x>=-1000&&person->facing_x<=1000&&person->facing_y>=-1000&&person->facing_y<=1000);
        auto length=std::int64_t(person->facing_x)*person->facing_x+std::int64_t(person->facing_y)*person->facing_y;
        require(length>=998000&&length<=1002000&&person->next_cue>=1&&person->next_cue<=1000000000000ULL);
        int m=count(32);std::set<Id> remembered;
        for(int j=0;j<m;++j) {
            PersonMemory memory{};in>>memory.id>>memory.x>>memory.y>>memory.group>>memory.alive>>memory.tick;
            require(memory.id!=id&&individual(memory.id)&&remembered.insert(memory.id).second&&location(memory.x,memory.y)&&memory.group>=0&&memory.group<next_group&&memory.tick<=tick);
            person->person_memory.push_back(memory);
        }
        m=count(32);remembered.clear();
        for(int j=0;j<m;++j) {
            LandmarkMemory memory{};in>>memory.id>>memory.x>>memory.y>>memory.radius>>memory.tick;
            auto obstacle=std::find_if(obstacles.begin(),obstacles.end(),[&](const auto& o){return o.id==memory.id;});
            require(obstacle!=obstacles.end()&&remembered.insert(memory.id).second&&memory.x==obstacle->x&&memory.y==obstacle->y&&memory.radius==obstacle->radius&&memory.tick<=tick);
            person->landmark_memory.push_back(memory);
        }
        m=count(16);std::uint64_t previous=0,previous_tick=0;
        for(int j=0;j<m;++j) {
            HeardCue cue{};in>>cue.id>>cue.tick>>cue.origin_x>>cue.origin_y>>cue.bearing>>cue.distance_band>>cue.confidence>>cue.kind;
            require(cue.id>previous&&cue.id<person->next_cue&&cue.tick>=previous_tick&&cue.tick<=tick&&tick-cue.tick<=40&&location(cue.origin_x,cue.origin_y)&&cue.bearing>=0&&cue.bearing<8&&cue.distance_band>=0&&cue.distance_band<3&&cue.confidence>=35&&cue.confidence<=90&&sound_kind(cue.kind));
            person->heard.push_back(cue);previous=cue.id;previous_tick=cue.tick;
        }
    }
}
}
