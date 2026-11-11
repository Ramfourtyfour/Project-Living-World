#include "world.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace lw {
namespace {
std::int64_t distance2(int ax, int ay, int bx, int by) {
    auto x = std::int64_t(ax) - bx, y = std::int64_t(ay) - by;
    return x*x + y*y;
}
bool close(int ax, int ay, int bx, int by) { return distance2(ax,ay,bx,by) <= reach*reach; }
int total(const std::array<int,5>& a) { return std::accumulate(a.begin(),a.end(),0); }
bool resource_valid(int r) { return r >= 0 && r < 5; }
std::string quoted(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
template<class T> void array_json(std::ostream& o, const T& a) {
    o << '['; bool comma=false; for (auto v:a) { if(comma) o << ','; comma=true; o << v; } o << ']';
}
std::string checksum(const std::string& s) {
    std::uint64_t h=14695981039346656037ULL;
    for(unsigned char c:s) { h ^= c; h *= 1099511628211ULL; }
    std::ostringstream o; o << std::hex << std::setw(16) << std::setfill('0') << h; return o.str();
}
void node_json(std::ostream& o,const Node& n) {
    o << "{\"id\":" << n.id << ",\"resource\":" << n.resource << ",\"x\":" << n.x
      << ",\"y\":" << n.y << ",\"amount\":" << n.amount << '}';
}
void person_json(std::ostream& o,const Individual& a,bool self) {
    o << "{\"id\":" << a.id << ",\"x\":" << a.x << ",\"y\":" << a.y
      << ",\"group\":" << a.group << ",\"alive\":" << (a.health>0?"true":"false");
    if(self) {
        o << ",\"health\":" << a.health << ",\"hunger\":" << a.hunger << ",\"thirst\":" << a.thirst
          << ",\"fatigue\":" << a.fatigue << ",\"style\":" << a.style << ",\"last_request\":" << a.last_request
          << ",\"inventory\":"; array_json(o,a.inventory);
    }
    o << '}';
}
}
std::uint64_t World::random() {
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return rng*2685821657736338717ULL;
}
World::World(std::uint64_t s,int population,int scenario,int rules):seed(s),rng(s?s:0x9e3779b97f4a7c15ULL),rules_version(rules) {
    if(population<1 || population>64) throw std::invalid_argument("population must be 1..64");
    if(scenario<0 || scenario>1) throw std::invalid_argument("scenario must be 0 (normal) or 1 (scarce)");
    if(rules<1 || rules>3) throw std::invalid_argument("rules version must be 1..3");
    const std::array<std::array<int,2>,4> camps = {{{4000,4000},{20000,4000},{4000,20000},{20000,20000}}};
    for(int g=0;g<4;++g) {
        int cx=camps[g][0], cy=camps[g][1];
        for(int r=0;r<5;++r) {
            int dx = r==0?-650:r==1?650:int(random()%1801)-900;
            int dy = r<2?0:int(random()%1801)-900;
            int stock=scenario && r<2?12:40;
            int period=r==0?(scenario?20:5):r==1?(scenario?12:3):r==2?15:r==4?12:0;
            nodes.push_back({next_id++,r,cx+dx,cy+dy,stock,stock,period});
        }
    }
    for(int j=0;j<20;++j) {
        int r=int(random()%5);
        nodes.push_back({next_id++,r,2000+int(random()%20001),2000+int(random()%20001),24,24,r<2?6:0});
    }
    obstacles={{12000,7000,900},{12000,17000,900},{8000,12000,700},{16000,12000,700}};
    for(int i=0;i<population;++i) {
        int g=i%4; Individual a; a.id=next_id++; a.group=g+1;
        a.x=camps[g][0]+int(random()%401)-200; a.y=camps[g][1]+int(random()%401)-200;
        a.hunger=10+int(random()%26); a.thirst=10+int(random()%26); a.fatigue=10+int(random()%26);
        a.style=int(random()%1000); people.push_back(a);
    }
    for(auto& n:nodes) initial[n.resource]+=n.amount;
    for(auto& a:people) for(auto& b:people) if(a.id!=b.id && a.group==b.group) a.familiarity[b.id]=1;
    if(rules_version>=3)initialize_perception(true);
    refresh_memory();
    if(rules_version>=2) initialize_social(true);
    if(rules_version>=3)refresh_perception();
}
Individual* World::individual(Id id) {
    auto it=std::find_if(people.begin(),people.end(),[=](auto& a){return a.id==id;});
    return it==people.end()?nullptr:&*it;
}
const Individual* World::individual(Id id) const {
    auto it=std::find_if(people.begin(),people.end(),[=](auto& a){return a.id==id;});
    return it==people.end()?nullptr:&*it;
}
bool World::sees(const Individual& a,int x,int y,int concealment) const {
    if(a.health<=0 || distance2(a.x,a.y,x,y)>sight*sight) return false;
    if(rules_version>=3) {
        auto dx=std::int64_t(x)-a.x,dy=std::int64_t(y)-a.y,length=dx*dx+dy*dy;
        auto dot=dx*a.facing_x+dy*a.facing_y;
        auto direction_length=std::int64_t(a.facing_x)*a.facing_x+std::int64_t(a.facing_y)*a.facing_y;
        if(dot<0||4*dot*dot<length*direction_length)return false;
        int density=std::clamp(cover_density(x,y)+concealment,0,90);
        int range=std::max(100,(300+15*illumination())*(100-density)/100);
        if(length>std::int64_t(range)*range)return false;
    }
    return !occluded(a.x,a.y,x,y);
}
void World::emit(Id actor,Id target,const std::string& type,const std::string& reason,int quantity,bool priv) {
    Event e{next_id++,actor,target,tick,type,reason,quantity,{}};
    auto source=individual(actor);
    for(auto& a:people) if(a.id==actor || (!priv && source && sees_person(a,*source))) e.witnesses.push_back(a.id);
    if(rules_version>=2 && target && type!="group_founded" && type!="group_renamed") {
        int x=0,y=0;bool spatial=false;
        if(auto p=individual(target)){x=p->x;y=p->y;spatial=true;}
        for(auto& n:nodes) if(n.id==target){x=n.x;y=n.y;spatial=true;break;}
        for(auto& c:caches) if(c.id==target){x=c.x;y=c.y;spatial=true;break;}
        for(auto observer:e.witnesses) if(observer!=actor) {
            auto p=individual(observer);
            auto target_person=individual(target);
            if(!spatial || !p || !(target_person?sees_person(*p,*target_person):sees(*p,x,y)))redacted_targets[e.id].push_back(observer);
        }
    }
    events.push_back(e);
    if(events.size()>4096){redacted_targets.erase(events.front().id);events.pop_front();}
}
void World::refresh_memory() {
    for(auto& a:people) {
        if(a.health<=0) continue;
        for(auto& n:nodes) if(sees(a,n.x,n.y)) {
            auto it=std::find_if(a.memory.begin(),a.memory.end(),[&](auto& m){return m.id==n.id;});
            Memory m{n.id,n.resource,n.x,n.y,n.amount,tick};
            if(it==a.memory.end()) a.memory.push_back(m); else *it=m;
        }
        std::stable_sort(a.memory.begin(),a.memory.end(),[](auto& x,auto& y){return x.tick>y.tick;});
        if(a.memory.size()>32) a.memory.resize(32);
        // Familiarity is directional and only increases after a witnessed completed transfer.
    }
}
std::string World::apply(Individual& a,const Action& q) {
    if(a.health<=0) return "deceased";
    if(q.kind<Wait || q.kind>(rules_version>=3?Crouch:ReportRequest)) return "unknown_action";
    if(q.kind==Turn) {
        if(q.x<0||q.y<0||q.x>extent||q.y>extent)return "outside_world";
        if(q.x==a.x&&q.y==a.y)return "no_direction";
        face(a,q.x,q.y);return "";
    }
    if(q.kind==Call){make_sound(a,"call",4000);return "";}
    if(q.kind==Crouch){if(q.quantity!=0&&q.quantity!=1)return "invalid_posture";a.crouching=q.quantity!=0;return "";}
    if(q.kind>=FoundGroup)return rules_version>=2?apply_social(a,q):"unknown_action";
    switch(q.kind) {
    case Wait: return "";
    case Move: {
        if(q.x<0||q.y<0||q.x>extent||q.y>extent) return "outside_world";
        // The whole segment must clear all obstructions, not just its endpoint.
        auto dx=std::int64_t(q.x)-a.x, dy=std::int64_t(q.y)-a.y;
        auto d=std::sqrt(double(dx*dx+dy*dy));
        int speed=rules_version>=3&&a.crouching?100:200;
        if(d>speed) { dx=std::int64_t(std::round(double(dx)*speed/d)); dy=std::int64_t(std::round(double(dy)*speed/d)); }
        int nx=a.x+int(dx), ny=a.y+int(dy);
        for(auto& b:obstacles) {
            double vx=double(nx-a.x),vy=double(ny-a.y),l=vx*vx+vy*vy;
            double t=l?std::clamp(((b.x-a.x)*vx+(b.y-a.y)*vy)/l,0.0,1.0):0;
            double ex=a.x+t*vx-b.x, ey=a.y+t*vy-b.y;
            if(ex*ex+ey*ey <= double(b.radius+35)*(b.radius+35)) return "blocked";
        }
        for(auto& c:caches) {
            double vx=double(nx-a.x),vy=double(ny-a.y),l=vx*vx+vy*vy;
            double t=l?std::clamp(((c.x-a.x)*vx+(c.y-a.y)*vy)/l,0.0,1.0):0;
            double ex=a.x+t*vx-c.x,ey=a.y+t*vy-c.y;
            if(ex*ex+ey*ey<=125.0*125.0)return "blocked";
        }
        if(rules_version>=3&&d)face(a,q.x,q.y);
        a.x=nx;a.y=ny;
        if(d)make_sound(a,"footsteps",a.crouching?150:450);
        return "";
    }
    case Inspect: {
        auto n=std::find_if(nodes.begin(),nodes.end(),[&](auto& n){return n.id==q.target;});
        if(n!=nodes.end() && sees(a,n->x,n->y)) {emit(a.id,n->id,"inspect");return "";}
        auto p=individual(q.target);
        if(p && p->id!=a.id && sees_person(a,*p)) {emit(a.id,p->id,"inspect");return "";}
        for(auto& c:caches)if(c.id==q.target&&sees(a,c.x,c.y)){emit(a.id,c.id,"inspect");return "";}
        return "not_visible";
    }
    case Gather: {
        if(q.quantity<1||q.quantity>2) return "invalid_quantity";
        auto n=std::find_if(nodes.begin(),nodes.end(),[&](auto& n){return n.id==q.target;});
        if(n==nodes.end() || !sees(a,n->x,n->y)) return "not_visible";
        if(!close(a.x,a.y,n->x,n->y)) return "out_of_reach";
        if(n->amount<q.quantity) return "depleted";
        if(total(a.inventory)+q.quantity>capacity) return "inventory_full";
        n->amount-=q.quantity; a.inventory[n->resource]+=q.quantity;
        emit(a.id,n->id,"gather","",q.quantity);make_sound(a,"work",900);return "";
    }
    case Consume: {
        if(q.resource<0||q.resource>1||q.quantity!=1) return "invalid_consumable";
        if(a.inventory[q.resource]<1) return "insufficient_items";
        --a.inventory[q.resource]; ++consumed[q.resource];
        if(q.resource==0) a.hunger=std::max(0,a.hunger-45); else a.thirst=std::max(0,a.thirst-55);
        emit(a.id,0,"consume","",1); return "";
    }
    case OfferGift: case Trade: case Invite: {
        auto b=individual(q.target);
        if(!b || b==&a || b->health<=0 || !sees_person(a,*b)) return "not_visible";
        if(!close(a.x,a.y,b->x,b->y)) return "out_of_reach";
        if(std::count_if(offers.begin(),offers.end(),[&](auto& o){return o.from==a.id;})>=8) return "too_many_offers";
        if(q.kind==Invite) {
            if(a.group==0) return "no_group";
            offers.push_back({next_id++,a.id,b->id,-1,0,-1,0,a.group,tick+40});
            if(rules_version>=2 && a.known_groups.contains(a.group)) {
                auto known=b->known_groups.find(a.group);auto message=a.known_groups.at(a.group);
                if(known==b->known_groups.end()||known->second.revision<message.revision){message.learned_tick=tick;b->known_groups[a.group]=message;}
            }
        } else {
            if(!resource_valid(q.resource)||q.quantity<1||q.quantity>capacity) return "invalid_quantity";
            if(a.inventory[q.resource]<q.quantity) return "insufficient_items";
            if(q.kind==Trade && (!resource_valid(q.ask_resource)||q.ask_quantity<1||q.ask_quantity>capacity)) return "invalid_trade";
            offers.push_back({next_id++,a.id,b->id,q.resource,q.quantity,q.ask_resource,q.kind==Trade?q.ask_quantity:0,0,tick+40});
        }
        emit(a.id,b->id,q.kind==Invite?"invitation":"offer");make_sound(a,"voice",700);return "";
    }
    case Accept: case Decline: {
        auto it=std::find_if(offers.begin(),offers.end(),[&](auto& o){return o.id==q.target && o.to==a.id;});
        if(it==offers.end()) return "unknown_offer";
        if(q.kind==Decline) {emit(a.id,it->from,"decline");make_sound(a,"voice",700);offers.erase(it);return "";}
        auto b=individual(it->from);
        if(it->expires<tick || !b || b->health<=0) return "expired_offer";
        if(!close(a.x,a.y,b->x,b->y) || !sees_person(a,*b)) return "out_of_reach";
        if(it->group) {
            if(b->group!=it->group) return "stale_invitation";
            a.group=it->group;emit(a.id,b->id,"join");make_sound(a,"voice",700);offers.erase(it);return "";
        }
        // Validate both final balances before changing either inventory, including same-resource trades.
        auto ai=a.inventory, bi=b->inventory;
        if(bi[it->resource]<it->quantity || (it->ask_quantity && ai[it->ask_resource]<it->ask_quantity)) return "insufficient_items";
        bi[it->resource]-=it->quantity; ai[it->resource]+=it->quantity;
        if(it->ask_quantity) {ai[it->ask_resource]-=it->ask_quantity;bi[it->ask_resource]+=it->ask_quantity;}
        if(total(ai)>capacity || total(bi)>capacity) return "inventory_full";
        if(!a.familiarity.contains(b->id) && a.familiarity.size()>=64) return "relationship_capacity";
        a.inventory=ai;b->inventory=bi;
        a.familiarity[b->id]=std::min(100,a.familiarity[b->id]+1);
        if(rules_version>=2) {
            interaction(a.id,b->id,it->ask_quantity?2:3,true,false);
            if(it->ask_quantity)interaction(b->id,a.id,2,true,false);
            transferred(b->id,a.id,it->resource,it->quantity);
            if(it->ask_quantity)transferred(a.id,b->id,it->ask_resource,it->ask_quantity);
        }
        emit(a.id,b->id,it->ask_quantity?"trade":"gift","",it->quantity);offers.erase(it);return "";
    }
    case Rest: a.fatigue=std::max(0,a.fatigue-9); return "";
    case Depart: if(!a.group) return "no_group"; a.group=0;emit(a.id,0,"depart");return "";
    }
    return "unknown_action";
}
void World::step(const std::vector<Action>& input) {
    if(input.size()>256) throw std::invalid_argument("batch limit is 256");
    if(tick>=1000000000ULL) throw std::runtime_error("tick limit reached");
    ++tick;results.clear();pending_sounds.clear();
    auto actions=input;
    std::stable_sort(actions.begin(),actions.end(),[](auto& a,auto& b){return a.actor<b.actor;});
    std::map<Id,int> counts;for(auto& q:actions) ++counts[q.actor];
    for(auto& q:actions) {
        auto a=individual(q.actor);std::string error;
        if(!a) error="unknown_actor";
        else if(counts[q.actor]>1) error="duplicate_actor_in_batch";
        else if(q.request==0 || q.request<=a->last_request) error="duplicate_request";
        else {a->last_request=q.request;error=apply(*a,q);}
        results.push_back({q.actor,q.request,error.empty(),error});
        if(!error.empty() && a) emit(a->id,q.target,"rejected",error,0,true);
    }
    for(auto& a:people) if(a.health>0) {
        a.hunger=std::min(100,a.hunger+1);a.thirst=std::min(100,a.thirst+1);a.fatigue=std::min(100,a.fatigue+1);
        int damage=(a.hunger==100?2:0)+(a.thirst==100?2:0)+(a.fatigue==100?1:0);
        a.health=std::max(0,a.health-damage);
        if(a.health==0) emit(a.id,0,"death");
    }
    for(auto& n:nodes) if(n.period && tick%std::uint64_t(n.period)==0 && n.amount<n.maximum) {
        int gain=std::min(2,n.maximum-n.amount);n.amount+=gain;regenerated[n.resource]+=gain;
    }
    std::erase_if(offers,[&](auto& o){return o.expires<tick;});
    refresh_memory();
    if(rules_version>=2)refresh_social();
    if(rules_version>=3)refresh_perception();
    if(!conserved()) throw std::logic_error("resource conservation violated");
}
bool World::conserved() const {
    std::array<std::int64_t,5> current{};
    for(auto& n:nodes) current[n.resource]+=n.amount;
    for(auto& a:people) for(int r=0;r<5;++r) current[r]+=a.inventory[r];
    for(auto& c:caches)for(int r=0;r<5;++r)current[r]+=c.inventory[r]+c.materials[r];
    for(int r=0;r<5;++r) if(initial[r]+regenerated[r]-consumed[r]!=current[r]) return false;
    return true;
}
std::string World::observation(Id id) const {
    auto a=individual(id);if(!a) throw std::invalid_argument("unknown observer");
    std::ostringstream o;o << "{\"version\":" << rules_version << ",\"tick\":" << tick << ",\"self\":";person_json(o,*a,true);
    o << ",\"nodes\":[";bool comma=false;
    for(auto& n:nodes) if(sees(*a,n.x,n.y)) {if(comma)o<<',';comma=true;node_json(o,n);}
    o<<']';
    o << ",\"people\":[";comma=false;
    for(auto& b:people) if(a->id!=b.id && sees_person(*a,b)) {if(comma)o<<',';comma=true;person_json(o,b,false);}
    o<<']';
    o << ",\"memory\":[";comma=false;
    for(auto& m:a->memory) {if(comma)o<<',';comma=true;
        o<<"{\"id\":"<<m.id<<",\"resource\":"<<m.resource<<",\"x\":"<<m.x<<",\"y\":"<<m.y
         <<",\"amount_at_observation\":"<<m.amount<<",\"seen_tick\":"<<m.tick<<",\"age\":"<<tick-m.tick
         <<",\"source\":\"sight\",\"confidence\":"<<std::max(10,100-int(std::min<std::uint64_t>(90,tick-m.tick)))<<'}';}o<<']';
    o << ",\"offers\":[";comma=false;
    for(auto& f:offers) if(f.to==id||f.from==id) {if(comma)o<<',';comma=true;
        o<<"{\"id\":"<<f.id<<",\"from\":"<<f.from<<",\"to\":"<<f.to<<",\"resource\":"<<f.resource
         <<",\"quantity\":"<<f.quantity<<",\"ask_resource\":"<<f.ask_resource<<",\"ask_quantity\":"<<f.ask_quantity
         <<",\"group\":"<<f.group<<",\"expires\":"<<f.expires<<'}';}o<<']';
    o << ",\"relationships\":[";comma=false;
    for(auto& [other,f]:a->familiarity) {if(comma)o<<',';comma=true;o<<"{\"id\":"<<other<<",\"familiarity\":"<<f;
        if(rules_version>=2){auto rel=a->relationships.find(other);Relationship state=rel==a->relationships.end()?Relationship{}:rel->second;
            o<<",\"trust\":"<<state.trust<<",\"helped\":"<<state.helped<<",\"broken\":"<<state.broken<<",\"last_tick\":"<<state.last_tick;}
        o<<'}';}o<<']';
    o << ",\"obstacles\":[";comma=false;
    for(auto& b:obstacles) if(sees_obstacle(*a,b)) {
        if(comma)o<<',';
        comma=true;o<<"{\"x\":"<<b.x<<",\"y\":"<<b.y<<",\"radius\":"<<b.radius;
        if(rules_version>=3)o<<",\"id\":"<<b.id;
        o<<'}';
    }
    o<<']';
    if(rules_version>=2)o<<social_observation(*a);
    if(rules_version>=3)o<<perception_observation(*a);
    o<<",\"events\":"<<events_json(id)<<'}';return o.str();
}
std::string World::events_json(Id observer) const {
    std::ostringstream o;o<<'[';bool comma=false;int count=0;
    // A bounded personal diary contains only events witnessed when they occurred.
    for(auto it=events.rbegin();it!=events.rend() && count<32;++it) {
        auto& e=*it;if(observer && std::find(e.witnesses.begin(),e.witnesses.end(),observer)==e.witnesses.end())continue;
        if(comma)o<<',';
        comma=true;++count;
        auto target=e.target;auto hidden=redacted_targets.find(e.id);
        if(observer && hidden!=redacted_targets.end() && std::find(hidden->second.begin(),hidden->second.end(),observer)!=hidden->second.end())target=0;
        o<<"{\"id\":"<<e.id<<",\"tick\":"<<e.tick<<",\"actor\":"<<e.actor<<",\"target\":"<<target
         <<",\"type\":"<<quoted(e.type)<<",\"reason\":"<<quoted(e.reason)<<",\"quantity\":"<<e.quantity<<'}';
    }o<<']';return o.str();
}
std::string World::results_json() const {
    std::ostringstream o;o<<'[';bool comma=false;
    for(auto& r:results) {if(comma)o<<',';comma=true;o<<"{\"actor\":"<<r.actor<<",\"request\":"<<r.request
        <<",\"ok\":"<<(r.ok?"true":"false")<<",\"reason\":"<<quoted(r.reason)<<'}';}o<<']';return o.str();
}
std::string World::debug_json() const {
    std::ostringstream o;o<<"{\"oracle\":true,\"tick\":"<<tick<<",\"seed\":"<<seed<<",\"hash\":"<<quoted(hash())<<",\"people\":[";
    bool comma=false;for(auto& a:people){if(comma)o<<',';comma=true;person_json(o,a,true);}o<<"],\"nodes\":[";
    comma=false;for(auto& n:nodes){if(comma)o<<',';comma=true;node_json(o,n);}o<<"],\"ledger\":{\"initial\":";
    array_json(o,initial);o<<",\"regenerated\":";array_json(o,regenerated);o<<",\"consumed\":";array_json(o,consumed);
    o<<"},\"conserved\":"<<(conserved()?"true":"false");
    if(rules_version>=2)o<<social_debug();
    if(rules_version>=3)o<<perception_debug();
    o<<'}';return o.str();
}
std::string World::save() const {
    std::ostringstream o;o<<"LIVING_WORLD_V"<<rules_version<<'\n'<<seed<<' '<<rng<<' '<<tick<<' '<<next_id<<'\n';
    for(auto& a:{initial,regenerated,consumed}){for(auto v:a)o<<v<<' ';o<<'\n';}
    o<<people.size()<<'\n';
    for(auto& a:people) {
        o<<a.id<<' '<<a.x<<' '<<a.y<<' '<<a.group<<' '<<a.health<<' '<<a.hunger<<' '<<a.thirst<<' '<<a.fatigue<<' '<<a.last_request<<' '<<a.style<<'\n';
        for(auto v:a.inventory)o<<v<<' ';
        o<<'\n'<<a.memory.size()<<'\n';
        for(auto& m:a.memory)o<<m.id<<' '<<m.resource<<' '<<m.x<<' '<<m.y<<' '<<m.amount<<' '<<m.tick<<'\n';
        o<<a.familiarity.size()<<'\n';for(auto [id,f]:a.familiarity)o<<id<<' '<<f<<'\n';
    }
    o<<nodes.size()<<'\n';for(auto& n:nodes)o<<n.id<<' '<<n.resource<<' '<<n.x<<' '<<n.y<<' '<<n.amount<<' '<<n.maximum<<' '<<n.period<<'\n';
    o<<obstacles.size()<<'\n';for(auto& b:obstacles)o<<b.x<<' '<<b.y<<' '<<b.radius<<'\n';
    o<<offers.size()<<'\n';for(auto& f:offers)o<<f.id<<' '<<f.from<<' '<<f.to<<' '<<f.resource<<' '<<f.quantity<<' '<<f.ask_resource<<' '<<f.ask_quantity<<' '<<f.group<<' '<<f.expires<<'\n';
    o<<events.size()<<'\n';for(auto& e:events){o<<e.id<<' '<<e.actor<<' '<<e.target<<' '<<e.tick<<' '<<std::quoted(e.type)<<' '<<std::quoted(e.reason)<<' '<<e.quantity<<' '<<e.witnesses.size();for(auto id:e.witnesses)o<<' '<<id;o<<'\n';}
    o<<results.size()<<'\n';for(auto& r:results)o<<r.actor<<' '<<r.request<<' '<<r.ok<<' '<<std::quoted(r.reason)<<'\n';
    if(rules_version>=2)save_social(o);
    if(rules_version>=3)save_perception(o);
    auto body=o.str();return body+"CHECKSUM "+checksum(body)+"\n";
}
World World::load(const std::string& text) {
    if(text.size()>16*1024*1024) throw std::invalid_argument("snapshot too large");
    auto pos=text.rfind("CHECKSUM ");
    if(pos==std::string::npos||text.substr(pos)!="CHECKSUM "+checksum(text.substr(0,pos))+"\n") throw std::invalid_argument("snapshot checksum mismatch");
    std::istringstream in(text.substr(0,pos));std::string version;in>>version;
    if(version!="LIVING_WORLD_V1"&&version!="LIVING_WORLD_V2"&&version!="LIVING_WORLD_V3") throw std::invalid_argument("unsupported snapshot version");
    World w(1,1);w.people.clear();w.nodes.clear();w.obstacles.clear();w.offers.clear();w.events.clear();w.results.clear();
    w.rules_version=version=="LIVING_WORLD_V3"?3:version=="LIVING_WORLD_V2"?2:1;
    auto require=[&](bool b){if(!b||!in)throw std::invalid_argument("invalid snapshot");};
    auto count=[&](std::size_t limit){long long n=-1;in>>n;require(n>=0&&std::uint64_t(n)<=limit);return int(n);};
    auto location=[](int x,int y){return x>=0&&y>=0&&x<=extent&&y<=extent;};
    auto meter=[](int x){return x>=0&&x<=100;};
    in>>w.seed>>w.rng>>w.tick>>w.next_id;require(w.rng!=0&&w.tick<=1000000000ULL&&w.next_id>0);
    for(auto a:{&w.initial,&w.regenerated,&w.consumed})for(auto& v:*a){in>>v;require(v>=0&&v<=1000000000000LL);}
    std::set<Id> identities;
    auto identity=[&](Id id){require(id>0&&id<w.next_id&&identities.insert(id).second);};
    int n=count(64);require(n>0);
    for(int j=0;j<n;++j){Individual a;in>>a.id>>a.x>>a.y>>a.group>>a.health>>a.hunger>>a.thirst>>a.fatigue>>a.last_request>>a.style;
        identity(a.id);require(location(a.x,a.y)&&a.group>=0&&a.group<1000000000&&(w.rules_version>=2||a.group<=4)&&meter(a.health)&&meter(a.hunger)&&meter(a.thirst)&&meter(a.fatigue)&&a.style>=0&&a.style<1000);
        for(auto& v:a.inventory){in>>v;require(v>=0&&v<=capacity);}require(total(a.inventory)<=capacity);
        int m=count(32);std::set<Id> remembered;for(int k=0;k<m;++k){Memory mem;in>>mem.id>>mem.resource>>mem.x>>mem.y>>mem.amount>>mem.tick;require(location(mem.x,mem.y)&&resource_valid(mem.resource)&&mem.amount>=0&&mem.amount<=100000&&mem.tick<=w.tick&&remembered.insert(mem.id).second);a.memory.push_back(mem);}
        m=count(64);for(int k=0;k<m;++k){Id id;int f;in>>id>>f;require(f>=0&&f<=100&&id!=a.id&&a.familiarity.emplace(id,f).second);}
        w.people.push_back(a);
    }
    n=count(10000);for(int j=0;j<n;++j){Node v;in>>v.id>>v.resource>>v.x>>v.y>>v.amount>>v.maximum>>v.period;identity(v.id);require(resource_valid(v.resource)&&location(v.x,v.y)&&v.amount>=0&&v.maximum>=v.amount&&v.maximum<=100000&&v.period>=0&&v.period<=100000);w.nodes.push_back(v);}
    n=count(10000);for(int j=0;j<n;++j){Obstacle b;in>>b.x>>b.y>>b.radius;require(location(b.x,b.y)&&b.radius>0&&b.radius<=extent);w.obstacles.push_back(b);}
    n=count(512);for(int j=0;j<n;++j){Offer f;in>>f.id>>f.from>>f.to>>f.resource>>f.quantity>>f.ask_resource>>f.ask_quantity>>f.group>>f.expires;identity(f.id);
        require(w.individual(f.from)&&w.individual(f.to)&&f.from!=f.to&&f.expires>=w.tick&&f.expires<=w.tick+40&&f.group>=0&&f.group<1000000000&&(w.rules_version>=2||f.group<=4));
        require(f.group?(f.resource==-1&&f.quantity==0&&f.ask_quantity==0):(resource_valid(f.resource)&&f.quantity>=1&&f.quantity<=capacity&&f.ask_quantity>=0&&f.ask_quantity<=capacity&&(!f.ask_quantity||resource_valid(f.ask_resource))));w.offers.push_back(f);}
    n=count(4096);for(int j=0;j<n;++j){Event e;in>>e.id>>e.actor>>e.target>>e.tick>>std::quoted(e.type)>>std::quoted(e.reason)>>e.quantity;identity(e.id);require(w.individual(e.actor)&&e.tick<=w.tick&&e.type.size()<64&&e.reason.size()<64&&e.quantity>=0&&e.quantity<=capacity);int m=count(64);std::set<Id> seen;for(int k=0;k<m;++k){Id id;in>>id;require(w.individual(id)&&seen.insert(id).second);e.witnesses.push_back(id);}w.events.push_back(e);}
    n=count(256);for(int j=0;j<n;++j){Result r;in>>r.actor>>r.request>>r.ok>>std::quoted(r.reason);require(r.reason.size()<64);w.results.push_back(r);}
    if(w.rules_version>=2)w.load_social(in);
    if(w.rules_version>=3)w.load_perception(in);
    in>>std::ws;require(in.eof());
    for(auto& a:w.people){for(auto& m:a.memory)require(std::any_of(w.nodes.begin(),w.nodes.end(),[&](auto& v){return v.id==m.id&&v.resource==m.resource&&v.x==m.x&&v.y==m.y;}));for(auto [id,f]:a.familiarity){(void)f;require(w.individual(id));}}
    require(w.conserved());return w;
}
std::string World::hash() const {auto s=save();return checksum(s.substr(0,s.rfind("CHECKSUM ")));}
}
