#include "world.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace lw {
namespace {
constexpr int cache_capacity=120, communication_range=1200;
std::int64_t distance2(int ax,int ay,int bx,int by) {
    auto x=std::int64_t(ax)-bx,y=std::int64_t(ay)-by;return x*x+y*y;
}
bool reachable(int ax,int ay,int bx,int by) {return distance2(ax,ay,bx,by)<=reach*reach;}
int total(const std::array<int,5>& values) {return std::accumulate(values.begin(),values.end(),0);}
bool location(int x,int y) {return x>=0&&x<=extent&&y>=0&&y<=extent;}
bool valid_name(const std::string& text) {
    if(text.empty()||text.size()>40||text.front()==' '||text.back()==' ')return false;
    for(std::size_t i=0;i<text.size();) {
        auto c=static_cast<unsigned char>(text[i++]);
        if(c<128){if(c<32||c==127)return false;continue;}
        int remaining=0;std::uint32_t value=0,minimum=0;
        if(c>=0xc2&&c<=0xdf){remaining=1;value=c&31;minimum=0x80;}
        else if(c>=0xe0&&c<=0xef){remaining=2;value=c&15;minimum=0x800;}
        else if(c>=0xf0&&c<=0xf4){remaining=3;value=c&7;minimum=0x10000;}
        else return false;
        if(i+remaining>text.size())return false;
        while(remaining--){auto next=static_cast<unsigned char>(text[i++]);if((next&0xc0)!=0x80)return false;value=(value<<6)|(next&63);}
        if(value<minimum||value>0x10ffff||(value>=0xd800&&value<=0xdfff))return false;
    }
    return true;
}
template<class T> void json_array(std::ostream& out,const T& values) {
    out<<'[';bool comma=false;for(auto v:values){if(comma)out<<',';comma=true;out<<v;}out<<']';
}
void cache_json(std::ostream& out,const SharedCache& cache,bool contents) {
    out<<"{\"id\":"<<cache.id<<",\"group\":"<<cache.group<<",\"x\":"<<cache.x<<",\"y\":"<<cache.y<<",\"radius\":90,\"capacity\":"<<cache_capacity;
    if(contents){out<<",\"inventory\":";json_array(out,cache.inventory);}
    out<<'}';
}
}

void World::initialize_social(bool starting_caches) {
    if(!groups.empty())throw std::logic_error("social records already initialized");
    for(int id=1;id<=4;++id)groups.push_back({id,0,0,"Circle "+std::to_string(id),1});
    next_group=5;
    for(auto& a:people)if(a.group&&(starting_caches||a.health>0))a.known_groups[a.group]={"Circle "+std::to_string(a.group),1,0};
    if(starting_caches) {
        const int positions[4][2]={{4000,5100},{20000,5100},{4000,21100},{20000,21100}};
        for(int i=0;i<4;++i) {
            SharedCache cache{};cache.id=next_id++;cache.group=i+1;cache.x=positions[i][0];cache.y=positions[i][1];
            cache.materials[2]=3;cache.materials[4]=2;caches.push_back(cache);
            for(int r=0;r<5;++r)initial[r]+=cache.materials[r];
            for(auto& a:people)if(a.group==cache.group)a.places.push_back({cache.id,cache.x,cache.y,cache.group,0,0,true,{}});
        }
    }
    refresh_social();
}

void World::upgrade_rules() {
    if(rules_version>=2)return;
    World upgraded=*this;
    upgraded.rules_version=2;
    upgraded.initialize_social(false); // Migration never spawns infrastructure or materials.
    // Earlier events lack a record of target visibility. Retain own references,
    // conservatively redact everyone else's targets instead of inventing knowledge.
    for(auto& event:upgraded.events)if(event.target)for(auto witness:event.witnesses)
        if(witness!=event.actor)upgraded.redacted_targets[event.id].push_back(witness);
    *this=std::move(upgraded);
}

void World::interaction(Id observer,Id other,int delta,bool help,bool broken) {
    auto a=individual(observer);auto b=individual(other);
    if(!a||!b||a==b||a->health<=0)return;
    a->familiarity[other]=std::min(100,a->familiarity[other]+1);
    auto& relation=a->relationships[other];relation.trust=std::clamp(relation.trust+delta,-100,100);
    if(help)relation.helped=std::min<std::uint64_t>(1000000000000ULL,relation.helped+1);
    if(broken)relation.broken=std::min<std::uint64_t>(1000000000000ULL,relation.broken+1);
    relation.last_tick=tick;
}

void World::tell(Id actor,Id target,const std::string& type,const std::vector<Id>& audience) {
    Event event{next_id++,actor,target,tick,type,"",0,{}};
    for(auto id:audience){auto a=individual(id);if(a&&a->health>0&&std::find(event.witnesses.begin(),event.witnesses.end(),id)==event.witnesses.end())event.witnesses.push_back(id);}
    events.push_back(event);
    if(events.size()>4096){redacted_targets.erase(events.front().id);events.pop_front();}
}

void World::transferred(Id giver,Id receiver,int resource,int quantity) {
    int available=quantity;
    for(auto& request:requests) {
        if(!available)break;
        if(request.status!=1||request.from!=receiver||request.to!=giver||request.intent!=resource||request.intent>1||request.expires<tick)continue;
        int contribution=std::min(available,request.quantity-request.progress);
        request.progress+=contribution;available-=contribution;
        request.from_progress=request.progress;request.from_updated=tick;
        if(request.progress==request.quantity){request.status=3;request.completed_tick=tick;request.reported_tick=tick;confirm_request(request,false);tell(giver,receiver,"request_fulfilled",{giver,receiver});}
    }
}

void World::confirm_request(SocialRequest& request,bool credit) {
    if(request.from_status!=3 && credit) {
        int adjustment=2;
        if(request.from_penalized && request.completed_tick<=request.expires) {
            auto observer=individual(request.from);
            if(observer&&observer->relationships.contains(request.to)) {
                auto& relation=observer->relationships.at(request.to);
                if(relation.broken)--relation.broken;
            }
            adjustment+=2;request.from_penalized=false;
            tell(request.to,request.from,"belief_corrected",{request.from});
        }
        interaction(request.from,request.to,adjustment,true,false);
    }
    request.from_status=3;request.from_progress=request.progress;request.from_updated=tick;
}

std::string World::apply_social(Individual& a,const Action& q) {
    auto own_group=[&](){return std::find_if(groups.begin(),groups.end(),[&](const auto& g){return g.id==a.group;});};
    switch(q.kind) {
    case FoundGroup: {
        if(a.group)return "already_in_group";
        if(groups.size()>=1024||next_group>=1000000000)return "group_record_capacity";
        auto name=q.text.empty()?"Circle "+std::to_string(next_group):q.text;
        if(!valid_name(name))return "invalid_group_name";
        int id=next_group++;groups.push_back({id,a.id,tick,name,1});a.group=id;a.known_groups[id]={name,1,tick};
        emit(a.id,Id(id),"group_founded");
        for(auto witness:events.back().witnesses)if(auto other=individual(witness))other->known_groups[id]={name,1,tick};
        return "";
    }
    case RenameGroup: {
        auto group=own_group();if(group==groups.end())return "no_group";
        if(!valid_name(q.text))return "invalid_group_name";
        if(group->revision>=1000000000000ULL)return "group_revision_capacity";
        group->name=q.text;++group->revision;a.known_groups[group->id]={group->name,group->revision,tick};
        emit(a.id,Id(group->id),"group_renamed");
        for(auto witness:events.back().witnesses)if(auto other=individual(witness))other->known_groups[group->id]={group->name,group->revision,tick};
        return "";
    }
    case RequestHelp: {
        auto b=individual(q.target);
        if(!b||b==&a||b->health<=0||!sees_person(a,*b))return "not_visible";
        if(distance2(a.x,a.y,b->x,b->y)>communication_range*communication_range)return "out_of_hearing_range";
        if(q.resource<0||q.resource>3||q.quantity<1||q.quantity>4||(q.resource>=2&&q.quantity!=1))return "invalid_request";
        if(q.resource>=2&&!location(q.x,q.y))return "outside_world";
        if(requests.size()>=512||std::count_if(requests.begin(),requests.end(),[&](const auto& r){return r.from==a.id&&(r.status==0||r.status==1);})>=8)return "too_many_requests";
        requests.push_back({next_id++,a.id,b->id,q.resource,q.quantity,q.resource<2?a.x:q.x,q.resource<2?a.y:q.y,0,0,tick,tick+60,0});
        requests.back().contact_x=a.x;requests.back().contact_y=a.y;requests.back().from_updated=tick;
        a.familiarity.try_emplace(b->id,0);b->familiarity.try_emplace(a.id,0);
        tell(a.id,b->id,"request_received",{a.id,b->id});make_sound(a,"voice",700);return "";
    }
    case AnswerRequest: {
        auto request=std::find_if(requests.begin(),requests.end(),[&](const auto& r){return r.id==q.target&&r.to==a.id&&r.status==0;});
        if(request==requests.end())return "unknown_request";
        if(request->expires<tick)return "expired_request";
        if(q.quantity!=0&&q.quantity!=1)return "invalid_answer";
        auto sender=individual(request->from);
        if(!sender||sender->health<=0||!sees_person(a,*sender)||distance2(a.x,a.y,sender->x,sender->y)>communication_range*communication_range)return "sender_unreachable";
        request->status=q.quantity?1:2;request->answered=tick;
        request->from_status=request->status;request->from_updated=tick;
        tell(a.id,sender->id,q.quantity?"request_accepted":"request_declined",{a.id,sender->id});make_sound(a,"voice",700);return "";
    }
    case ReportRequest: {
        auto request=std::find_if(requests.begin(),requests.end(),[&](const auto& r){return r.id==q.target&&r.to==a.id&&(r.status==1||r.status==3||r.status==4);});
        if(request==requests.end())return "unknown_request";
        auto recipient=individual(request->from);
        if(!recipient||recipient->health<=0||!sees_person(a,*recipient)||distance2(a.x,a.y,recipient->x,recipient->y)>communication_range*communication_range)return "recipient_unreachable";
        request->reported_tick=tick;request->contact_x=recipient->x;request->contact_y=recipient->y;
        if(request->status==3)confirm_request(*request,true);
        else {request->from_status=request->status;request->from_progress=request->progress;request->from_updated=tick;}
        tell(a.id,recipient->id,"task_report",{a.id,recipient->id});make_sound(a,"voice",700);return "";
    }
    case Deposit: case Withdraw: {
        auto cache=std::find_if(caches.begin(),caches.end(),[&](const auto& c){return c.id==q.target;});
        if(cache==caches.end()||!sees(a,cache->x,cache->y))return "not_visible";
        if(!reachable(a.x,a.y,cache->x,cache->y))return "out_of_reach";
        if(!a.group||a.group!=cache->group)return "access_denied";
        if(q.resource<0||q.resource>=5||q.quantity<1||q.quantity>capacity)return "invalid_quantity";
        if(q.kind==Deposit) {
            if(a.inventory[q.resource]<q.quantity)return "insufficient_items";
            if(total(cache->inventory)+q.quantity>cache_capacity)return "storage_full";
            a.inventory[q.resource]-=q.quantity;cache->inventory[q.resource]+=q.quantity;
            emit(a.id,cache->id,"deposit","",q.quantity);
            for(auto& observer:people)if(observer.id!=a.id&&observer.group==a.group&&sees_person(observer,a)&&sees(observer,cache->x,cache->y))interaction(observer.id,a.id,1,true,false);
        } else {
            if(cache->inventory[q.resource]<q.quantity)return "insufficient_items";
            if(total(a.inventory)+q.quantity>capacity)return "inventory_full";
            cache->inventory[q.resource]-=q.quantity;a.inventory[q.resource]+=q.quantity;
            emit(a.id,cache->id,"withdraw","",q.quantity);
        }
        return "";
    }
    case PlaceCache: {
        if(!a.group)return "no_group";
        if(caches.size()>=512)return "storage_record_capacity";
        int x=q.x,y=q.y;
        if(x==0&&y==0){x=a.x+(a.x+200<=extent?200:-200);y=a.y;}
        if(!location(x,y)||x<90||y<90||x>extent-90||y>extent-90)return "invalid_site";
        if(!reachable(a.x,a.y,x,y)||!sees(a,x,y))return "out_of_reach";
        for(auto& obstacle:obstacles)if(distance2(x,y,obstacle.x,obstacle.y)<=std::int64_t(obstacle.radius+125)*(obstacle.radius+125))return "site_blocked";
        for(auto& cache:caches)if(distance2(x,y,cache.x,cache.y)<=250*250)return "site_blocked";
        for(auto& person:people)if(person.health>0&&distance2(x,y,person.x,person.y)<=125*125)return "site_occupied";
        if(a.inventory[2]<3||a.inventory[4]<2)return "insufficient_materials";
        SharedCache cache{};cache.id=next_id++;cache.creator=a.id;cache.group=a.group;cache.x=x;cache.y=y;cache.created=tick;
        cache.materials[2]=3;cache.materials[4]=2;a.inventory[2]-=3;a.inventory[4]-=2;caches.push_back(cache);
        emit(a.id,cache.id,"cache_built");make_sound(a,"work",1200);return "";
    }
    default:return "unknown_action";
    }
}

void World::refresh_social() {
    for(auto& a:people)if(a.health>0) {
        for(const auto& b:people)if(a.id!=b.id&&sees_person(a,b)) {
            a.familiarity.try_emplace(b.id,0);
            auto badge=b.known_groups.find(b.group);
            if(badge!=b.known_groups.end()) {
                auto known=a.known_groups.find(b.group);
                if(known==a.known_groups.end()||known->second.revision<badge->second.revision)
                    a.known_groups[b.group]={badge->second.name,badge->second.revision,tick};
            }
        }
        for(const auto& cache:caches)if(sees(a,cache.x,cache.y)) {
            auto memory=std::find_if(a.places.begin(),a.places.end(),[&](const auto& m){return m.id==cache.id;});
            if(memory==a.places.end()) {
                a.places.push_back({cache.id,cache.x,cache.y,cache.group,tick,0,false,{}});memory=std::prev(a.places.end());
            }
            memory->seen_tick=tick;
            if(a.group==cache.group&&reachable(a.x,a.y,cache.x,cache.y)) {
                memory->contents_known=true;memory->contents_tick=tick;memory->inventory=cache.inventory;
            }
        }
        std::stable_sort(a.places.begin(),a.places.end(),[](const auto& x,const auto& y){return x.seen_tick>y.seen_tick;});
        if(a.places.size()>32)a.places.resize(32);
    }
    for(auto& request:requests) {
        if(request.status==1&&request.expires>=tick&&request.intent>=2) {
            auto actor=individual(request.to);
            if(actor&&actor->health>0&&reachable(actor->x,actor->y,request.x,request.y)) {
                if(request.intent==2)request.progress=1;else ++request.progress;
                if(request.progress>=(request.intent==2?1:10)) {
                    request.status=3;request.completed_tick=tick;
                    tell(request.to,request.from,"task_completed",{request.to});
                }
            } else if(request.intent==3)request.progress=0;
        }
        // The requester sees its own evidence, not the authoritative task counter.
        // An unseen companion may finish and report later without broadcasting.
        if(request.from_status==1&&request.intent>=2&&request.expires>=tick) {
            auto actor=individual(request.to);auto witness=individual(request.from);
            bool verified=actor&&witness&&actor->health>0&&witness->health>0&&sees_person(*witness,*actor)&&reachable(actor->x,actor->y,request.x,request.y);
            request.from_progress=verified?(request.intent==2?1:std::min(10,request.from_progress+1)):0;
            request.from_updated=tick;
            if(request.from_progress>=(request.intent==2?1:10)) {
                confirm_request(request,true);tell(request.to,request.from,"request_fulfilled",{request.from});
            }
        }
        if((request.status==0||request.status==1)&&request.expires<tick) {
            request.status=4;tell(request.to,request.from,"task_expired",{request.to});
        }
        if((request.from_status==0||request.from_status==1)&&request.expires<tick) {
            bool promised=request.from_status==1;request.from_status=4;request.from_updated=tick;
            auto observer=individual(request.from);
            if(promised&&observer&&observer->health>0){interaction(request.from,request.to,-2,false,true);request.from_penalized=true;}
            tell(request.from,request.to,promised?"promise_unverified":"request_expired",{request.from});
        }
    }
    std::erase_if(requests,[&](const auto& request){return request.expires+80<tick;});
}

std::string World::social_observation(const Individual& a) const {
    std::ostringstream out;bool comma=false;
    out<<",\"agreements\":[";
    for(const auto& [id,known]:a.known_groups){if(comma)out<<',';comma=true;out<<"{\"id\":"<<id<<",\"name\":"<<std::quoted(known.name)<<",\"revision\":"<<known.revision<<",\"learned_tick\":"<<known.learned_tick<<'}';}
    out<<"],\"caches\":[";comma=false;
    for(const auto& cache:caches)if(sees(a,cache.x,cache.y)){if(comma)out<<',';comma=true;cache_json(out,cache,a.group==cache.group&&reachable(a.x,a.y,cache.x,cache.y));}
    out<<"],\"place_memory\":[";comma=false;
    for(const auto& memory:a.places){if(comma)out<<',';comma=true;
        out<<"{\"id\":"<<memory.id<<",\"x\":"<<memory.x<<",\"y\":"<<memory.y<<",\"group\":"<<memory.group
           <<",\"seen_tick\":"<<memory.seen_tick<<",\"age\":"<<tick-memory.seen_tick<<",\"contents_tick\":"<<memory.contents_tick<<",\"inventory_at_observation\":";
        if(memory.contents_known)json_array(out,memory.inventory);else out<<"null";out<<'}';}
    out<<"],\"requests\":[";comma=false;
    for(const auto& request:requests)if(request.from==a.id||request.to==a.id){if(comma)out<<',';comma=true;
        bool own_task=request.to==a.id;
        out<<"{\"id\":"<<request.id<<",\"from\":"<<request.from<<",\"to\":"<<request.to<<",\"intent\":"<<request.intent
           <<",\"quantity\":"<<request.quantity<<",\"x\":"<<request.x<<",\"y\":"<<request.y<<",\"status\":"<<(own_task?request.status:request.from_status)
           <<",\"progress\":"<<(own_task?request.progress:request.from_progress)<<",\"contact_x\":"<<request.contact_x<<",\"contact_y\":"<<request.contact_y
           <<",\"knowledge_tick\":"<<(own_task?tick:request.from_updated)<<",\"report_tick\":"<<request.reported_tick;
        if(own_task)out<<",\"completed_tick\":"<<request.completed_tick;
        out<<",\"created_tick\":"<<request.created<<",\"answered_tick\":"<<request.answered<<",\"expires\":"<<request.expires<<'}';}
    out<<']';return out.str();
}

std::string World::social_debug() const {
    std::ostringstream out;out<<",\"rules_version\":"<<rules_version<<",\"agreements\":[";bool comma=false;
    for(const auto& group:groups){if(comma)out<<',';comma=true;out<<"{\"id\":"<<group.id<<",\"founder\":"<<group.founder<<",\"name\":"<<std::quoted(group.name)<<",\"revision\":"<<group.revision<<'}';}
    out<<"],\"caches\":[";comma=false;for(const auto& cache:caches){if(comma)out<<',';comma=true;cache_json(out,cache,true);}
    out<<']';return out.str();
}

void World::save_social(std::ostream& out) const {
    out<<next_group<<' '<<groups.size()<<'\n';
    for(const auto& group:groups)out<<group.id<<' '<<group.founder<<' '<<group.created<<' '<<group.revision<<' '<<std::quoted(group.name)<<'\n';
    out<<caches.size()<<'\n';
    for(const auto& cache:caches) {
        out<<cache.id<<' '<<cache.creator<<' '<<cache.group<<' '<<cache.x<<' '<<cache.y<<' '<<cache.created<<'\n';
        for(auto value:cache.inventory)out<<value<<' ';
        out<<'\n';
        for(auto value:cache.materials)out<<value<<' ';
        out<<'\n';
    }
    out<<people.size()<<'\n';
    for(const auto& actor:people) {
        out<<actor.id<<' '<<actor.relationships.size()<<'\n';
        for(const auto& [id,relation]:actor.relationships)out<<id<<' '<<relation.trust<<' '<<relation.helped<<' '<<relation.broken<<' '<<relation.last_tick<<'\n';
        out<<actor.known_groups.size()<<'\n';
        for(const auto& [id,known]:actor.known_groups)out<<id<<' '<<known.revision<<' '<<known.learned_tick<<' '<<std::quoted(known.name)<<'\n';
        out<<actor.places.size()<<'\n';
        for(const auto& memory:actor.places){out<<memory.id<<' '<<memory.x<<' '<<memory.y<<' '<<memory.group<<' '<<memory.seen_tick<<' '<<memory.contents_tick<<' '<<memory.contents_known;for(auto value:memory.inventory)out<<' '<<value;out<<'\n';}
    }
    out<<requests.size()<<'\n';for(const auto& r:requests)out<<r.id<<' '<<r.from<<' '<<r.to<<' '<<r.intent<<' '<<r.quantity<<' '<<r.x<<' '<<r.y<<' '<<r.status<<' '<<r.progress<<' '<<r.created<<' '<<r.expires<<' '<<r.answered<<' '
        <<r.from_status<<' '<<r.from_progress<<' '<<r.contact_x<<' '<<r.contact_y<<' '<<r.from_updated<<' '<<r.reported_tick<<' '<<r.completed_tick<<' '<<r.from_penalized<<'\n';
    out<<redacted_targets.size()<<'\n';for(const auto& [id,hidden]:redacted_targets){out<<id<<' '<<hidden.size();for(auto observer:hidden)out<<' '<<observer;out<<'\n';}
}

void World::load_social(std::istream& in) {
    auto require=[&](bool value){if(!value||!in)throw std::invalid_argument("invalid social snapshot");};
    auto count=[&](int limit){int n=-1;in>>n;require(n>=0&&n<=limit);return n;};
    std::set<Id> identities;
    for(const auto& a:people)identities.insert(a.id);
    for(const auto& n:nodes)identities.insert(n.id);
    for(const auto& o:offers)identities.insert(o.id);
    for(const auto& e:events)identities.insert(e.id);
    auto identity=[&](Id id){require(id>0&&id<next_id&&identities.insert(id).second);};
    in>>next_group;require(next_group>=5&&next_group<=1000000000);
    int n=count(1024);std::set<int> group_ids;
    require(n>=4&&next_group==n+1);
    for(int i=0;i<n;++i){GroupAgreement group{};in>>group.id>>group.founder>>group.created>>group.revision>>std::quoted(group.name);
        require(group.id>0&&group.id<next_group&&group_ids.insert(group.id).second&&(!group.founder||individual(group.founder))&&group.created<=tick&&group.revision>0&&group.revision<=1000000000000ULL&&valid_name(group.name));
        require(group.id<=4?(group.founder==0&&group.created==0):(group.founder!=0&&group.created>0));groups.push_back(group);}
    for(const auto& a:people)require(a.group==0||group_ids.contains(a.group));
    for(const auto& offer:offers)require(offer.group==0||group_ids.contains(offer.group));
    n=count(512);
    for(int i=0;i<n;++i){SharedCache cache{};in>>cache.id>>cache.creator>>cache.group>>cache.x>>cache.y>>cache.created;identity(cache.id);
        require((!cache.creator||individual(cache.creator))&&group_ids.contains(cache.group)&&location(cache.x,cache.y)&&cache.created<=tick);
        auto owner=std::find_if(groups.begin(),groups.end(),[&](const auto& g){return g.id==cache.group;});
        require(cache.created>=owner->created&&(cache.creator?cache.created>0:(cache.created==0&&cache.group<=4)));
        for(auto& v:cache.inventory){in>>v;require(v>=0&&v<=cache_capacity);}require(total(cache.inventory)<=cache_capacity);
        for(auto& v:cache.materials)in>>v;
        require(cache.materials==std::array<int,5>{0,0,3,0,2});caches.push_back(cache);}
    n=count(64);require(std::size_t(n)==people.size());std::set<Id> seen_people;
    for(int i=0;i<n;++i){Id id;in>>id;auto actor=individual(id);require(actor&&seen_people.insert(id).second);
        int m=count(64);for(int j=0;j<m;++j){Id other;Relationship relation{};in>>other>>relation.trust>>relation.helped>>relation.broken>>relation.last_tick;
            require(other!=id&&individual(other)&&actor->familiarity.contains(other)&&relation.trust>=-100&&relation.trust<=100&&relation.helped<=1000000000000ULL&&relation.broken<=1000000000000ULL&&relation.last_tick<=tick&&actor->relationships.emplace(other,relation).second);}
        m=count(1024);for(int j=0;j<m;++j){int group_id;KnownGroup known;in>>group_id>>known.revision>>known.learned_tick>>std::quoted(known.name);
            auto group=std::find_if(groups.begin(),groups.end(),[&](const auto& g){return g.id==group_id;});
            require(group!=groups.end()&&known.revision>0&&known.revision<=group->revision&&known.learned_tick<=tick&&valid_name(known.name)&&(known.revision!=group->revision||known.name==group->name)&&actor->known_groups.emplace(group_id,known).second);}
        m=count(32);std::set<Id> places;
        for(int j=0;j<m;++j){PlaceMemory memory{};in>>memory.id>>memory.x>>memory.y>>memory.group>>memory.seen_tick>>memory.contents_tick>>memory.contents_known;
            for(auto& v:memory.inventory){in>>v;require(v>=0&&v<=cache_capacity);}
            auto cache=std::find_if(caches.begin(),caches.end(),[&](const auto& c){return c.id==memory.id;});
            require(cache!=caches.end()&&places.insert(memory.id).second&&memory.x==cache->x&&memory.y==cache->y&&memory.group==cache->group&&memory.seen_tick<=tick&&memory.contents_tick<=memory.seen_tick&&total(memory.inventory)<=cache_capacity);
            require(memory.contents_known||(memory.contents_tick==0&&total(memory.inventory)==0));actor->places.push_back(memory);}
    }
    n=count(512);
    for(int i=0;i<n;++i){SocialRequest r{};in>>r.id>>r.from>>r.to>>r.intent>>r.quantity>>r.x>>r.y>>r.status>>r.progress>>r.created>>r.expires>>r.answered
        >>r.from_status>>r.from_progress>>r.contact_x>>r.contact_y>>r.from_updated>>r.reported_tick>>r.completed_tick>>r.from_penalized;identity(r.id);
        require(individual(r.from)&&individual(r.to)&&r.from!=r.to&&r.intent>=0&&r.intent<=3&&r.quantity>=1&&r.quantity<=4&&location(r.x,r.y)&&r.status>=0&&r.status<=4&&r.progress>=0&&r.progress<=(r.intent<2?r.quantity:r.intent==2?1:10)&&r.created<=tick&&r.expires==r.created+60&&r.answered<=tick&&(!r.answered||r.answered>=r.created));
        require(r.created>0&&(r.status>1||r.expires>=tick));
        require(r.status!=0||(r.answered==0&&r.progress==0));
        require((r.status!=1&&r.status!=2&&r.status!=3)||r.answered>0);
        require(r.from_status>=0&&r.from_status<=4&&r.from_progress>=0&&r.from_progress<=(r.intent<2?r.quantity:r.intent==2?1:10)&&location(r.contact_x,r.contact_y));
        require(r.from_updated>=r.created&&r.from_updated<=tick&&r.reported_tick<=tick&&r.completed_tick<=tick);
        require((r.status==3)==(r.completed_tick>0));
        if(r.completed_tick)require(r.completed_tick>=r.answered&&r.completed_tick<=r.expires);
        if(r.reported_tick)require(r.reported_tick>=r.answered);
        require(r.from_status!=3||r.status==3);
        require(!r.from_penalized||r.from_status==4);
        require(r.status!=3||r.progress==(r.intent<2?r.quantity:r.intent==2?1:10));requests.push_back(r);}
    n=count(4096);
    for(int i=0;i<n;++i){Id id;in>>id;int m=count(64);auto event=std::find_if(events.begin(),events.end(),[&](const auto& e){return e.id==id;});require(event!=events.end()&&!redacted_targets.contains(id));
        std::set<Id> hidden;std::vector<Id> values;
        for(int j=0;j<m;++j){Id observer;in>>observer;require(observer!=event->actor&&hidden.insert(observer).second&&std::find(event->witnesses.begin(),event->witnesses.end(),observer)!=event->witnesses.end());values.push_back(observer);}
        redacted_targets[id]=values;}
}
}
