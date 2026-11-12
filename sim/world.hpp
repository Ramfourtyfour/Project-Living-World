#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace lw {
using Id = std::uint64_t;
constexpr int extent = 24000; // Centimetres, quantized continuous plane.
constexpr int sight = 1800, reach = 300, capacity = 24;
enum Kind { Wait, Move, Inspect, Gather, Consume, OfferGift, Accept, Trade, Rest,
            Depart, Invite, Decline, FoundGroup, RenameGroup, RequestHelp,
            AnswerRequest, Deposit, Withdraw, PlaceCache, ReportRequest, Turn, Call, Crouch };
struct Action {
    Id actor = 0, request = 0;
    int kind = Wait;
    Id target = 0;
    int x = 0, y = 0, resource = 0, quantity = 1, ask_resource = 0, ask_quantity = 0;
    std::string text{};
};
struct Memory { Id id; int resource, x, y, amount; std::uint64_t tick; };
struct Relationship {
    int trust = 0;
    std::uint64_t helped = 0, broken = 0, last_tick = 0;
};
struct PlaceMemory {
    Id id;
    int x, y, group;
    std::uint64_t seen_tick, contents_tick;
    bool contents_known;
    std::array<int, 5> inventory{};
};
struct KnownGroup {
    std::string name;
    std::uint64_t revision = 1, learned_tick = 0;
};
struct PersonMemory {
    Id id;
    int x, y, group;
    bool alive;
    std::uint64_t tick;
};
struct LandmarkMemory { Id id; int x,y,radius; std::uint64_t tick; };
struct HeardCue {
    std::uint64_t id, tick;
    int origin_x, origin_y, bearing, distance_band, confidence;
    std::string kind;
};
struct Cover { Id id; int x, y, radius, density; };
struct PendingSound { Id actor; int x, y, radius; std::string kind; };
struct Individual {
    Id id = 0;
    int x = 0, y = 0, group = 0, health = 100, hunger = 0, thirst = 0, fatigue = 0;
    std::array<int, 5> inventory{};
    Id last_request = 0;
    int style = 0;
    std::vector<Memory> memory;
    std::map<Id, int> familiarity;
    std::map<Id, Relationship> relationships;
    std::vector<PlaceMemory> places;
    std::map<int, KnownGroup> known_groups;
    int facing_x = 1000, facing_y = 0;
    bool crouching = false;
    std::uint64_t next_cue = 1;
    std::vector<PersonMemory> person_memory;
    std::vector<LandmarkMemory> landmark_memory;
    std::deque<HeardCue> heard;
};
struct Node { Id id; int resource, x, y, amount, maximum, period; };
struct Obstacle { int x, y, radius; Id id = 0; };
struct Offer {
    Id id, from, to;
    int resource, quantity, ask_resource, ask_quantity, group;
    std::uint64_t expires;
};
struct Event {
    Id id, actor, target;
    std::uint64_t tick;
    std::string type, reason;
    int quantity;
    std::vector<Id> witnesses;
};
struct Result { Id actor, request; bool ok; std::string reason; };
struct GroupAgreement {
    int id;
    Id founder;
    std::uint64_t created;
    std::string name;
    std::uint64_t revision = 1;
};
struct SharedCache {
    Id id, creator;
    int group, x, y;
    std::uint64_t created;
    std::array<int, 5> inventory{}, materials{};
};
struct SocialRequest {
    Id id, from, to;
    int intent, quantity, x, y, status = 0, progress = 0;
    std::uint64_t created = 0, expires = 0, answered = 0;
    int from_status = 0, from_progress = 0, contact_x = 0, contact_y = 0;
    std::uint64_t from_updated = 0, reported_tick = 0, completed_tick = 0;
    bool from_penalized = false;
};
class World {
public:
    std::uint64_t seed = 0, rng = 1, tick = 0;
    Id next_id = 1;
    int rules_version = 1, next_group = 5;
    std::vector<Individual> people;
    std::vector<Node> nodes;
    std::vector<Obstacle> obstacles;
    std::vector<Offer> offers;
    std::deque<Event> events;
    std::vector<Result> results;
    std::array<std::int64_t, 5> initial{}, regenerated{}, consumed{};
    std::vector<GroupAgreement> groups;
    std::vector<SharedCache> caches;
    std::vector<SocialRequest> requests;
    std::map<Id, std::vector<Id>> redacted_targets;
    std::vector<Cover> cover;
    explicit World(std::uint64_t seed = 1, int population = 32, int scenario = 0, int rules = 1);
    void step(const std::vector<Action>& actions);
    std::string observation(Id actor) const;
    std::string debug_json() const;
    std::string events_json(Id observer = 0) const;
    std::string results_json() const;
    std::string save() const;
    static World load(const std::string& text);
    std::string hash() const;
    bool conserved() const;
    bool sees(const Individual& a, int x, int y, int concealment = 0) const;
    bool sees_person(const Individual& observer, const Individual& target) const;
    bool sees_obstacle(const Individual& observer, const Obstacle& obstacle) const;
    bool occluded(int ax, int ay, int bx, int by) const;
    int illumination() const;
    int cover_density(int x, int y) const;
    Individual* individual(Id id);
    const Individual* individual(Id id) const;
    void upgrade_rules();
    void upgrade_perception();
private:
    std::vector<PendingSound> pending_sounds;
    std::uint64_t random();
    std::string apply(Individual& actor, const Action& action);
    void refresh_memory();
    void emit(Id actor, Id target, const std::string& type,
              const std::string& reason = "", int quantity = 0, bool private_event = false);
    void initialize_social(bool starting_caches);
    std::string apply_social(Individual& actor, const Action& action);
    void refresh_social();
    void interaction(Id observer, Id other, int trust, bool help, bool broken);
    void transferred(Id giver, Id receiver, int resource, int quantity);
    void confirm_request(SocialRequest& request, bool credit);
    void tell(Id actor, Id target, const std::string& type, const std::vector<Id>& audience);
    std::string social_observation(const Individual& observer) const;
    std::string social_debug() const;
    void save_social(std::ostream& out) const;
    void load_social(std::istream& in);
    void initialize_perception(bool generate_cover);
    void face(Individual& actor, int x, int y);
    void make_sound(const Individual& actor, const std::string& kind, int radius);
    void refresh_perception();
    std::string perception_observation(const Individual& observer) const;
    std::string perception_debug() const;
    void save_perception(std::ostream& out) const;
    void load_perception(std::istream& in);
};
}
