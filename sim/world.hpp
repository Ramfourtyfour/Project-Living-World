#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace lw {
using Id = std::uint64_t;
constexpr int extent = 24000; // Centimetres, quantized continuous plane.
constexpr int sight = 1800, reach = 300, capacity = 24;
enum Kind { Wait, Move, Inspect, Gather, Consume, OfferGift, Accept, Trade, Rest,
            Depart, Invite, Decline };
struct Action {
    Id actor = 0, request = 0;
    int kind = Wait;
    Id target = 0;
    int x = 0, y = 0, resource = 0, quantity = 1, ask_resource = 0, ask_quantity = 0;
};
struct Memory { Id id; int resource, x, y, amount; std::uint64_t tick; };
struct Individual {
    Id id = 0;
    int x = 0, y = 0, group = 0, health = 100, hunger = 0, thirst = 0, fatigue = 0;
    std::array<int, 5> inventory{};
    Id last_request = 0;
    int style = 0;
    std::vector<Memory> memory;
    std::map<Id, int> familiarity;
};
struct Node { Id id; int resource, x, y, amount, maximum, period; };
struct Obstacle { int x, y, radius; };
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
class World {
public:
    std::uint64_t seed = 0, rng = 1, tick = 0;
    Id next_id = 1;
    std::vector<Individual> people;
    std::vector<Node> nodes;
    std::vector<Obstacle> obstacles;
    std::vector<Offer> offers;
    std::deque<Event> events;
    std::vector<Result> results;
    std::array<std::int64_t, 5> initial{}, regenerated{}, consumed{};
    explicit World(std::uint64_t seed = 1, int population = 32, int scenario = 0);
    void step(const std::vector<Action>& actions);
    std::string observation(Id actor) const;
    std::string debug_json() const;
    std::string events_json(Id observer = 0) const;
    std::string results_json() const;
    std::string save() const;
    static World load(const std::string& text);
    std::string hash() const;
    bool conserved() const;
    bool sees(const Individual& a, int x, int y) const;
    Individual* individual(Id id);
    const Individual* individual(Id id) const;
private:
    std::uint64_t random();
    std::string apply(Individual& actor, const Action& action);
    void refresh_memory();
    void emit(Id actor, Id target, const std::string& type,
              const std::string& reason = "", int quantity = 0, bool private_event = false);
};
}
