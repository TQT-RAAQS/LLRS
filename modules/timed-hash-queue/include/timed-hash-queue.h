#ifndef _TIMED_HASH_QUEUE_
#define _TIMED_HASH_QUEUE_

#include <unordered_map>
#include <cstdint>

struct Node {
    
    int64_t hash;
    Node *prev;
    Node *next;

    Node(int64_t h, Node* prev = nullptr, Node* next = nullptr) : hash(h), prev(prev), next(next) {}
};

class TimedHashQueue {
    
    Node *head = nullptr, *tail = nullptr;
    std::unordered_map<int64_t, Node*> hash_map;

public:
    TimedHashQueue();
    ~TimedHashQueue();

    TimedHashQueue(const TimedHashQueue&) = delete;
    TimedHashQueue& operator=(const TimedHashQueue&) = delete;

    void clear();

    void add_hash(int64_t hash);

    void touch_hash(int64_t hash);

    int64_t remove_oldest_hash();

    void remove_hash(int64_t hash);

    bool contains_hash(int64_t hash) const;

    const Node* getHead() const { return this->head; };
    const Node* getTail() const { return  this->tail; };
 
    size_t size() const;
};

#endif