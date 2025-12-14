#include "timed-hash-queue.h"
#include <iostream>

TimedHashQueue::TimedHashQueue() {}

void TimedHashQueue::addHash(int64_t hash) {
    const int count = this->hash_map.size();

    if (this->hash_map.find(hash) != this->hash_map.end()) {
        // Hash already exists, do nothing
        return;
    }

    if (count == 0) {
        this->head = new Node(hash, nullptr, nullptr);
        this->tail = head;
    } else {
        auto new_node = new Node(hash, this->tail, nullptr);
        this->tail->next = new_node;
        this->tail = new_node;
    }

    this->hash_map[hash] = this->tail;
}

void TimedHashQueue::touchHash(int64_t hash) {
        if (this->hash_map.find(hash) == this->hash_map.end()) {
        return; // Hash not found, do nothing
    }

    auto node = this->hash_map[hash];

    if (node == this->tail) {
        return; // Already the most recently used
    }

    // Connecting the previous and next nodes to each other
    node->next->prev = node->prev;    
    if (node->prev != nullptr) {
        node->prev->next = node->next;
    } else {
        this->head = node->next; // Update head if the node is the first node
    }

    // Attaching the node to the end of the list
    this->tail->next = node;
    node->next = nullptr;
    node->prev = this->tail;

    // Saving the node as the new tail
    this->tail = node;
}

int TimedHashQueue::removeOldestHash() {
    if (this->hash_map.size() == 0) {
        throw std::runtime_error("TimedHashQueue: Attempting to remove hash from empty queue.");
    }

    auto output = this->head->hash;
    this->removeHash(output);

    return output;
}

void TimedHashQueue::removeHash(int64_t hash) {
    const int count = this->hash_map.size();

    if (count == 0) {
        return;
    }

    if (this->hash_map.find(hash) == this->hash_map.end()) {
        return;
    }

    auto node = this->hash_map[hash];

    // Update the previous node's next pointer
    if (node->prev != nullptr) {
        node->prev->next = node->next;
    } else {
        this->head = node->next; // Update head if the node is the first node
    }

    // Update the next node's prev pointer
    if (node->next != nullptr) {
        node->next->prev = node->prev;
    } else {
        this->tail = node->prev; // Update tail if the node is the last node
    }

    // Remove the node from the hash map and free memory
    this->hash_map.erase(hash);
    delete node;
}

bool TimedHashQueue::containsHash(int64_t hash) const {
    return this->hash_map.find(hash) != this->hash_map.end();
}

size_t TimedHashQueue::size() const {
    return this->hash_map.size();
}

TimedHashQueue::~TimedHashQueue() {
    auto current = this->head;
    while (current != nullptr) {
        auto next_node = current->next;
        delete current;
        current = next_node;
    }
}