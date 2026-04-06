#include "qdac-client.h"
#include <iostream>

int main() {
    auto client = QdacClient("default.yml");

    client.handshake();
    client.send_b_field(0);
}