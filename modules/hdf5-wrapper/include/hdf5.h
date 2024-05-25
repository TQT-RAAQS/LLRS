#ifndef HDF5_WRAPPER_
#define HDF5_WRAPPER_

#include <iostream>
#include <string>

class HDF5 {

    std::string address;

public:

    HDF5(std::string address) :
        address(address) {}

    void say_hello();
};

#endif