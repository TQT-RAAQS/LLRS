#ifndef HDF5_WRAPPER_GLOBALS_NOT_FOUND_EXCEPTION_H_
#define HDF5_WRAPPER_GLOBALS_NOT_FOUND_EXCEPTION_H_

#include <cstring>
#include <iostream>

class GlobalsNotFoundException : public std::exception {
    std::string message;

  public:
    GlobalsNotFoundException(const std::string &message) : message(message) {}
    const char *what() const noexcept override { return message.c_str(); }
};

#endif