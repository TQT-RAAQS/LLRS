#ifndef LOG_H_
#define LOG_H_
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

static std::time_t time_now = std::time(nullptr);

#define INFO                                                                   \
    time_now = std::time(nullptr);                                             \
    std::clog << std::put_time(std::localtime(&time_now),                      \
                               "%y-%m-%d %OH:%OM:%OS")                         \
              << " [INFO] " << __FILENAME__ << "(" << __FUNCTION__ << ":"      \
              << __LINE__ << ") >> "

#define ERROR                                                                  \
    time_now = std::time(nullptr);                                             \
    std::clog << std::put_time(std::localtime(&time_now),                      \
                               "%y-%m-%d %OH:%OM:%OS")                         \
              << " [ERROR] " << __FILENAME__ << "(" << __FUNCTION__ << ":"     \
              << __LINE__ << ") >> "

#define __FILENAME__                                                           \
    (__builtin_strrchr(__FILE__, '/')                                          \
         ? __builtin_strrchr(__FILE__, '/') + 1                                \
         : __FILE__) // only show filename and not it's path (less clutter)

#endif
