#ifndef __LOG_H__
#define __LOG_H__
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>

static std::time_t time_now = std::time(nullptr);

#define INFO \
    time_now = std::time(nullptr); \
    std::clog << std::put_time(std::localtime(&time_now), "%y-%m-%d %OH:%OM:%OS") << " [INFO] " << __FILENAME__ << "(" << __FUNCTION__ << ":" << __LINE__ << ") >> "

#define ERROR \
    time_now = std::time(nullptr); \
    std::clog << std::put_time(std::localtime(&time_now), "%y-%m-%d %OH:%OM:%OS") << " [ERROR] " << __FILENAME__ << "(" << __FUNCTION__ << ":" << __LINE__ << ") >> "


#endif
