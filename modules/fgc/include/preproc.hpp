
#ifndef _PRE_PROC_HPP_
#define _PRE_PROC_HPP_

#include "log.h"
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#define __FILENAME__                                                           \
    (__builtin_strrchr(__FILE__, '/')                                          \
         ? __builtin_strrchr(__FILE__, '/') + 1                                \
         : __FILE__) // only show filename and not it's path (less clutter)
#endif
