/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
// platform (primarily: OS) abstraction

#ifndef __PLATFORM_H__
#define __PLATFORM_H__

// Shared return values
#define PLATFORM_WAIT_FAILED 4
#define PLATFORM_WAIT_TIMEOUT 5
#define PLATFORM_WAIT_SUCCEEDED 0

#ifdef WIN32
#include <assert.h>
#include <stdint.h>

#define NOMINMAX // don't define min and max macros

#include <windows.h>

// just in case...:
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// atomic operations
inline int interlocked_increment(int volatile *val) {
    return (int)InterlockedIncrement((LONG volatile *)val);
}
inline int interlocked_decrement(int volatile *val) {
    return (int)InterlockedDecrement((LONG volatile *)val);
}

// memory allocation
inline void *malloc_aligned(int size, int alignment) {
    return _aligned_malloc(size, alignment);
}
inline void free_aligned(void *ptr) { _aligned_free(ptr); }

// synchronization
typedef CRITICAL_SECTION Lock;

inline bool lock_create(Lock *lock) {
    InitializeCriticalSection(lock);
    return true;
}
inline void lock_destroy(Lock *lock) { DeleteCriticalSection(lock); }
inline void lock_acquire(Lock *lock) { EnterCriticalSection(lock); }
inline void lock_release(Lock *lock) { LeaveCriticalSection(lock); }

typedef HANDLE Semaphore;

inline bool semaphore_create(Semaphore *sem, int init_count, int max_count) {
    *sem = CreateSemaphore(NULL, init_count, max_count, NULL);
    return *sem != NULL;
}
inline void semaphore_destroy(Semaphore *sem) {
    CloseHandle(*sem);
    *sem = NULL;
}
inline void semaphore_signal(Semaphore *sem) {
    ReleaseSemaphore(*sem, 1, NULL);
}
inline uint32_t semaphore_wait(Semaphore *sem) {
    DWORD ret = WaitForSingleObject(*sem, INFINITE);
    // No timeout
    return ((ret == WAIT_OBJECT_0) || (ret == WAIT_ABANDONED))
               ? PLATFORM_WAIT_SUCCEEDED
               : PLATFORM_WAIT_FAILED;
}
inline uint32_t semaphore_check(Semaphore *sem) {
    DWORD ret = WaitForSingleObject(*sem, 0);
    if (WAIT_TIMEOUT == ret)
        return PLATFORM_WAIT_TIMEOUT;
    return ((ret == WAIT_OBJECT_0) || (ret == WAIT_ABANDONED))
               ? PLATFORM_WAIT_SUCCEEDED
               : PLATFORM_WAIT_FAILED;
}

typedef HANDLE Event;

inline bool event_create(Event *evt) {
    *evt = CreateEvent(NULL, FALSE, FALSE, NULL);
    return *evt != NULL;
}
inline void event_destroy(Event *evt) {
    CloseHandle(*evt);
    *evt = NULL;
}
inline void event_signal(Event *evt) { SetEvent(*evt); }
inline uint32_t event_wait(Event *evt) {
    DWORD ret = WaitForSingleObject(*evt, INFINITE);
    // No timeout
    return ((ret == WAIT_OBJECT_0) || (ret == WAIT_ABANDONED))
               ? PLATFORM_WAIT_SUCCEEDED
               : PLATFORM_WAIT_FAILED;
}
inline uint32_t event_wait(Event *evt, uint32_t timeout) {
    DWORD ret = WaitForSingleObject(*evt, timeout);
    if (WAIT_TIMEOUT == ret)
        return PLATFORM_WAIT_TIMEOUT;
    return ((ret == WAIT_OBJECT_0) || (ret == WAIT_ABANDONED))
               ? PLATFORM_WAIT_SUCCEEDED
               : PLATFORM_WAIT_FAILED;
}

// thread support
typedef HANDLE Thread;

DWORD WINAPI ThreadStartProc(LPVOID lpParameter);

struct thread_create_s {
    void *(*func)(void *);
    void *par;
};

inline bool thread_create(Thread *thread, void *(*func)(void *), void *par) {
    thread_create_s *p = new thread_create_s;
    if (p == NULL)
        return false;
    p->func = func;
    p->par = par;
    *thread = CreateThread(NULL, 0, ThreadStartProc, p, 0, NULL);
    return *thread != NULL;
}

inline void thread_destroy(Thread *thread) { /*TerminateThread(*thread, 0);*/
    CloseHandle(*thread);
    *thread = NULL;
}
inline uint32_t thread_wait(Thread *thread) {
    DWORD ret = WaitForSingleObject(*thread, INFINITE);
    return ((ret == WAIT_OBJECT_0) || (ret == WAIT_ABANDONED))
               ? PLATFORM_WAIT_SUCCEEDED
               : PLATFORM_WAIT_FAILED;
}
inline uint32_t thread_wait(Thread *thread, uint32_t millis) {
    DWORD ret = WaitForSingleObject(*thread, millis);
    if (WAIT_TIMEOUT == ret)
        return PLATFORM_WAIT_TIMEOUT;
    return ((ret == WAIT_OBJECT_0) || (ret == WAIT_ABANDONED))
               ? PLATFORM_WAIT_SUCCEEDED
               : PLATFORM_WAIT_FAILED;
}
inline void thread_priority_set(Thread *thread, int priority) {
    SetThreadPriority(*thread, priority);
}
inline int thread_priority_get(Thread *thread) {
    return GetThreadPriority(*thread);
}

int cpu_features();

#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)
inline unsigned int bitscan_forward_32(unsigned int val) {
    DWORD ndx;
    _BitScanForward(&ndx, val);
    return ndx;
}
inline unsigned int bitscan_reverse_32(unsigned int val) {
    DWORD ndx;
    _BitScanReverse(&ndx, val);
    return ndx;
}

#else // GCC

#include <assert.h>
#include <stdint.h>

typedef uint64_t LARGE_INTEGER;

#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <time.h>

#ifndef INFINITE
#define INFINITE 0xffffff
#endif

// atomic operations
inline int interlocked_increment(int volatile *val) {
    return (int)__sync_add_and_fetch((int volatile *)val, 1);
}
inline int interlocked_decrement(int volatile *val) {
    return (int)__sync_sub_and_fetch((int volatile *)val, 1);
}

// memory allocation
inline void *malloc_aligned(int size, int alignment) {
    void *ptr;
    posix_memalign(&ptr, (size_t)alignment, (size_t)size);
    return ptr;
}
inline void free_aligned(void *ptr) { free(ptr); }

// synchronization
typedef pthread_mutex_t Lock;

inline bool lock_create(Lock *lock) {
    pthread_mutex_init(lock, NULL);
    return true;
}
inline void lock_destroy(Lock *lock) { pthread_mutex_destroy(lock); }
inline void lock_acquire(Lock *lock) { pthread_mutex_lock(lock); }
inline void lock_release(Lock *lock) { pthread_mutex_unlock(lock); }

typedef sem_t Semaphore;

inline bool semaphore_create(Semaphore *sem, int init_count, int max_count) {
    sem_init(sem, 0, init_count);
    return true;
}
inline void semaphore_destroy(Semaphore *sem) { sem_destroy(sem); }
inline void semaphore_signal(Semaphore *sem) { sem_post(sem); }
inline int semaphore_wait(Semaphore *sem) { return sem_wait(sem); }
inline int semaphore_check(Semaphore *sem) { return sem_trywait(sem); }

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Event_s;

typedef Event_s *Event;

inline bool event_create(Event *evt) {
    *evt = (Event)malloc(sizeof(Event_s));
    if (!(*evt))
        return false;
    if (!pthread_mutex_init(&(*evt)->mutex, NULL)) {
        free((*evt));
        (*evt) = 0;
        return false;
    }
    if (!pthread_cond_init(&(*evt)->cond, NULL)) {
        free((*evt));
        (*evt) = 0;
        return false;
    }
    return true;
}
inline void event_destroy(Event *evt) {
    if ((*evt)) {
        pthread_mutex_destroy(&(*evt)->mutex);
        pthread_cond_destroy(&(*evt)->cond);
        free(evt);
        (*evt) = 0;
    }
}
inline void event_signal(Event *evt) {
    pthread_mutex_lock(&(*evt)->mutex);
    pthread_cond_signal(&(*evt)->cond);
    pthread_mutex_unlock(&(*evt)->mutex);
}
inline uint32_t event_wait(Event *evt) {
    pthread_mutex_lock(&(*evt)->mutex);
    int res = pthread_cond_wait(&(*evt)->cond, &(*evt)->mutex);
    pthread_mutex_unlock(&(*evt)->mutex);
    if (0 != res) {
        return PLATFORM_WAIT_FAILED;
    }
    return PLATFORM_WAIT_SUCCEEDED;
}

inline uint32_t event_wait(Event *evt, uint32_t millis) {
    timespec abs_time;
    if (clock_gettime(CLOCK_REALTIME, &abs_time) != 0) {
        return PLATFORM_WAIT_FAILED;
    }
    abs_time.tv_sec += millis / 1000;
    abs_time.tv_nsec += (millis % 1000) * 1000000;
    pthread_mutex_lock(&(*evt)->mutex);

    int res = 0;
    if (millis == INFINITE) {
        res = pthread_cond_wait(&(*evt)->cond, &(*evt)->mutex);
    } else {
        res = pthread_cond_timedwait(&(*evt)->cond, &(*evt)->mutex, &abs_time);
    }
    pthread_mutex_unlock(&(*evt)->mutex);
    if (res == ETIMEDOUT) {
        return PLATFORM_WAIT_TIMEOUT;
    }
    if (0 != res) {
        return PLATFORM_WAIT_FAILED;
    }
    return PLATFORM_WAIT_SUCCEEDED;
}

// thread support
typedef pthread_t Thread;

inline bool thread_create(Thread *thread, void *(*func)(void *), void *par) {
    int ret = pthread_create(thread, NULL, func, par);
    return (ret == 0);
}

inline void thread_destroy(Thread *thread) { pthread_join(*thread, NULL); }
inline void thread_wait(Thread *thread) { pthread_join(*thread, NULL); }
inline uint32_t thread_wait(Thread *thread, uint32_t millis) {
    timespec abs_time;
    if (clock_gettime(CLOCK_REALTIME, &abs_time) != 0) {
        return PLATFORM_WAIT_FAILED;
    }
    abs_time.tv_sec += millis / 1000;
    abs_time.tv_nsec += (millis % 1000) * 1000000;
    pthread_join(*thread, NULL);
    // int res = pthread_timedjoin_np(*thread,&ret, &abs_time);
    /*if (res == ETIMEDOUT) {
        return PLATFORM_WAIT_TIMEOUT;
    }
    if (0 != res) {
        return PLATFORM_WAIT_FAILED;
    }*/

    return PLATFORM_WAIT_SUCCEEDED;
}

inline void thread_priority_set(Thread *thread, int priority) {
    struct sched_param param;
    int policy = SCHED_RR;
    param.sched_priority = priority;
    pthread_setschedparam(*thread, policy, &param);
}

inline int thread_priority_get(Thread *thread) {
    struct sched_param param;
    int policy;
    pthread_getschedparam(*thread, &policy, &param);
    return param.sched_priority;
}

inline unsigned int bitscan_forward_32(unsigned int val) {
    return __builtin_clz(val);
}
inline unsigned int bitscan_reverse_32(unsigned int val) {
    return __builtin_ctz(val);
}

#endif // WIN32

// Locks the process to the first CPU
void winlockProcessCPU();
// time related functions
void winSleep(int time);
void winGetCurrentTimeTicks(LARGE_INTEGER *ticks);
void winSetFrequency(LARGE_INTEGER *ticks);

#endif // __PLATFORM_H__
