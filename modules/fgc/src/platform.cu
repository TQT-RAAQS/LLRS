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
#include "platform.hpp"

#ifdef WIN32
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <windows.h>

DWORD WINAPI ThreadStartProc(LPVOID lpParameter) {
    thread_create_s p = *(thread_create_s *)lpParameter; // copy to stack
    delete (thread_create_s *)lpParameter;
    p.func(p.par);
    return 0;
}

void winlockProcessCPU() {
    // There is no way to detect where the GPU is connected
    // on windows. DEVPKEY_Device_Numa_Node and DEVPKEY_Numa_Proximity_Domain
    // are the closest property, but they return not existing for the GPU on
    // win7 and win8. This test runs on one GPU and we assume this is attached
    // to the first GPU. Note the system has to have the _PXM property set for
    // this to be detected by the OS properly.

    // First see if this is a numa machine at all
    ULONG highestNumaNode = 0;
    BOOL isNuma = GetNumaHighestNodeNumber(&highestNumaNode);

    if (isNuma && highestNumaNode > 0) {
        printf(
            "WARNING!!! - Numa architecture detected, defaulting to CPU %d \n",
            highestNumaNode);
        HANDLE process = GetCurrentProcess();
        ULONG currentProctAffinityMask = 0, currentSysAffinityMask = 0;
        ULONGLONG newMask = 0x3F; // 2nd CPU 0xFC0;
        BOOL res = GetProcessAffinityMask(process,
                                          (PDWORD_PTR)&currentProctAffinityMask,
                                          (PDWORD_PTR)&currentSysAffinityMask);

        if (res) {
            // get the mask for the first node
            res = GetNumaNodeProcessorMask(0, &newMask);
            if (res) {
                res = SetProcessAffinityMask(process, (DWORD_PTR)newMask);

                printf("Salling SetProcessAffinityMask prevmask 0x%lu sysmask "
                       "0x%lu newMask 0x%llu\n",
                       currentProctAffinityMask, currentSysAffinityMask,
                       newMask);
                if (!res) {
                    printf("SetProcessAffinityMask failed process 0x%p error "
                           "0x%x\n",
                           process, GetLastError());
                } else {
                    printf("SetProcessAffinityMask succeeded process 0x%p "
                           "error 0x%x\n",
                           process, GetLastError());
                }
            }
        }
    }
}

void winGetCurrentTimeTicks(LARGE_INTEGER *ticks) {
    QueryPerformanceCounter(ticks);
}

void winSetFrequency(LARGE_INTEGER *ticks) { QueryPerformanceFrequency(ticks); }

void winSleep(int time) { Sleep(time); }

#else
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

void winlockProcessCPU() { assert(!"unimplemented"); }

uint64_t perfFreq = 1000000;

void winGetCurrentTimeTicks(uint64_t *ticks) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *ticks = (uint64_t)tv.tv_sec * perfFreq + tv.tv_usec;
}

void winSetFrequency(uint64_t *ticks) { *ticks = 1000000; }

void winSleep(int time) { usleep(time * 1000); }

#endif // WIN32
