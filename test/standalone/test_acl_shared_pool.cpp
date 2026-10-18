/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 */
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "acc_offload_acl_shared_pool.h"

using ock::offload::AccOffloadAclSharedPool;
namespace {
constexpr uint32_t RANKS = 2U;
constexpr uint64_t POOL_BYTES = 64U * 1024U;
constexpr uint64_t COMMON_ADDRESS_START = 0x200000000000ULL;
constexpr uint64_t TEST_DVA_BASE = 0x600000000000ULL;
constexpr uint64_t TEST_DVA_STRIDE = 1ULL << 32U;
constexpr uint32_t TEST_REGISTER_FLAGS = 0x10000002U;
uint32_t rankId;
int registerCount;
int unregisterCount;
int failureKind;
bool failUnregister;

void Require(bool success, const char *message)
{
    if (!success) {
        throw std::runtime_error(message);
    }
}

void Transfer(int fd, char *data, size_t bytes, bool sending)
{
    while (bytes != 0U) {
        const ssize_t count = sending ? write(fd, data, bytes) : read(fd, data, bytes);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        Require(count > 0, "socket exchange failed");
        data += count;
        bytes -= static_cast<size_t>(count);
    }
}

AccOffloadAclSharedPool::Exchange Exchange(int fd)
{
    return [fd](const char *src, uint32_t srcBytes, char *dst, uint32_t dstBytes) {
        Require(dstBytes == RANKS * srcBytes, "invalid exchange size");
        std::memcpy(dst + rankId * srcBytes, src, srcBytes);
        char *peer = dst + (1U - rankId) * srcBytes;
        if (rankId == 0U) {
            Transfer(fd, const_cast<char *>(src), srcBytes, true);
            Transfer(fd, peer, srcBytes, false);
        } else {
            Transfer(fd, peer, srcBytes, false);
            Transfer(fd, const_cast<char *>(src), srcBytes, true);
        }
        return 0;
    };
}

int32_t SetDevice(int32_t device)
{
    Require(device == static_cast<int32_t>(rankId), "wrong device context");
    return failureKind == 3 && rankId == 1U ? -1 : 0;
}

int32_t Register(void *, uint64_t size, uint32_t flags)
{
    Require(size == POOL_BYTES && flags == TEST_REGISTER_FLAGS, "wrong ACL registration");
    if (failureKind == 1 && rankId == 1U) {
        return -1;
    }
    ++registerCount;
    return 0;
}

int32_t DevicePointer(void *, void **device, uint32_t flags)
{
    Require(flags == 0U, "wrong device-pointer flags");
    *device = reinterpret_cast<void *>(TEST_DVA_BASE + TEST_DVA_STRIDE * rankId);
    return failureKind == 2 && rankId == 1U ? -1 : 0;
}

int32_t Unregister(void *)
{
    if (failUnregister) {
        failUnregister = false;
        return -1;
    }
    ++unregisterCount;
    return 0;
}

void Barrier(const AccOffloadAclSharedPool::Exchange &exchange)
{
    uint32_t local = rankId;
    uint32_t peers[RANKS]{};
    Require(exchange(reinterpret_cast<const char *>(&local), sizeof(local), reinterpret_cast<char *>(peers),
                     sizeof(peers)) == 0,
            "barrier failed");
}

void VerifySharedBytes(AccOffloadAclSharedPool &pool, const AccOffloadAclSharedPool::Exchange &exchange, bool collision)
{
    const auto hva = reinterpret_cast<uint64_t>(pool.Base());
    Require(!collision || hva != COMMON_ADDRESS_START, "common HVA retry failed");
    auto *words = static_cast<uint32_t *>(pool.Base());
    const size_t wordCount = POOL_BYTES / sizeof(uint32_t);
    for (size_t index = rankId; index < wordCount; index += RANKS) {
        words[index] = static_cast<uint32_t>(index + 100U);
    }
    Barrier(exchange);
    for (size_t index = 0U; index < wordCount; ++index) {
        Require(words[index] == index + 100U, "shared backing content mismatch");
    }
    uint64_t dva = 0U;
    Require(pool.GetDva(hva + 123U, &dva) == 0, "DVA lookup failed");
    Require(dva == TEST_DVA_BASE + TEST_DVA_STRIDE * rankId + 123U && dva != hva + 123U,
            "DVA must use this process's registration and preserve offsets");
    Require(pool.GetDva(hva - 1U, &dva) != 0 && pool.GetDva(hva + POOL_BYTES, &dva) != 0 &&
                pool.GetDva(hva, nullptr) != 0,
            "DVA range validation failed");
    Barrier(exchange);
    if (rankId == 0U) {
        Require(pool.Close(), "owner cleanup failed");
    }
    Barrier(exchange);
    if (rankId != 0U) {
        Require(words[wordCount - 1U] == wordCount - 1U + 100U, "peer backing lost after owner close");
        failUnregister = true;
        Require(!pool.Close() && pool.Base() != nullptr, "must retain backing when unregister fails");
        Require(pool.Close(), "retry unregister failed");
    }
}

offload_config_t MakeConfig(int fault)
{
    offload_config_t config{};
    config.deviceId = rankId;
    config.reserveSize = POOL_BYTES + (fault == 5 && rankId == 1U ? POOL_BYTES : 0U);
    config.allocSize = rankId == 0U || fault == 4 ? POOL_BYTES : 0U;
    config.rankId = rankId;
    config.worldSize = RANKS;
    config.scene = OFFLOAD_SCENE_SHARED;
    return config;
}

void Worker(int fd, int fault, bool collision)
{
    alarm(15U);
    failureKind = fault;
    registerCount = 0;
    unregisterCount = 0;
    void *occupied = MAP_FAILED;
    if (collision && rankId == 1U) {
        occupied = mmap(reinterpret_cast<void *>(COMMON_ADDRESS_START), POOL_BYTES, PROT_NONE,
                        MAP_FIXED_NOREPLACE | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        Require(occupied != MAP_FAILED, "cannot reserve collision test range");
    }
    const auto config = MakeConfig(fault);
    AccOffloadAclSharedPool pool;
    const auto exchange = Exchange(fd);
    const int ret = pool.Initialize(config, exchange, {SetDevice, Register, DevicePointer, Unregister});
    if (fault == 0) {
        Require(ret == 0, "shared pool initialization failed");
        VerifySharedBytes(pool, exchange, collision);
    } else {
        Require(ret != 0 && pool.Base() == nullptr, "peer failure did not roll back all ranks");
    }
    Require(registerCount == unregisterCount, "ACL registration leaked");
    Require(pool.Close(), "idempotent close failed");
    if (occupied != MAP_FAILED) {
        Require(munmap(occupied, POOL_BYTES) == 0, "collision range cleanup failed");
    }
    close(fd);
}

void RunCase(int fault, bool collision)
{
    int sockets[RANKS];
    Require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair failed");
    const pid_t pid = fork();
    Require(pid >= 0, "fork failed");
    rankId = pid == 0 ? 1U : 0U;
    close(sockets[1U - rankId]);
    try {
        Worker(sockets[rankId], fault, collision);
    } catch (const std::exception &error) {
        std::cerr << "rank " << rankId << ": " << error.what() << std::endl;
        if (pid == 0) {
            _exit(1);
        }
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        throw;
    }
    if (pid == 0) {
        _exit(0);
    }
    int status = 0;
    Require(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0, "peer process failed");
    alarm(0U);
}
} // namespace

int main()
{
    try {
        RunCase(0, false);
        RunCase(0, true);
        for (int fault = 1; fault <= 5; ++fault) {
            RunCase(fault, false);
        }
        std::cout << "PASS: 7 CPU_ONLY shared-backing / mocked-ACL cases (no NPU validation)" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
