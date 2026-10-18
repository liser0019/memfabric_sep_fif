/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 */
#include "acc_offload_acl_shared_pool.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>
#include "acc_offload_define.h"

namespace ock {
namespace offload {
namespace {
constexpr uint32_t ACL_SHARED_REGISTER_FLAGS = 0x10000000U | 0x2U; // PINNED | MAPPED
constexpr uint64_t COMMON_ADDRESS_START = 0x200000000000ULL;
constexpr uint64_t COMMON_ADDRESS_STRIDE = 1ULL << 40U;
constexpr uint32_t COMMON_ADDRESS_ATTEMPTS = 16U;
constexpr uint32_t MAX_SHARED_RANKS = 1024U;
} // namespace

AccOffloadAclSharedPool::~AccOffloadAclSharedPool()
{
    (void)Close();
}

int32_t AccOffloadAclSharedPool::PrepareBacking(PeerInfo &info)
{
    utsname system{};
    unsigned int major = 0U;
    unsigned int minor = 0U;
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (uname(&system) != 0 || std::sscanf(system.release, "%u.%u", &major, &minor) != 2 || major < 5U ||
        (major == 5U && minor <= 10U) || pageSize <= 0L || size_ == 0U ||
        size_ % static_cast<uint64_t>(pageSize) != 0U || size_ > COMMON_ADDRESS_STRIDE) {
        OFFLOAD_LOG_ERROR("ACL shared pool requires kernel >5.10, page-aligned size <=1TiB, rankId: "
                          << rank_ << ", kernel: " << system.release << ", size: " << size_);
        return OFFLOAD_ERROR;
    }
    std::ifstream bootFile("/proc/sys/kernel/random/boot_id");
    std::string bootId;
    if (!(bootFile >> bootId) || bootId.size() >= sizeof(info.bootId)) {
        OFFLOAD_LOG_ERROR("read boot_id failed, rankId: " << rank_);
        return OFFLOAD_ERROR;
    }
    std::memcpy(info.bootId, bootId.c_str(), bootId.size() + 1U);
    if ((rank_ == 0U && info.allocSize != size_) || (rank_ != 0U && info.allocSize != 0U)) {
        OFFLOAD_LOG_ERROR("ACL shared pool requires owner rank0 alloc_size=reserve_size and peer alloc_size=0, "
                          "rankId: "
                          << rank_ << ", size: " << size_ << ", allocSize: " << info.allocSize);
        return OFFLOAD_ERROR;
    }
    return rank_ == 0U ? CreateBacking(info) : OFFLOAD_OK;
}

int32_t AccOffloadAclSharedPool::CreateBacking(PeerInfo &info)
{
    char name[] = "/dev/shm/mf_acl_pool_XXXXXX";
    int fd = mkstemp(name);
    if (fd < 0) {
        OFFLOAD_LOG_ERROR("mkstemp failed, rankId: " << rank_ << ", errno: " << errno);
        return OFFLOAD_ERROR;
    }
    path_ = name;
    // Allocate backing before registration: a sparse tmpfs file can otherwise
    // SIGBUS on first touch if /dev/shm cannot hold the requested pool.
    const int ret = posix_fallocate(fd, 0, static_cast<off_t>(size_));
    struct stat st {};
    const bool valid = ret == 0 && fstat(fd, &st) == 0;
    close(fd);
    if (!valid) {
        OFFLOAD_LOG_ERROR("prepare shared backing failed, path: "
                          << path_ << ", size: " << size_ << ", posix_fallocate ret: " << ret << ", errno: " << errno);
        return OFFLOAD_ERROR;
    }
    info.inode = static_cast<uint64_t>(st.st_ino);
    info.device = static_cast<uint64_t>(st.st_dev);
    std::memcpy(info.path, name, sizeof(name));
    return OFFLOAD_OK;
}

int32_t AccOffloadAclSharedPool::CheckPeers(const PeerInfo *peers) const
{
    for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
        if (peers[rank].result != OFFLOAD_OK || peers[rank].rank != rank || peers[rank].size != size_ ||
            std::memcmp(peers[rank].bootId, peers[0].bootId, sizeof(peers[0].bootId)) != 0 ||
            peers[rank].allocSize != (rank == 0U ? size_ : 0U)) {
            OFFLOAD_LOG_ERROR("shared pool peer/config mismatch, rankId: " << rank_ << ", peerRank: " << rank
                                                                           << ", result: " << peers[rank].result
                                                                           << ", peer size: " << peers[rank].size);
            return OFFLOAD_ERROR;
        }
    }
    return OFFLOAD_OK;
}

int32_t AccOffloadAclSharedPool::Agree(int32_t result, const Exchange &exchange, const char *stage) const
{
    std::vector<int32_t> results(worldSize_);
    int32_t ret = exchange(reinterpret_cast<const char *>(&result), sizeof(result),
                           reinterpret_cast<char *>(results.data()), sizeof(result) * worldSize_);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("ACL shared pool exchange failed, stage: " << stage << ", rankId: " << rank_
                                                                     << ", ret: " << ret);
        return OFFLOAD_ERROR;
    }
    for (uint32_t rank = 0U; rank < worldSize_; ++rank) {
        if (results[rank] != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("ACL shared pool stage failed, stage: " << stage << ", rankId: " << rank_
                                                                      << ", peerRank: " << rank
                                                                      << ", ret: " << results[rank]);
            return OFFLOAD_ERROR;
        }
    }
    return OFFLOAD_OK;
}

int32_t AccOffloadAclSharedPool::ReserveCommonAddress(const Exchange &exchange)
{
    for (uint32_t attempt = 0U; attempt < COMMON_ADDRESS_ATTEMPTS; ++attempt) {
        auto *candidate = reinterpret_cast<void *>(COMMON_ADDRESS_START + COMMON_ADDRESS_STRIDE * attempt);
        void *mapping =
            mmap(candidate, size_, PROT_NONE, MAP_FIXED_NOREPLACE | MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
        const bool reserved = mapping != MAP_FAILED && mapping == candidate;
        if (!reserved && mapping != MAP_FAILED) {
            munmap(mapping, size_);
        }
        if (Agree(reserved ? OFFLOAD_OK : OFFLOAD_ERROR, exchange, "reserve common HVA") == OFFLOAD_OK) {
            base_ = mapping;
            return OFFLOAD_OK;
        }
        if (reserved) {
            munmap(mapping, size_);
        }
    }
    OFFLOAD_LOG_ERROR("no common free HVA range, rankId: " << rank_ << ", size: " << size_);
    return OFFLOAD_ERROR;
}

int32_t AccOffloadAclSharedPool::MapBacking(const PeerInfo &owner)
{
    if (owner.path[sizeof(owner.path) - 1U] != '\0') {
        OFFLOAD_LOG_ERROR("invalid shared backing path, rankId: " << rank_);
        return OFFLOAD_ERROR;
    }
    int fd = open(owner.path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    struct stat st {};
    if (fd < 0 || fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != geteuid() ||
        static_cast<uint64_t>(st.st_size) != size_ || static_cast<uint64_t>(st.st_ino) != owner.inode ||
        static_cast<uint64_t>(st.st_dev) != owner.device) {
        OFFLOAD_LOG_ERROR("open/validate shared backing failed, path: " << owner.path << ", rankId: " << rank_
                                                                        << ", size: " << size_ << ", errno: " << errno);
        if (fd >= 0) {
            close(fd);
        }
        return OFFLOAD_ERROR;
    }
    void *mapped = mmap(base_, size_, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, 0);
    const int mapErrno = errno;
    close(fd);
    if (mapped == MAP_FAILED || mapped != base_) {
        OFFLOAD_LOG_ERROR("map shared backing failed, rankId: " << rank_ << ", size: " << size_
                                                                << ", errno: " << mapErrno);
        return OFFLOAD_ERROR;
    }
    return RegisterMapping();
}

int32_t AccOffloadAclSharedPool::RegisterMapping()
{
    auto ret = api_.setDevice(deviceId_);
    if (ret != 0) {
        OFFLOAD_LOG_ERROR("ACL set device before shared registration failed, rankId: "
                          << rank_ << ", deviceId: " << deviceId_ << ", ret: " << ret);
        return OFFLOAD_ERROR;
    }
    ret = api_.registerHost(base_, size_, ACL_SHARED_REGISTER_FLAGS);
    if (ret != 0) {
        OFFLOAD_LOG_ERROR("ACL register shared host failed, rankId: " << rank_ << ", deviceId: " << deviceId_
                                                                      << ", size: " << size_ << ", flags: "
                                                                      << ACL_SHARED_REGISTER_FLAGS << ", ret: " << ret);
        return OFFLOAD_ERROR;
    }
    registered_ = true;
    void *devicePtr = nullptr;
    ret = api_.getDevicePointer(base_, &devicePtr, 0U);
    if (ret != 0 || devicePtr == nullptr) {
        OFFLOAD_LOG_ERROR("ACL get shared host DVA failed, rankId: " << rank_ << ", deviceId: " << deviceId_
                                                                     << ", size: " << size_ << ", ret: " << ret);
        return OFFLOAD_ERROR;
    }
    dva_ = reinterpret_cast<uint64_t>(devicePtr);
    OFFLOAD_LOG_INFO("ACL shared host mapped, rankId: " << rank_ << ", deviceId: " << deviceId_ << ", size: " << size_
                                                        << ", hvaEqualsDva: "
                                                        << (reinterpret_cast<uint64_t>(base_) == dva_));
    return OFFLOAD_OK;
}

int32_t AccOffloadAclSharedPool::Initialize(const offload_config_t &config, const Exchange &exchange, AclApi api)
{
    if (base_ != nullptr || worldSize_ != 0U || config.worldSize == 0U || config.worldSize > MAX_SHARED_RANKS ||
        config.rankId >= config.worldSize || !exchange || api.setDevice == nullptr || api.registerHost == nullptr ||
        api.getDevicePointer == nullptr || api.unregisterHost == nullptr) {
        OFFLOAD_LOG_ERROR("invalid ACL shared initialization, rankId: " << config.rankId
                                                                        << ", worldSize: " << config.worldSize);
        return OFFLOAD_ERROR;
    }
    api_ = api;
    rank_ = config.rankId;
    worldSize_ = config.worldSize;
    deviceId_ = static_cast<int32_t>(config.deviceId);
    size_ = config.reserveSize;
    PeerInfo local{};
    local.size = size_;
    local.allocSize = config.allocSize;
    local.rank = rank_;
    local.result = PrepareBacking(local);
    std::vector<PeerInfo> peers(worldSize_);
    int32_t ret = exchange(reinterpret_cast<const char *>(&local), sizeof(local),
                           reinterpret_cast<char *>(peers.data()), sizeof(local) * worldSize_);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("shared backing exchange failed, rankId: " << rank_ << ", ret: " << ret);
    } else {
        ret = CheckPeers(peers.data());
    }
    if (ret == OFFLOAD_OK) {
        ret = ReserveCommonAddress(exchange);
    }
    if (ret == OFFLOAD_OK) {
        ret = Agree(MapBacking(peers[0]), exchange, "ACL mapping");
    }
    UnlinkBacking(); // All peers finished opening their own mapping before this point.
    ready_ = ret == OFFLOAD_OK;
    if (!ready_) {
        (void)Close();
    }
    return ret;
}

int32_t AccOffloadAclSharedPool::GetDva(uint64_t hostPtr, uint64_t *dvaPtr) const
{
    const uint64_t base = reinterpret_cast<uint64_t>(base_);
    if (!ready_ || dvaPtr == nullptr || hostPtr < base || hostPtr - base >= size_ ||
        dva_ > std::numeric_limits<uint64_t>::max() - (hostPtr - base)) {
        OFFLOAD_LOG_ERROR("shared host DVA lookup out of range/not ready, rankId: " << rank_ << ", size: " << size_);
        return OFFLOAD_ERROR;
    }
    *dvaPtr = dva_ + (hostPtr - base);
    return OFFLOAD_OK;
}

void AccOffloadAclSharedPool::UnlinkBacking()
{
    if (!path_.empty()) {
        if (unlink(path_.c_str()) != 0 && errno != ENOENT) {
            OFFLOAD_LOG_ERROR("unlink shared backing failed, path: " << path_ << ", errno: " << errno);
        } else {
            path_.clear();
        }
    }
}

bool AccOffloadAclSharedPool::Close()
{
    ready_ = false;
    if (registered_) {
        int32_t ret = api_.setDevice(deviceId_);
        if (ret == 0) {
            ret = api_.unregisterHost(base_);
        }
        if (ret != 0) {
            OFFLOAD_LOG_ERROR("ACL unregister failed; retain backing to avoid unmapping registered pages, rankId: "
                              << rank_ << ", deviceId: " << deviceId_ << ", ret: " << ret);
            return false;
        }
        registered_ = false;
    }
    if (base_ != nullptr && munmap(base_, size_) != 0) {
        OFFLOAD_LOG_ERROR("unmap shared pool failed, rankId: " << rank_ << ", size: " << size_ << ", errno: " << errno);
        return false;
    }
    base_ = nullptr;
    dva_ = 0U;
    size_ = 0U;
    worldSize_ = 0U;
    UnlinkBacking();
    return true;
}

} // namespace offload
} // namespace ock
