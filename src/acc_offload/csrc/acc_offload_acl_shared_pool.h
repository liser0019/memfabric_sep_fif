/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 */
#ifndef MEMFABRIC_ACC_OFFLOAD_ACL_SHARED_POOL_H
#define MEMFABRIC_ACC_OFFLOAD_ACL_SHARED_POOL_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include "acc_offload.h"

namespace ock {
namespace offload {

// Each process registers the same backing in its own device context. No ACL
// mapping, device address, allocator state or context is exported to peers.
class AccOffloadAclSharedPool {
public:
    using Exchange = std::function<int32_t(const char *, uint32_t, char *, uint32_t)>;
    struct AclApi {
        int32_t (*setDevice)(int32_t);
        int32_t (*registerHost)(void *, uint64_t, uint32_t);
        int32_t (*getDevicePointer)(void *, void **, uint32_t);
        int32_t (*unregisterHost)(void *);
    };

    static int32_t RuntimeApi(uint32_t deviceId, AclApi &api);
    ~AccOffloadAclSharedPool();
    AccOffloadAclSharedPool() = default;
    AccOffloadAclSharedPool(const AccOffloadAclSharedPool &) = delete;
    AccOffloadAclSharedPool &operator=(const AccOffloadAclSharedPool &) = delete;
    int32_t Initialize(const offload_config_t &config, const Exchange &exchange, AclApi api);
    bool Close();
    int32_t GetDva(uint64_t hostPtr, uint64_t *dvaPtr) const;
    void *Base() const
    {
        return base_;
    }
    uint64_t Size() const
    {
        return size_;
    }

private:
    struct PeerInfo {
        uint64_t size{};
        uint64_t allocSize{};
        uint64_t inode{};
        uint64_t device{};
        uint32_t rank{};
        int32_t result{};
        char path[96]{};
        char bootId[40]{};
    };
    int32_t PrepareBacking(PeerInfo &info);
    int32_t CreateBacking(PeerInfo &info);
    int32_t CheckPeers(const PeerInfo *peers) const;
    int32_t ReserveCommonAddress(const Exchange &exchange);
    int32_t MapBacking(const PeerInfo &owner);
    int32_t RegisterMapping();
    int32_t Agree(int32_t result, const Exchange &exchange, const char *stage) const;
    void UnlinkBacking();

    AclApi api_{};
    void *base_{};
    uint64_t size_{};
    uint64_t dva_{};
    uint32_t rank_{};
    uint32_t worldSize_{};
    int32_t deviceId_{};
    bool registered_{};
    bool ready_{};
    std::string path_;
};

} // namespace offload
} // namespace ock
#endif
