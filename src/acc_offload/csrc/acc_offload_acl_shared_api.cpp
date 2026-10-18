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
#include "dl_acl_api.h"

namespace ock {
namespace offload {
int32_t AccOffloadAclSharedPool::RuntimeApi(uint32_t deviceId, AclApi &api)
{
    using ock::mf::DlAclApi;
    if (DlAclApi::GetAscendSocType() != ock::mf::AscendSocType::ASCEND_950) {
        BM_LOG_ERROR("ACL shared host pool requires Ascend950, deviceId: " << deviceId);
        return -1;
    }
    api = {[](int32_t device) { return DlAclApi::AclrtSetDevice(device); }, &DlAclApi::AclrtHostRegisterV2,
           &DlAclApi::AclrtHostGetDevicePointer, &DlAclApi::AclrtHostUnregister};
    return 0;
}
} // namespace offload
} // namespace ock
