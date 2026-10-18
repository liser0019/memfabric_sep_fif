#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# http://license.coscl.org.cn/MulanPSL2
set -euo pipefail
test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${test_dir}/../.." && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/mf_acl_shared_test.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT
"${CXX:-g++}" -std=c++17 -Wall -Wextra -Werror -g -pthread \
    -I"${repo_dir}/src/acc_offload/csrc" \
    -I"${repo_dir}/src/acc_offload/csrc/common" \
    -I"${repo_dir}/src/acc_offload/include/host" \
    -I"${repo_dir}/src/util/csrc" \
    "${repo_dir}/src/acc_offload/csrc/acc_offload_acl_shared_pool.cpp" \
    "${test_dir}/test_acl_shared_pool.cpp" -o "${build_dir}/test_acl_shared_pool"
"${build_dir}/test_acl_shared_pool"
