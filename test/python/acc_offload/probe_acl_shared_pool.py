#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# http://license.coscl.org.cn/MulanPSL2
"""A5 shared Host backing: real sparse_copy round trips, optional real URMA pull.

This is a standalone correctness probe, not a model or performance benchmark.
The parent never initializes ACL. Every child owns its device context.
"""

import argparse
import ctypes
import hashlib
import json
import multiprocessing
import queue
import time
import traceback

POOL_BYTES = 1 << 30
PAYLOAD_BYTES = 256 << 10


def pattern(size, epoch):
    return bytes(((index * 13 + epoch * 31 + (index >> 8) + (index >> 16) * 7) & 255) for index in range(size))


def verify(actual, expected, phase, rank, results):
    if actual != expected:
        mismatch = next(index for index, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise RuntimeError(f"{phase}: rank={rank} first mismatch byte={mismatch}")
    results.put({"phase": phase, "rank": rank, "sha256": hashlib.sha256(actual).hexdigest(), "ok": True})


def make_engine(mf, config, device, role):
    mf.set_conf_store_tls(False, "")
    engine = mf.TransferEngine()
    ret = engine.initialize(
        config.urma_store,
        config.bind_ip,
        role,
        device,
        store_server_role="Prefill",
        data_op_type=mf.TransDataOpType.DEVICE_URMA,
    )
    if ret != 0:
        raise RuntimeError(f"URMA initialize failed: role={role} device={device} ret={ret}")
    return engine


def close_engine(engine, registered):
    for name, arguments in [("unregister_memory", (registered,)), ("destroy", ()), ("unInitialize", ())]:
        ret = getattr(engine, name)(*arguments)
        if ret not in (None, 0):
            raise RuntimeError(f"{name} failed: ret={ret}")


def copy_sparse(offload, torch, device, source, target, size):
    src = torch.tensor([source], dtype=torch.int64, device=device)
    dst = torch.tensor([target], dtype=torch.int64, device=device)
    lengths = torch.tensor([size], dtype=torch.int32, device=device)
    count = torch.tensor([1], dtype=torch.int32, device=device)
    ret = offload.sparse_copy(src, dst, lengths, count, device)
    if ret not in (None, 0):
        raise RuntimeError(f"sparse_copy failed: ret={ret}")
    torch.npu.synchronize()


def check_device_view(offload, torch, device, dva, expected, phase, rank, results):
    target = torch.empty(len(expected), dtype=torch.uint8, device=device)
    copy_sparse(offload, torch, device, dva, target.data_ptr(), len(expected))
    verify(bytes(target.cpu().tolist()), expected, phase, rank, results)


def check_local_writes(offload, torch, device, hva, dva, rank, ranks, barrier, results):
    first = pattern(PAYLOAD_BYTES, 1)
    if rank == 0:
        ctypes.memmove(hva, first, len(first))
    barrier.wait()
    check_device_view(offload, torch, device, dva, first, "CPU-to-all-AIV", rank, results)
    piece_size = PAYLOAD_BYTES // ranks
    start = rank * piece_size
    full = pattern(PAYLOAD_BYTES, 2)
    source = torch.tensor(list(full[start : start + piece_size]), dtype=torch.uint8, device=device)
    copy_sparse(offload, torch, device, source.data_ptr(), dva + start, piece_size)
    barrier.wait()
    verify(ctypes.string_at(hva, PAYLOAD_BYTES), full, "all-AIV-to-shared-CPU", rank, results)
    check_device_view(offload, torch, device, dva, full, "all-AIV-to-all-AIV", rank, results)
    barrier.wait()
    # Rewrite an already-read suffix, rather than only testing cold allocation.
    if rank == 0:
        ctypes.memmove(hva + piece_size, first[piece_size:], PAYLOAD_BYTES - piece_size)
    barrier.wait()
    rewritten = full[:piece_size] + first[piece_size:]
    check_device_view(offload, torch, device, dva, rewritten, "CPU-suffix-rewrite", rank, results)


def check_urma(mf, offload, torch, config, device, hva, dva, rank, ranks, barrier, results):
    engine = make_engine(mf, config, device.index, "Decode")
    registered = False
    try:
        ret = engine.register_memory(hva, PAYLOAD_BYTES)  # HVA, never the mapped DVA.
        if ret != 0:
            raise RuntimeError(f"URMA Host registration failed: rank={rank} ret={ret}")
        registered = True
        piece = PAYLOAD_BYTES // ranks
        start = rank * piece
        ret = engine.batch_transfer_sync_read(
            config.source_session, [hva + start], [config.source_address + start], [piece]
        )
        if ret != 0:
            raise RuntimeError(f"URMA pull failed: rank={rank} ret={ret}")
        barrier.wait()
        check_device_view(
            offload, torch, device, dva, pattern(PAYLOAD_BYTES, 3), "URMA-disjoint-pull-to-all-AIV", rank, results
        )
        barrier.wait()
    finally:
        if registered:
            close_engine(engine, hva)
        else:
            engine.destroy()
            engine.unInitialize()


def close_pool(offload, rank, closed, timeout):
    try:
        if offload is not None:
            if rank == 0:
                for event in closed[1:]:
                    if not event.wait(timeout):
                        raise TimeoutError("peer pool cleanup timed out; keep owner control store alive")
            offload.uninitialize()
    finally:
        closed[rank].set()


def decode_worker(config, devices, rank, shared_hva, barrier, closed, results):
    initialized = False
    try:
        import torch
        import torch_npu  # noqa: F401
        import memfabric_hybrid as mf

        device = torch.device(f"npu:{devices[rank]}")
        torch.npu.set_device(device)
        offload = mf.offload
        pool_config = offload.OffloadConfig()
        pool_config.device_id = device.index
        pool_config.reserve_size = POOL_BYTES
        pool_config.alloc_size = POOL_BYTES if rank == 0 else 0
        pool_config.world_size = len(devices)
        pool_config.rank_id = rank
        pool_config.scene = offload.Scene.SHARED
        pool_config.store_url = config.pool_store
        pool_config.flags = offload.OFFLOAD_FLAG_ACL_SHARED_HOST
        ret = offload.initialize(pool_config)
        if ret != 0:
            raise RuntimeError(f"offload.initialize failed: rank={rank} ret={ret}")
        initialized = True
        if rank == 0:
            shared_hva.value = offload.malloc(PAYLOAD_BYTES)
        barrier.wait()
        hva = shared_hva.value
        dva = offload.get_dva(hva)
        if not hva or not dva:
            raise RuntimeError(f"pool address lookup failed: rank={rank} hva={hva:#x} dva={dva:#x}")
        results.put(
            {"phase": "mapped", "rank": rank, "device": device.index, "hva": hex(hva), "dva": hex(dva), "ok": True}
        )
        check_local_writes(offload, torch, device, hva, dva, rank, len(devices), barrier, results)
        barrier.wait()
        if config.urma_source_device is not None:
            check_urma(mf, offload, torch, config, device, hva, dva, rank, len(devices), barrier, results)
        torch.npu.synchronize()
        barrier.wait()
    except BaseException as error:
        barrier.abort()
        results.put({"rank": rank, "ok": False, "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        close_pool(offload if initialized else None, rank, closed, config.timeout)


def source_worker(config, connection):
    engine = None
    address = 0
    try:
        import torch
        import torch_npu  # noqa: F401
        import memfabric_hybrid as mf

        torch.npu.set_device(config.urma_source_device)
        source = torch.tensor(
            list(pattern(PAYLOAD_BYTES, 3)), dtype=torch.uint8, device=f"npu:{config.urma_source_device}"
        )
        torch.npu.synchronize()
        engine = make_engine(mf, config, config.urma_source_device, "Prefill")
        ret = engine.register_memory(source.data_ptr(), PAYLOAD_BYTES)
        if ret != 0:
            raise RuntimeError(f"source HBM registration failed: ret={ret}")
        address = source.data_ptr()
        connection.send({"session": f"{config.bind_ip}:{engine.get_rpc_port()}", "address": address})
        if not connection.poll(config.timeout):
            raise TimeoutError("source waiting for Decode cleanup timed out")
        connection.recv()
    except BaseException as error:
        connection.send({"error": str(error)})
        raise
    finally:
        if engine is not None:
            if address:
                close_engine(engine, address)
            else:
                engine.destroy()
                engine.unInitialize()
        connection.close()


def wait_workers(processes, results, timeout):
    ok = True
    deadline = time.monotonic() + timeout
    while any(process.is_alive() for process in processes):
        if time.monotonic() > deadline:
            raise TimeoutError("Decode test exceeded timeout")
        try:
            message = results.get(timeout=0.2)
            print(json.dumps(message, ensure_ascii=False), flush=True)
            ok &= message["ok"]
        except queue.Empty:
            pass
    for process in processes:
        process.join()
        ok &= process.exitcode == 0
    return ok


def drain_results(results):
    ok = True
    while True:
        try:
            message = results.get_nowait()
            print(json.dumps(message, ensure_ascii=False), flush=True)
            ok &= message["ok"]
        except queue.Empty:
            break
    results.close()
    return ok


def run(config, devices):
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(len(devices), timeout=config.timeout)
    shared_hva = ctx.Value(ctypes.c_uint64, 0)
    closed = [ctx.Event() for _ in devices]
    results = ctx.Queue()
    processes = []
    source = None
    parent = None
    ok = True
    try:
        if config.urma_source_device is not None:
            parent, child = ctx.Pipe()
            source = ctx.Process(target=source_worker, args=(config, child))
            source.start()
            child.close()
            if not parent.poll(config.timeout):
                raise TimeoutError("URMA source initialization timed out")
            info = parent.recv()
            if "error" in info:
                raise RuntimeError(info["error"])
            config.source_session, config.source_address = info["session"], info["address"]
        for rank in range(len(devices)):
            process = ctx.Process(
                target=decode_worker, args=(config, devices, rank, shared_hva, barrier, closed, results)
            )
            process.start()
            processes.append(process)
        ok = wait_workers(processes, results, config.timeout)
    finally:
        barrier.abort()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        if source:
            if source.is_alive():
                parent.send("Decode finished")
                source.join(timeout=15)
            if source.is_alive():
                source.terminate()
                source.join(timeout=5)
                ok = False
            ok &= source.exitcode == 0
            parent.close()
        ok &= drain_results(results)
    if not ok:
        raise RuntimeError("shared pool test failed; see per-rank output")
    print("PASS: AIV shared pool" + (" + real URMA" if source else " (URMA not tested)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="Decode logical device IDs, e.g. 0,1 or 0,0")
    parser.add_argument("--pool-store", default="tcp://127.0.0.1:18500")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--urma-source-device", type=int)
    parser.add_argument("--bind-ip")
    parser.add_argument("--urma-store")
    config = parser.parse_args()
    devices = [int(part) for part in config.devices.split(",")]
    if len(devices) not in (2, 4, 8, 16, 32, 64) or any(device < 0 for device in devices) or config.timeout <= 0:
        parser.error("use 2/4/8/16/32/64 nonnegative Decode device IDs and a positive timeout")
    if config.urma_source_device is not None and (
        config.urma_source_device in devices
        or config.urma_source_device < 0
        or not config.bind_ip
        or not config.urma_store
        or config.urma_store == config.pool_store
    ):
        parser.error("URMA requires a separate source device, --bind-ip and a separate --urma-store URL")
    run(config, devices)


if __name__ == "__main__":
    main()
