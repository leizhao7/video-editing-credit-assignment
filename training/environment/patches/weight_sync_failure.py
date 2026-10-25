"""Coordinate vLLM weight-sync failures across all training ranks.

The upstream NCCL sender lets rank 0 perform the HTTP start/update/finish
handshake while peer ranks immediately enter FSDP weight extraction and then
an unconditional barrier.  If rank 0's handshake raises, peers wait until the
large NCCL timeout and obscure the original error.  Add a preflight broadcast
and a post-transfer status all-reduce so every rank exits with a useful error.
"""
from __future__ import annotations

import py_compile
import sys


TARGET = (
    "/opt/SkyRL/skyrl/backends/skyrl_train/weight_sync/"
    "broadcast_strategy.py"
)
MARKER = "S12_WEIGHT_SYNC_FAILURE_PROPAGATION"

OLD = '''        if torch.distributed.get_rank() == 0:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )

            await self._inference_client.start_weight_update(is_checkpoint_format=True)

            update_info = {**weight_metadata, "packed": True}
            update_task = asyncio.create_task(self._inference_client.update_weights_nccl(update_info))

            # Run in thread so the HTTP update_task can progress concurrently
            await asyncio.to_thread(
                NCCLWeightTransferEngine.trainer_send_weights,
                iterator=weight_iterator(),
                trainer_args={"group": self._model_update_group, "packed": True},
            )
            await update_task

            await self._inference_client.finish_weight_update()
        else:
            # Non-rank-0 still needs to participate in the all-gather
            for _ in weight_iterator():
                pass

        torch.distributed.barrier()
'''

NEW = '''        # S12_WEIGHT_SYNC_FAILURE_PROPAGATION: use the training process
        # group as a small control plane around the separate vLLM NCCL group.
        rank = torch.distributed.get_rank()
        control_device = torch.device("cuda", torch.cuda.current_device())

        start_error = None
        if rank == 0:
            try:
                await self._inference_client.start_weight_update(is_checkpoint_format=True)
            except BaseException as exc:
                start_error = exc
        start_ok = torch.tensor(
            [0 if start_error is not None else 1],
            dtype=torch.int32,
            device=control_device,
        )
        torch.distributed.broadcast(start_ok, src=0)
        if int(start_ok.item()) == 0:
            if start_error is not None:
                raise start_error
            raise RuntimeError("vLLM weight-sync start failed on training rank 0")

        transfer_error = None
        update_task = None
        rank0_iterator = None
        try:
            if rank == 0:
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLWeightTransferEngine,
                )

                update_info = {**weight_metadata, "packed": True}
                update_task = asyncio.create_task(
                    self._inference_client.update_weights_nccl(update_info)
                )
                rank0_iterator = iter(weight_iterator())

                # Run in thread so the HTTP update_task can progress concurrently.
                await asyncio.to_thread(
                    NCCLWeightTransferEngine.trainer_send_weights,
                    iterator=rank0_iterator,
                    trainer_args={"group": self._model_update_group, "packed": True},
                )
                await update_task
                await self._inference_client.finish_weight_update()
            else:
                # Non-rank-0 still needs to participate in every FSDP all-gather.
                for _ in weight_iterator():
                    pass
        except BaseException as exc:
            transfer_error = exc
            # If the transport aborted after consuming only part of rank 0's
            # iterator, finish the remaining FSDP collectives so peer ranks can
            # reach the coordinated status reduction instead of hanging.
            if rank == 0 and rank0_iterator is not None:
                try:
                    for _ in rank0_iterator:
                        pass
                except BaseException:
                    pass
        finally:
            if (
                rank == 0
                and transfer_error is not None
                and update_task is not None
                and not update_task.done()
            ):
                update_task.cancel()
                await asyncio.gather(update_task, return_exceptions=True)

        transfer_ok = torch.tensor(
            [0 if transfer_error is not None else 1],
            dtype=torch.int32,
            device=control_device,
        )
        torch.distributed.all_reduce(
            transfer_ok, op=torch.distributed.ReduceOp.MIN
        )
        if int(transfer_ok.item()) == 0:
            if transfer_error is not None:
                raise transfer_error
            raise RuntimeError("vLLM weight sync failed on a peer training rank")
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[weight-sync-failure] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[weight-sync-failure] already patched", flush=True)
        return 0
    count = source.count(OLD)
    if count != 1:
        print(f"[weight-sync-failure] anchor matched {count}; refusing", flush=True)
        return 1
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source.replace(OLD, NEW))
    py_compile.compile(TARGET, doraise=True)
    print(f"[weight-sync-failure] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
