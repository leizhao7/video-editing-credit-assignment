"""Explicit new-PRM-only sandbox admission; no Modal imports or remote calls.

This is ONE parent event loop's bound, not workspace quota reservation. The
caller supplies the SDK create coroutine and confirms successful termination.
Ambiguous creation outcomes consume a slot and halt further creates; they are
never automatically retried/released. A process restart cannot reconstruct this
in-memory ledger: recovery must reconcile its durable attempt journals first.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import random
import time
import uuid


class AdmissionError(RuntimeError):
    pass


class UncertainCreation(AdmissionError):
    pass


@dataclass(frozen=True)
class Policy:
    app_name: str
    owner: str
    interval_s: float = 2.0
    max_live: int = 192
    create_inflight: int = 1


_POLICY = None
_MANAGER = None
_LOOP = None
DEFAULT_POLICY = dict(version=1, interval_s=2.0, outer_requests_per_second=.5,
                      burst=1, create_inflight=1, max_live=192)


def configure(*, app_name, owner, policy=None):
    """Call only after the new run's explicit PRM plan has been validated."""
    global _POLICY
    if not isinstance(app_name, str) or not app_name or not isinstance(owner, str) or not owner:
        raise AdmissionError('Explicit nonempty new-PRM app and run owner are required')
    supplied = DEFAULT_POLICY if policy is None else policy
    if (type(supplied) is not dict or json.dumps(supplied, sort_keys=True) !=
            json.dumps(DEFAULT_POLICY, sort_keys=True)):
        raise AdmissionError('The immutable plan must use the exact reviewed sandbox admission policy')
    configured = Policy(app_name=app_name, owner=owner)
    if _POLICY is not None and _POLICY != configured:
        raise AdmissionError('Cannot repurpose PRM admission for another app/run')
    _POLICY = configured
    return {**DEFAULT_POLICY, 'app_name': app_name, 'owner': owner}


def get_admission(app):
    """Default off, and never applies to an unrelated App in this process."""
    if _POLICY is None or getattr(app, 'name', None) != _POLICY.app_name:
        return None
    global _MANAGER, _LOOP
    loop = asyncio.get_running_loop()
    if _MANAGER is None:
        _MANAGER, _LOOP = Admission(_POLICY), loop
    if _LOOP is not loop:
        raise AdmissionError('Another event loop cannot reset or bypass this PRM live ledger')
    if _MANAGER.policy != _POLICY:
        raise AdmissionError('Event loop already belongs to another PRM admission policy')
    return _MANAGER


def is_capacity_refusal(exc):
    """Only explicit server refusal, not arbitrary text mentioning a quota.

GRPC status 8 / RESOURCE_EXHAUSTED, HTTP 429, and Modal's corresponding typed
exception confirm refusal. An SDK transport timeout does not confirm rejection.
"""
    if type(exc).__name__ == 'ResourceExhaustedError':
        return True
    status = getattr(exc, 'status', None)
    if status == 429:
        return True
    return (getattr(status, 'name', None) == 'RESOURCE_EXHAUSTED'
            or str(status) in ('Status.RESOURCE_EXHAUSTED', 'RESOURCE_EXHAUSTED')
            or (type(exc).__name__ == 'GRPCError' and status == 8))


async def _discard_event(kind, payload):
    return None


class Admission:
    def __init__(self, policy, *, clock=None, sleep=None, jitter=None):
        if (policy.interval_s != 2.0 or policy.max_live != 192 or policy.create_inflight != 1):
            raise AdmissionError('Only the reviewed 0.5/s, single-create, 192-live policy is supported')
        self.policy = policy
        self.clock = clock or asyncio.get_running_loop().time
        self.sleep = sleep or asyncio.sleep
        self.jitter = jitter or random.random
        self.create_lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.ledger = {}
        self.next_start = 0.0
        self.cooldown_until = 0.0
        self.refusal_streak = 0
        self.rate_retries = 0
        self.halt_reason = None

    def bind(self, attempt_id, event=None):
        if not isinstance(attempt_id, str) or not attempt_id:
            raise AdmissionError('A trajectory attempt identity is required')
        return AttemptAdmission(self, attempt_id, event or _discard_event)

    def counts(self):
        counts = {key: 0 for key in ('creating', 'running', 'retained', 'unknown')}
        for row in self.ledger.values():
            counts[row['state']] += 1
        return {**counts, 'occupied': len(self.ledger), 'max_live': self.policy.max_live}

    def _check_open(self):
        if self.halt_reason:
            raise AdmissionError('New PRM creation halted: ' + self.halt_reason)

    def _halt(self, reason):
        self.halt_reason = reason
        self.changed.set()


class AttemptAdmission:
    def __init__(self, shared, attempt_id, event):
        self.shared, self.attempt_id, self.event = shared, attempt_id, event
        self.last_stats = {}
        self.requests = 0

    def snapshot(self):
        owner = self.shared
        return dict(owner=owner.policy.owner, app_name=owner.policy.app_name,
                    live_counts=owner.counts(), halt_reason=owner.halt_reason,
                    owned=[dict(row) for row in owner.ledger.values()
                           if row['attempt_id'] == self.attempt_id], last_create=dict(self.last_stats))

    async def _emit(self, kind, **extra):
        owner = self.shared
        await self.event(kind, dict(time=time.time(), attempt_id=self.attempt_id,
            app_name=owner.policy.app_name, owner=owner.policy.owner,
            live_counts=owner.counts(), **extra))

    async def create(self, factory):
        owner = self.shared
        queued_at = owner.clock()
        self.requests += 1
        ticket = None
        invoked = False
        started_at = None
        # Includes the complete SDK await (and its otherwise invisible retries).
        async with owner.create_lock:
            owner._check_open()
            while len(owner.ledger) >= owner.policy.max_live:
                owner.changed.clear()
                await self._emit('prm_sandbox_admission_full', admission_wait_s=owner.clock()-queued_at)
                if len(owner.ledger) < owner.policy.max_live:
                    continue
                await owner.changed.wait()
                owner._check_open()
            ticket = uuid.uuid4().hex
            row = dict(ticket=ticket, attempt_id=self.attempt_id, state='creating', sandbox_id=None)
            owner.ledger[ticket] = row
            try:
                while True:
                    owner._check_open()
                    delay = max(owner.next_start, owner.cooldown_until) - owner.clock()
                    if delay <= 0:
                        break
                    await owner.sleep(delay)
                self.last_stats = dict(admission_wait_s=owner.clock()-queued_at,
                    create_latency_s=None, outer_requests_per_second=.5, create_inflight=1,
                    request_number=self.requests, shared_rate_retries=owner.rate_retries,
                    shared_cooldown_remaining_s=max(0., owner.cooldown_until-owner.clock()), ticket=ticket)
                await self._emit('prm_sandbox_create_start', **self.last_stats)
                # Journal I/O can be slow. Space the ACTUAL SDK invocations,
                # not timestamps taken before an awaited filesystem operation.
                started_at = owner.clock()
                owner.next_start = started_at + owner.policy.interval_s
                self.last_stats['admission_wait_s'] = started_at-queued_at
                invoked = True
                sandbox = await factory()
                sid = getattr(sandbox, 'object_id', None)
                if not isinstance(sid, str) or not sid:
                    raise UncertainCreation('SDK create returned without a sandbox identity')
                if any(x['sandbox_id'] == sid for key, x in owner.ledger.items() if key != ticket):
                    raise UncertainCreation('SDK returned an already occupied sandbox identity')
                row.update(state='running', sandbox_id=sid)
                owner.refusal_streak = 0
                self.last_stats['create_latency_s'] = owner.clock()-started_at
                await self._emit('prm_sandbox_created', sandbox_id=sid, **self.last_stats)
                return sandbox
            except BaseException as exc:
                if started_at is not None and self.last_stats.get('create_latency_s') is None:
                    self.last_stats['create_latency_s'] = owner.clock()-started_at
                if not invoked:
                    # No SDK call happened: cancellation/persistence failure cannot allocate.
                    owner.ledger.pop(ticket, None)
                    owner.changed.set()
                    raise
                if is_capacity_refusal(exc) and row['sandbox_id'] is None:
                    owner.ledger.pop(ticket, None)
                    owner.changed.set()
                    owner.refusal_streak += 1
                    owner.rate_retries += 1
                    base = min(60., 5. * 2 ** min(owner.refusal_streak-1, 4))
                    delay = min(60., base * (1. + .2 * max(0., min(1., self.shared.jitter()))))
                    owner.cooldown_until = max(owner.cooldown_until, owner.clock()+delay)
                    await self._emit('prm_sandbox_shared_cooldown',
                        admission_wait_s=owner.clock()-queued_at,
                        create_latency_s=self.last_stats.get('create_latency_s'),
                        error_type=type(exc).__name__, error=str(exc)[:500],
                        rate_retry=owner.rate_retries, refusal_streak=owner.refusal_streak,
                        cooldown_s=delay, cooldown_until_monotonic=owner.cooldown_until)
                    raise
                # The response may have been lost after allocation. Never make a
                # replacement and silently forget a potentially live Sandbox.
                row['state'] = 'retained' if row['sandbox_id'] else 'unknown'
                row['reason'] = 'create_outcome_or_durability_unconfirmed'
                owner._halt(row['reason'] + '; ticket=' + ticket)
                try:
                    await self._emit('prm_sandbox_create_uncertain', error_type=type(exc).__name__,
                        error=str(exc)[:500], ticket=ticket, sandbox_id=row['sandbox_id'],
                        admission_wait_s=owner.clock()-queued_at,
                        create_latency_s=self.last_stats.get('create_latency_s'))
                except BaseException:
                    # The ledger stays occupied even when its journal is unavailable.
                    pass
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise UncertainCreation('New PRM halted; reconcile ticket ' + ticket + ' before another create') from exc

    def _owned(self, sandbox_id):
        matches = [(key, row) for key, row in self.shared.ledger.items()
                   if row['sandbox_id'] == sandbox_id and row['attempt_id'] == self.attempt_id]
        if len(matches) != 1:
            raise AdmissionError('Cannot release/retain an unowned or already released sandbox')
        return matches[0]

    async def retained(self, sandbox_id, reason):
        _, row = self._owned(sandbox_id)
        row.update(state='retained', reason=str(reason)[:300])
        await self._emit('prm_sandbox_retained', sandbox_id=sandbox_id, reason=row['reason'])

    async def released(self, sandbox_id):
        """Caller must await successful SDK terminate BEFORE calling this."""
        key, row = self._owned(sandbox_id)
        # Persist the release confirmation before granting another live slot.
        await self._emit('prm_sandbox_terminated', sandbox_id=sandbox_id,
                         previous_state=row['state'], release_confirmed=True)
        self.shared.ledger.pop(key)
        self.shared.changed.set()
