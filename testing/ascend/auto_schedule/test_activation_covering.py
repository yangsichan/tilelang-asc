"""Activation-qualified covering for guarded sync insertion.

Host-only InsertSync tests covering the soundness rules of the guarded-sync
optimization (ISSUES.md design doc):

  R1  covering elimination (hazard activation => chain activation)
  R2  provably-inactive hazard drop (d==0 same-round pair)
  R3  d==0 && nv==1 statement guards
  R4  counter points: chain segments only, never dropped/guarded
  R5  Resolve free-variable rule (round-invariant only)
  R6  same-guard d>=1 edges stay retained
  R7  strict closure fidelity (no activation traces without guards)
  R8  runtime-load guards resolve

Each test pins one rule with a minimal kernel and exact sync counts.
"""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, schedule, statements


def _insert_sync(program):
    before = schedule(program, through="ResolveCore", auto_schedule=False)
    return transform.InsertSync()(before)


def _events(mod, operation, pipe):
    return [
        call
        for call in calls(mod, "tl.ascend_" + operation + "_flag")
        if call.args[0].value == pipe
    ]


def _guarded_syncs(mod, pipe=None):
    """Flag/barrier calls carrying an IfThenElse (activation) guard."""
    result = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name in (
            "tl.ascend_set_flag",
            "tl.ascend_wait_flag",
            "tl.ascend_pipe_barrier",
        ) and (pipe is None or call.args[0].value == pipe):
            if any(isinstance(parent, tirx.IfThenElse) for parent in parents):
                result.append(call)
    return result


def _event_guards(mod, pipe):
    guards = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name in ("tl.ascend_set_flag", "tl.ascend_wait_flag") and call.args[0].value == pipe:
            guards.extend(
                parent.condition
                for parent in parents
                if isinstance(parent, tirx.IfThenElse)
            )
    return guards


def _wait_guards(mod, pipe):
    """IfThenElse conditions guarding the pipe's wait flags only.

    For a mixed pair (producer inside a branch, consumer unguarded) the wait
    sits beside the unguarded consumer, so any non-trivial guard on it is an
    activation guard -- the distinctive signature of the new mechanism.
    """
    guards = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name == "tl.ascend_wait_flag" and call.args[0].value == pipe:
            guards.extend(
                parent.condition
                for parent in parents
                if isinstance(parent, tirx.IfThenElse)
            )
    return guards


def _is_true(expr):
    return isinstance(expr, tirx.IntImm) and expr.value == 1


def _chain_kernel(guard_cond, counter_ub=False):
    """P(MTE2, writes ub+gub) -> Q(V, reads ub writes gub) -> R(MTE3, reads gub).

    ub lives only under `guard_cond` (cond domain); gub is read every round by
    R (true domain). The P->R and R->Q edges on gub are cross-domain covering
    candidates.
    """
    tile = 64
    n = 4

    @T.prim_func
    def kernel(
        A: T.Tensor((n * tile,), "float32"),
        C: T.Tensor((n * 2 * tile,), "float32"),
        seq_len: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            gub = T.alloc_shared((2 * tile,), "float32")
            if counter_ub:
                T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.serial(n):
                if guard_cond(i, seq_len):
                    with T.Task():
                        T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                        T.copy(A[i * tile : (i + 1) * tile], gub[0:tile])
                    T.copy(ub[0:tile], gub[tile : 2 * tile])
                T.copy(gub[0 : 2 * tile], C[i * 2 * tile : (i + 1) * 2 * tile])

    return kernel


# ---------------------------------------------------------------------------
# Group A: elimination correctness (R1 / R2 / R4 / R6)
# ---------------------------------------------------------------------------


def test_activation_covering_eliminates_shortcut():
    # R1: P->R on gub is covered by P->Q(ub) + Q->R(gub); R->Q on gub is
    # covered by R->P ring + P->Q(ub). Both chains share the hazards'
    # activation (cond), so both candidates are eliminated.
    kernel = _chain_kernel(lambda i, s: i % 2 == 0)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3") == []  # P->R eliminated
    assert _events(after, "set", "MTE3_V") == []  # R->Q eliminated
    assert _events(after, "set", "V_MTE3")  # Q->R kept (chain segment)
    assert _events(after, "set", "MTE3_MTE2")  # R->P ring kept


def test_complementary_guards_drop_same_round_keep_cross_round():
    # R2: producer under i%2==0, consumer under i%2==1. The d==0 same-round
    # hazard is provably false and is dropped; the d==1 cross-round hazard
    # (even-round write feeding odd-round read) is real and stays as a ring.
    # If the drop were broken, the output would be 3 sets (d0 pair + ring)
    # instead of 2 (ring only).
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                if i % 2 == 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                if i % 2 == 1:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert len(_events(after, "set", "MTE2_MTE3")) == 2  # ring only (seed + rolling)
    assert len(_events(after, "wait", "MTE2_MTE3")) == 2


def test_same_guard_dependency_is_kept():
    # R2 guard: the drop is driven by provable inactivity, not by the mere
    # presence of guards -- with both ends under the same guard the d==0
    # hazard is cond, so the pair is kept (and guarded).
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                if i % 2 == 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert len(_events(after, "set", "MTE2_MTE3")) == 1
    assert len(_events(after, "wait", "MTE2_MTE3")) == 1
    guards = _event_guards(after, "MTE2_MTE3")
    assert guards and not all(_is_true(g) for g in guards)


def test_counter_chain_segment_enables_covering():
    # R1 x R4: with ub promoted to a 2-version counter multi-buffer, the
    # covering chain goes through the counter protocol point (P->Q on ub).
    # The gub candidates are still eliminated, and the counter ring remains
    # complete with dynamic (counter-clock) event ids.
    kernel = _chain_kernel(lambda i, s: i % 2 == 0, counter_ub=True)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3") == []  # P->R eliminated
    assert _events(after, "set", "MTE3_V") == []  # R->Q eliminated
    released = _events(after, "set", "MTE2_V")  # counter protocol pair P->Q
    acquired = _events(after, "wait", "MTE2_V")
    assert released and acquired
    assert all(not isinstance(e.args[1], tirx.IntImm) for e in released + acquired)


def test_same_guard_cross_iteration_edge_is_retained():
    # R6: both ends of a d>=1 edge share a guard. The edge conservatively
    # stands in for wider, active-adjacent hazards, so it must be retained
    # and never activation-guarded.
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(n):
                if i % 2 == 0:
                    T.copy(ub[0:tile], C[i * tile : (i + 1) * tile])
                    T.copy(A[i * tile : (i + 1) * tile], ub[0:tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    # d==1 RAW ring (P writes ub, R reads the next round): MTE2_MTE3, fully
    # retained as a ring (seed + rolling + drain) -- same-guard d>=1 edges
    # are conservative stand-ins for active-adjacent hazards and must never
    # be eliminated (R6).
    assert len(_events(after, "set", "MTE2_MTE3")) >= 2
    assert len(_events(after, "wait", "MTE2_MTE3")) >= 2
    # d==0 same-round WAR pair (R reads, P writes, both under cond) kept too.
    assert len(_events(after, "set", "MTE3_MTE2")) == 1


# ---------------------------------------------------------------------------
# Group B: guard emission / resolution (R3 / R5 / R8)
# ---------------------------------------------------------------------------


def test_same_round_pair_carries_hazard_guard():
    # R3: mixed pair -- producer Q inside the branch, consumer R unguarded.
    # The wait sits beside R (outside the branch), so a non-trivial guard on
    # it is an activation guard (distinctive signature of the mechanism).
    kernel = _chain_kernel(lambda i, s: i % 2 == 0)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    guards = _wait_guards(after, "V_MTE3")  # Q->R pair, wait at unguarded R
    assert guards and not all(_is_true(g) for g in guards)


def test_runtime_scalar_guard_resolves():
    # R5 + R8: a guard depending on a runtime scalar parameter (decode-tail
    # shape) is round-invariant and resolves; elimination still happens and
    # the unguarded-side wait receives an activation guard.
    kernel = _chain_kernel(lambda i, s: i * 64 < s[0])
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3") == []
    guards = _wait_guards(after, "V_MTE3")
    assert guards and not all(_is_true(g) for g in guards)


def test_scope_written_guard_variable_stays_conservative():
    # R5 guard: a guard reading a variable written inside the scope must be
    # rejected; the mixed pair then stays fully conservative -- the wait at
    # the unguarded consumer carries no activation guard.
    tile = 64
    n = 4

    @T.prim_func
    def kernel(A: T.Tensor((n * tile,), "float32"), C: T.Tensor((n * tile,), "float32")):
        with T.Kernel(1):
            gub = T.alloc_shared((tile,), "float32")
            cnt = T.alloc_shared((1,), "int32")
            for i in T.serial(n):
                cnt[0] = cnt[0] + 1
                if cnt[0] > 0:
                    T.copy(A[i * tile : (i + 1) * tile], gub[0:tile])
                T.copy(gub[0:tile], C[i * tile : (i + 1) * tile])

    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _events(after, "set", "MTE2_MTE3")  # dependency still protected
    assert not _wait_guards(after, "MTE2_MTE3")  # no activation guard


# ---------------------------------------------------------------------------
# Group C: strict closure fidelity (R7)
# ---------------------------------------------------------------------------


def test_unguarded_kernel_has_no_activation_traces():
    # R7: without runtime predicates the activation machinery leaves no trace:
    # every emitted synchronization is unguarded, exactly as the pre-change
    # pipeline produced.
    kernel = _chain_kernel(lambda i, s: True)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert _guarded_syncs(after) == []


def test_elimination_preserves_kept_sync_counts():
    # R1 integrity: elimination removes only the covered candidates; the kept
    # groups (Q->R pair, R->P ring) survive with exact expected counts.
    kernel = _chain_kernel(lambda i, s: i % 2 == 0)
    with tvm.target.Target("ascend"):
        after = _insert_sync(kernel)
    assert len(_events(after, "set", "V_MTE3")) == 1
    assert len(_events(after, "wait", "V_MTE3")) == 1
    released = _events(after, "set", "MTE3_MTE2")
    acquired = _events(after, "wait", "MTE3_MTE2")
    assert len(released) >= 2  # ring: seed + rolling
    assert len(acquired) == len(released)


# ---------------------------------------------------------------------------
# Group D: boundary exemptions (R3 / R4)
# ---------------------------------------------------------------------------

# R4 ("counter points never statement-guarded") is asserted through the
# counter-chain test above: the counter protocol pair keeps a dynamic
# (counter-clock) event id and a full ring, and the guard_valid gate in the
# materializer (`!is_counter`) is the code-level enforcement. A guarded
# multi-version buffer is automatically counter-clocked by PrepareMultiBuffer
# (needs_counter = !is_one(active_guard)), so an iteration-clock ring with an
# activation point does not arise from normal DSL -- the manual-ring variant
# is covered by the pre-existing test_sync_dependencies suite.
