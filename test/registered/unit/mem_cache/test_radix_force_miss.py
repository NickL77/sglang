"""Unit tests for SGLANG_RADIX_FORCE_MISS.

The flag is gated at the scheduler boundary, so we test the helper directly
plus an end-to-end check of `match_kv_cache` driving a populated
RadixCache.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.common import bidir_state_block_cached, match_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class _StubReq:
    def __init__(self, token_ids):
        self.origin_input_ids = array("q", token_ids)
        self.output_ids = array("q")
        self.extra_key = None
        self.cache_salt = None
        self.prefix_len = 0
        self.last_node = None
        self.last_host_node = None
        self.best_match_node = None
        self.host_hit_length = None
        self.num_matched_prefix_tokens = 0
        self.swa_branching_seqlen = None
        self.bidir_prefix_len = -1
        self.skip_radix_cache_insert = False
        self.is_prefill_only = False
        self.return_logprob = False
        self.return_hidden_states = False
        self.kv = SimpleNamespace(cache_protected_len=None, holds_mamba=False)

    def _compute_max_prefix_len(self, input_len):
        return max(input_len - 1, 0)


class TestZeroMatchResult(unittest.TestCase):
    def test_zero_replaces_indices_and_nodes(self):
        tree = RadixCache.create_simulated()
        tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 4, 5]))))
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 9])))
        )
        self.assertGreater(match.device_prefix_len, 0)
        zeroed = zero_match_result(tree, match)
        self.assertEqual(int(zeroed.device_prefix_len), 0)
        self.assertIs(zeroed.last_device_node, tree.root_node)
        self.assertIs(zeroed.last_host_node, tree.root_node)
        self.assertIs(zeroed.best_match_node, tree.root_node)
        self.assertEqual(zeroed.host_hit_length, 0)

    def test_non_sharing_cache_is_passthrough(self):
        class _StubNonSharingCache:
            def supports_prefix_sharing(self) -> bool:
                return False

        original = MatchResult(
            device_prefix_len=0,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=0,
        )
        self.assertIs(zero_match_result(_StubNonSharingCache(), original), original)


class TestMatchKvCacheForceMiss(unittest.TestCase):
    def test_swa_branching_seqlen_is_cleared_without_new_branch(self):
        class _StubTreeCache:
            def swa_reprefill_tail_tokens(self, req=None):
                return 0

            def match_prefix(self, params):
                return MatchResult(
                    device_prefix_len=0,
                    last_device_node=None,
                    last_host_node=None,
                    best_match_node=None,
                    host_hit_length=0,
                    swa_branching_seqlen=None,
                )

        req = _StubReq([1, 2, 3, 4])
        req.swa_branching_seqlen = 8

        match_kv_cache(req, _StubTreeCache())

        self.assertIsNone(req.swa_branching_seqlen)

    def test_force_miss_zeros_req_prefix(self):
        tree = RadixCache.create_simulated()
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [10, 11, 12, 13, 14, 15, 16, 17]))
            )
        )

        # Sanity: without the flag, the same lookup hits.
        baseline_req = _StubReq([10, 11, 12, 13, 99, 100])
        with envs.SGLANG_RADIX_FORCE_MISS.override(False):
            match_kv_cache(baseline_req, tree)
        self.assertGreater(baseline_req.prefix_len, 0)
        self.assertIsNot(baseline_req.last_node, tree.root_node)

        # With the flag, the same lookup is forced to miss.
        forced_req = _StubReq([10, 11, 12, 13, 99, 100])
        with envs.SGLANG_RADIX_FORCE_MISS.override(True):
            match_kv_cache(forced_req, tree)
        self.assertEqual(forced_req.prefix_len, 0)
        self.assertIs(forced_req.last_node, tree.root_node)
        self.assertIs(forced_req.last_host_node, tree.root_node)
        self.assertEqual(forced_req.host_hit_length, 0)


class TestBidirQuestionInsert(unittest.TestCase):
    def _tree(self):
        tree = RadixCache.create_simulated()
        tree.insert(InsertParams(key=RadixKey(token_ids=array("q", range(10, 18)))))
        return tree

    def _question(self, token_ids):
        req = _StubReq(token_ids)
        req.bidir_prefix_len = 8
        return req

    def test_question_on_cached_state_is_not_inserted(self):
        req = self._question(list(range(10, 18)) + [99, 100])
        match_kv_cache(req, self._tree())
        self.assertTrue(req.skip_radix_cache_insert)

    def test_uncached_question_is_still_inserted(self):
        # The finished request is what caches its state block.
        req = self._question(list(range(20, 28)) + [99, 100])
        match_kv_cache(req, self._tree())
        self.assertFalse(req.skip_radix_cache_insert)

    def test_state_only_request_is_still_inserted(self):
        req = self._question(range(10, 18))
        match_kv_cache(req, self._tree())
        self.assertFalse(req.skip_radix_cache_insert)


class TestBidirStateBlockCached(unittest.TestCase):
    def _state_req(self, token_ids):
        req = _StubReq(token_ids)
        req.bidir_prefix_len = len(token_ids)
        req.is_prefill_only = True
        return req

    def _tree(self):
        tree = RadixCache.create_simulated()
        tree.insert(InsertParams(key=RadixKey(token_ids=array("q", range(10, 18)))))
        return tree

    def test_cached_state_only_request_has_nothing_to_compute(self):
        tree = self._tree()
        req = self._state_req(range(10, 18))

        # The scheduler's lookup stops one token short, which the full-block guard rejects.
        match_kv_cache(req, tree, max_prefix_len=req._compute_max_prefix_len(8))
        self.assertEqual(req.prefix_len, 0)

        self.assertTrue(bidir_state_block_cached(req, tree))

    def test_state_past_the_recurrent_checkpoint_is_cached(self):
        # A hybrid cache reports the last recurrent checkpoint as the device prefix.
        tree = self._tree()
        req = self._state_req(range(10, 18))
        full = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", range(10, 18))))
        )
        tree.match_prefix = lambda params: full._replace(
            device_prefix_len=4, full_kv_hit_length=8
        )
        self.assertTrue(bidir_state_block_cached(req, tree))

        tree.match_prefix = lambda params: full._replace(
            device_prefix_len=4, full_kv_hit_length=7
        )
        self.assertFalse(bidir_state_block_cached(req, tree))

    def test_uncached_or_partly_cached_state_is_computed(self):
        tree = self._tree()
        self.assertFalse(bidir_state_block_cached(self._state_req(range(20, 28)), tree))
        self.assertFalse(
            bidir_state_block_cached(self._state_req(list(range(10, 17)) + [99]), tree)
        )
        self.assertFalse(bidir_state_block_cached(self._state_req(range(10, 19)), tree))

    def test_only_state_only_prefills_qualify(self):
        tree = self._tree()

        question = self._state_req(range(10, 18))
        question.bidir_prefix_len = 6
        self.assertFalse(bidir_state_block_cached(question, tree))

        causal = self._state_req(range(10, 18))
        causal.bidir_prefix_len = -1
        self.assertFalse(bidir_state_block_cached(causal, tree))

        generating = self._state_req(range(10, 18))
        generating.is_prefill_only = False
        self.assertFalse(bidir_state_block_cached(generating, tree))

        with_logprob = self._state_req(range(10, 18))
        with_logprob.return_logprob = True
        self.assertFalse(bidir_state_block_cached(with_logprob, tree))

    def test_force_miss_is_respected(self):
        tree = self._tree()
        with envs.SGLANG_RADIX_FORCE_MISS.override(True):
            self.assertFalse(
                bidir_state_block_cached(self._state_req(range(10, 18)), tree)
            )


if __name__ == "__main__":
    unittest.main()
