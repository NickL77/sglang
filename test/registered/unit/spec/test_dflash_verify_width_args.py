"""Unit tests for the DFLASH draft-block / verify-width resolution."""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"


class TestDFlashVerifyWidthArgs(CustomTestCase):
    def _resolve(self, **fields) -> ServerArgs:
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="draft",
            **fields,
        )
        with (
            # The backend resolver probes the platform and the block-size
            # inference loads the draft HF config over the network (as does
            # the algorithm-alias step of the full hook, hence the direct call).
            patch(f"{HOOK_MODULE}._resolve_dflash_draft_attention_backend"),
            patch(f"{HOOK_MODULE}._infer_dflash_block_size", return_value=16),
        ):
            _handle_dflash(args)
        return args

    @staticmethod
    def _widths(args: ServerArgs) -> tuple[int, int]:
        return (
            resolution_result(args, "speculative_dflash_block_size"),
            resolution_result(args, "speculative_num_draft_tokens"),
        )

    def test_one_knob_sets_both_widths(self):
        cases = (
            ({}, (16, 16)),
            ({"speculative_num_draft_tokens": 8}, (8, 8)),
            ({"speculative_dflash_block_size": 8}, (8, 8)),
            (
                {"speculative_dflash_block_size": 8, "speculative_num_draft_tokens": 8},
                (8, 8),
            ),
        )
        for fields, expected in cases:
            with self.subTest(fields=fields):
                self.assertEqual(self._widths(self._resolve(**fields)), expected)

    def test_verify_width_may_be_narrower_than_the_draft_block(self):
        args = self._resolve(
            speculative_dflash_block_size=16, speculative_num_draft_tokens=8
        )
        self.assertEqual(self._widths(args), (16, 8))
        # KV is reserved for the whole drafted block, not just the verified head.
        self.assertEqual(
            SpeculativeAlgorithm.DFLASH.resolve_max_speculative_num_draft_tokens(args),
            16,
        )

    def test_verify_width_above_the_draft_block_is_rejected(self):
        for verify_width in (0, 17):
            with self.subTest(verify_width=verify_width):
                with self.assertRaisesRegex(ValueError, "verify width"):
                    self._resolve(
                        speculative_dflash_block_size=16,
                        speculative_num_draft_tokens=verify_width,
                    )

    def test_draft_window_is_checked_against_the_draft_block(self):
        with self.assertRaisesRegex(ValueError, "speculative-draft-window-size"):
            self._resolve(
                speculative_dflash_block_size=16,
                speculative_num_draft_tokens=8,
                speculative_draft_window_size=12,
            )

    def test_draft_worker_runs_the_full_block_while_target_verifies_its_head(self):
        override = get_context().override_server_args(
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="draft",
            speculative_dflash_block_size=16,
            speculative_num_draft_tokens=8,
        )
        override.install()
        self.addCleanup(override.restore)
        algo = SpeculativeAlgorithm.DFLASH
        self.assertEqual(
            algo.get_num_tokens_per_req_for_target_verify(8, is_draft_worker=True), 16
        )
        self.assertEqual(
            algo.get_num_tokens_per_req_for_target_verify(8, is_draft_worker=False), 8
        )


if __name__ == "__main__":
    unittest.main()
