import unittest

import numpy as np

from src.build_kgt1_dataset import (
    DATASIZE,
    make_open_times,
    overlap_fraction,
    relative_packets_from_cw,
    mix_live_pool,
)


class TestLiveMixer(unittest.TestCase):
    def test_relative_packet_conversion(self):
        trace = np.array([0.01, -0.02, 0.0, 0.03], dtype=np.float64)

        packets = relative_packets_from_cw(trace)

        self.assertEqual(
            packets,
            [
                (0.01, DATASIZE),
                (0.02, -DATASIZE),
                (0.03, DATASIZE),
            ],
        )

    def test_simultaneous_open_times(self):
        rng = np.random.default_rng(7)

        open_times = make_open_times(
            K=3,
            arrival_mode="simultaneous",
            stagger_seconds=5.0,
            rng=rng,
        )

        np.testing.assert_allclose(open_times, [0.0, 0.0, 0.0])

    def test_fixed_open_times(self):
        rng = np.random.default_rng(7)

        open_times = make_open_times(
            K=3,
            arrival_mode="fixed",
            stagger_seconds=5.0,
            rng=rng,
        )

        np.testing.assert_allclose(open_times, [0.0, 5.0, 10.0])

    def test_overlap_fraction(self):
        open_times = np.array([0.0, 5.0])
        close_times = np.array([10.0, 15.0])

        overlap = overlap_fraction(open_times, close_times)

        self.assertAlmostEqual(overlap, 5.0 / 15.0, places=8)

    def test_concurrent_mode_has_no_dummies(self):
        trace_a = np.array([0.01, -0.02, 0.03], dtype=np.float64)
        trace_b = np.array([0.01, -0.02, 0.03], dtype=np.float64)

        _, metadata = mix_live_pool(
            traces=[trace_a, trace_b],
            open_times=np.array([0.0, 1.0]),
            delta_t=0.01,
            N_out=1,
            N_in=1,
            seed=2024,
            mode="concurrent",
        )

        self.assertEqual(metadata["dummy_cells"], 0)
        self.assertEqual(metadata["real_cells_out"], 6)

    def test_full_mode_adds_dummies(self):
        trace = np.array([0.01, -0.02], dtype=np.float64)

        _, metadata = mix_live_pool(
            traces=[trace],
            open_times=np.array([0.0]),
            delta_t=0.01,
            N_out=2,
            N_in=2,
            seed=2024,
            mode="full",
        )

        self.assertGreater(metadata["dummy_cells"], 0)
        self.assertEqual(metadata["real_cells_out"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
