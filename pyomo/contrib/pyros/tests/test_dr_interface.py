"""
Tests for the decision rule interface.
"""

import pyomo.common.unittest as unittest
from pyomo.contrib.pyros.dr_interface import DecisionRuleInterface
from pyomo.core.base import ConcreteModel, Param, Var
from pyomo.common.dependencies import (
    attempt_import,
    numpy as np,
    numpy_available,
    scipy_available,
)

parameterized, param_available = attempt_import('parameterized')

if not (numpy_available and scipy_available and param_available):
    raise unittest.SkipTest("PyROS unit tests require parameterized, numpy, and scipy")


class TestConstructor(unittest.TestCase):
    @parameterized.parameterized.expand([[0], [1], [2]])
    def test_valid_construction(self, dr_order):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[1.5, 3.0, 4.5], [2.5, 5.0, 7.5]]
        quadratic_coeffs = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[2, 4, 6], [8, 10, 12], [14, 16, 18]],
        ]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs if dr_order >= 1 else None,
            quadratic_coeffs=quadratic_coeffs if dr_order >= 2 else None,
        )
        np.testing.assert_equal(dri.static_coeffs, static_coeffs)
        if dr_order >= 1:
            np.testing.assert_equal(dri.affine_coeffs, affine_coeffs)
        else:
            self.assertIsNone(dri.affine_coeffs)
        if dr_order >= 2:
            np.testing.assert_equal(dri.quadratic_coeffs, quadratic_coeffs)
        else:
            self.assertIsNone(dri.quadratic_coeffs)
        self.assertEqual(dri.order, dr_order)
        self.assertEqual(dri.second_stage_var_dim, 2)

    def test_invalid_construction_affine_quad(self):
        static_coeffs = [0.5, 0.8]
        quadratic_coeffs = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[2, 4, 6], [8, 10, 12], [14, 16, 18]],
        ]
        exc_str = "Argument `affine_coeffs`.*rather than None"
        with self.assertRaisesRegex(ValueError, exc_str):
            DecisionRuleInterface(
                static_coeffs=static_coeffs,
                affine_coeffs=None,
                quadratic_coeffs=quadratic_coeffs,
            )

    def test_invalid_construction_affine_shape(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[1.5, 3.0, 4.5], [2.5, 5.0, 7.5], [5, 10, 15]]
        exc_str = (
            "`affine_coeffs`.*same size along axis 0.*`static_coeffs`.*"
            "got 3, expected 2"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            DecisionRuleInterface(
                static_coeffs=static_coeffs,
                affine_coeffs=affine_coeffs,
                quadratic_coeffs=None,
            )

    def test_invalid_construction_quadratic_len(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[1.5, 3.0, 4.5], [2.5, 5.0, 7.5]]
        quadratic_coeffs = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
        exc_str = (
            "along axis 0.*`quadratic_coeffs`.*`static_coeffs`" ".*got 1, expected 2"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            DecisionRuleInterface(
                static_coeffs=static_coeffs,
                affine_coeffs=affine_coeffs,
                quadratic_coeffs=quadratic_coeffs,
            )

    def test_invalid_construction_quadratic_affine_mismatch(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[1.5, 3.0, 4.5], [2.5, 5.0, 7.5]]
        quadratic_coeffs = [[[1, 2], [4, 5]], [[2, 4], [8, 10]]]
        exc_str = (
            "along axis 1.*`quadratic_coeffs`.*that of `affine_coeffs`"
            ".*got 2, expected 3"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            DecisionRuleInterface(
                static_coeffs=static_coeffs,
                affine_coeffs=affine_coeffs,
                quadratic_coeffs=quadratic_coeffs,
            )

    def test_invalid_construction_quadratic_nonsquare(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[1.5, 3.0, 4.5], [2.5, 5.0, 7.5]]
        quadratic_coeffs = [[[1, 2], [4, 5], [7, 8]], [[2, 4], [8, 10], [14, 16]]]
        exc_str = "axis 1 size is 3, axis 2 size is 2"
        with self.assertRaisesRegex(ValueError, exc_str):
            DecisionRuleInterface(
                static_coeffs=static_coeffs,
                affine_coeffs=affine_coeffs,
                quadratic_coeffs=quadratic_coeffs,
            )


class TestComputePolynomialDegrees(unittest.TestCase):
    @parameterized.parameterized.expand([[0], [1], [2]])
    def test_degree_zero_coeffs(self, dr_order):
        static_coeffs = np.zeros(2)
        affine_coeffs = np.zeros((2, 3))
        quadratic_coeffs = np.zeros((2, 3, 3))
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs if dr_order >= 1 else None,
            quadratic_coeffs=quadratic_coeffs if dr_order >= 2 else None,
        )
        np.testing.assert_equal(dri.compute_polynomial_degrees(), [-1, -1])
        np.testing.assert_equal(dri.compute_polynomial_degrees(ss_idxs=[0]), [-1])
        np.testing.assert_equal(dri.compute_polynomial_degrees(ss_idxs=[1]), [-1])
        np.testing.assert_equal(
            dri.compute_polynomial_degrees(ss_idxs=[0, 1]), [-1, -1]
        )

    def test_degree_static_dr(self):
        static_coeffs = np.array([0, 1.5, 1e-12])
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        np.testing.assert_equal(dri.compute_polynomial_degrees(), [-1, 0, -1])
        np.testing.assert_equal(dri.compute_polynomial_degrees(tol=0), [-1, 0, 0])
        np.testing.assert_equal(dri.compute_polynomial_degrees([0, 1]), [-1, 0])

    def test_degree_affine_dr(self):
        static_coeffs = np.array([0, 1.5, 1e-12])
        affine_coeffs = np.array([[0, 0, 0, 1e-15], [1e-12, 0, 1e-10, 0], [1, 0, 2, 0]])
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        np.testing.assert_equal(dri.compute_polynomial_degrees(), [-1, 0, 1])
        np.testing.assert_equal(dri.compute_polynomial_degrees(tol=0), [1, 1, 1])
        np.testing.assert_equal(dri.compute_polynomial_degrees([1, 2]), [0, 1])
        np.testing.assert_equal(dri.compute_polynomial_degrees([1, 2], tol=0), [1, 1])

    def test_degree_quadratic_dr(self):
        static_coeffs = np.array([0, 1e-12, 1.5, 0, 1])
        affine_coeffs = np.array(
            [
                [0, 0, 0, 1e-15],
                [1e-12, 0, 1e-10, 0],
                [1, 0, 2, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        quadratic_coeffs = np.array(
            [
                [[1e-11] * 4] * 4,
                [[0] * 4] * 4,
                np.full((4, 4), 1e-12),
                np.arange(16).reshape((4, 4)),
                np.full((4, 4), 6e-11),
            ]
        )
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=quadratic_coeffs,
        )
        np.testing.assert_equal(dri.compute_polynomial_degrees(), [-1, -1, 1, 2, 2])
        np.testing.assert_equal(dri.compute_polynomial_degrees(tol=0), [2, 1, 2, 2, 2])
        np.testing.assert_equal(
            dri.compute_polynomial_degrees(tol=1e-9), [-1, -1, 1, 2, 0]
        )

        np.testing.assert_equal(dri.compute_polynomial_degrees([1, 4]), [-1, 2])
        np.testing.assert_equal(dri.compute_polynomial_degrees([1, 4], tol=0), [1, 2])
        np.testing.assert_equal(
            dri.compute_polynomial_degrees([1, 4], tol=1e-9), [-1, 0]
        )


class TestGetNumCoeffsPerSSDim(unittest.TestCase):
    def test_get_num_coeffs_static(self):
        static_coeffs = np.ones(5)
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        self.assertEqual(dri.get_num_coeffs_per_ss_dim(simplified=False), 1)
        self.assertEqual(dri.get_num_coeffs_per_ss_dim(simplified=True), 1)

    def test_get_num_coeffs_affine(self):
        static_coeffs = np.ones(5)
        affine_coeffs = np.ones((5, 6))
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        self.assertEqual(dri.get_num_coeffs_per_ss_dim(simplified=False), 7)
        self.assertEqual(dri.get_num_coeffs_per_ss_dim(simplified=True), 7)

    def test_get_num_coeffs_quadratic(self):
        static_coeffs = np.ones(5)
        affine_coeffs = np.ones((5, 6))
        quadratic_coeffs = np.ones((5, 6, 6))
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=quadratic_coeffs,
        )
        self.assertEqual(dri.get_num_coeffs_per_ss_dim(simplified=False), 43)
        self.assertEqual(dri.get_num_coeffs_per_ss_dim(simplified=True), 28)


class TestGetParamIndexToCoeffMap(unittest.TestCase):
    @parameterized.parameterized.expand([[True], [False]])
    def test_map_static(self, simplified):
        static_coeffs = [0.5, 0.8]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        self.assertDictEqual(
            dri.get_param_idx_to_coeff_map(0, simplified=simplified),
            {(): np.float64(0.5)},
        )
        self.assertDictEqual(
            dri.get_param_idx_to_coeff_map(1, simplified=simplified),
            {(): np.float64(0.8)},
        )

    @parameterized.parameterized.expand([[True], [False]])
    def test_map_affine(self, simplified):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        self.assertDictEqual(
            dri.get_param_idx_to_coeff_map(0, simplified=simplified),
            {(): np.float64(0.5), (0,): np.int_(2), (1,): np.int_(4), (2,): np.int_(6)},
        )
        self.assertDictEqual(
            dri.get_param_idx_to_coeff_map(1, simplified=simplified),
            {
                (): np.float64(0.8),
                (0,): np.int_(8),
                (1,): np.int_(9),
                (2,): np.int_(10),
            },
        )

    @parameterized.parameterized.expand([[True], [False]])
    def test_map_quadratic(self, simplified):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        quadratic_coeffs = np.arange(0.01, 0.19, 0.01).reshape((2, 3, 3))
        quadratic_coeffs = [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            [[0.10, 0.11, 0.12], [0.13, 0.14, 0.15], [0.16, 0.17, 0.18]],
        ]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=quadratic_coeffs,
        )
        if simplified:
            quadratic_map_0 = {
                (0, 0): np.float64(0.01),
                (0, 1): np.float64(0.02 + 0.04),
                (0, 2): np.float64(0.03 + 0.07),
                (1, 1): np.float64(0.05),
                (1, 2): np.float64(0.06 + 0.08),
                (2, 2): np.float64(0.09),
            }
        else:
            quadratic_map_0 = {
                (0, 0): np.float64(0.01),
                (0, 1): np.float64(0.02),
                (0, 2): np.float64(0.03),
                (1, 0): np.float64(0.04),
                (1, 1): np.float64(0.05),
                (1, 2): np.float64(0.06),
                (2, 0): np.float64(0.07),
                (2, 1): np.float64(0.08),
                (2, 2): np.float64(0.09),
            }
        self.assertDictEqual(
            dri.get_param_idx_to_coeff_map(0, simplified=simplified),
            {
                (): np.float64(0.5),
                (0,): np.int_(2),
                (1,): np.int_(4),
                (2,): np.int_(6),
                **quadratic_map_0,
            },
        )

        if simplified:
            quadratic_map_1 = {
                (0, 0): np.float64(0.10),
                (0, 1): np.float64(0.11 + 0.13),
                (0, 2): np.float64(0.12 + 0.16),
                (1, 1): np.float64(0.14),
                (1, 2): np.float64(0.15 + 0.17),
                (2, 2): np.float64(0.18),
            }
        else:
            quadratic_map_1 = {
                (0, 0): np.float64(0.10),
                (0, 1): np.float64(0.11),
                (0, 2): np.float64(0.12),
                (1, 0): np.float64(0.13),
                (1, 1): np.float64(0.14),
                (1, 2): np.float64(0.15),
                (2, 0): np.float64(0.16),
                (2, 1): np.float64(0.17),
                (2, 2): np.float64(0.18),
            }
        self.assertDictEqual(
            dri.get_param_idx_to_coeff_map(1, simplified=simplified),
            {
                (): np.float64(0.8),
                (0,): np.int_(8),
                (1,): np.int_(9),
                (2,): np.int_(10),
                **quadratic_map_1,
            },
        )


class TestSetupDRComponents(unittest.TestCase):
    @parameterized.parameterized.expand([[True], [False]])
    def test_setup_dr_components_static(self, simplified):
        static_coeffs = [0.5, 0.8]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        comp_list = dri.setup_dr_components(simplified=simplified)
        self.assertEqual(len(comp_list), 2)
        for comp in comp_list:
            self.assertTrue(comp.mutable)
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[0].items()}, {0: np.float64(0.5)}
        )
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[1].items()}, {0: np.float64(0.8)}
        )

    @parameterized.parameterized.expand([[True], [False]])
    def test_setup_dr_components_affine(self, simplified):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        comp_list = dri.setup_dr_components(simplified=simplified)
        self.assertEqual(len(comp_list), 2)
        for comp in comp_list:
            self.assertTrue(comp.mutable)
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[0].items()},
            {0: np.float64(0.5), 1: np.int_(2), 2: np.int_(4), 3: np.int_(6)},
        )
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[1].items()},
            {0: np.float64(0.8), 1: np.int_(8), 2: np.int_(9), 3: np.int_(10)},
        )

    def test_setup_dr_components_quadratic_simplified(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        quadratic_coeffs = [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            [[0.10, 0.11, 0.12], [0.13, 0.14, 0.15], [0.16, 0.17, 0.18]],
        ]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=quadratic_coeffs,
        )
        comp_list = dri.setup_dr_components(simplified=True)
        self.assertEqual(len(comp_list), 2)
        for comp in comp_list:
            self.assertTrue(comp.mutable)
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[0].items()},
            {
                0: np.float64(0.5),
                1: np.int_(2),
                2: np.int_(4),
                3: np.int_(6),
                4: np.float64(0.01),
                5: np.float64(0.02 + 0.04),
                6: np.float64(0.03 + 0.07),
                7: np.float64(0.05),
                8: np.float64(0.06 + 0.08),
                9: np.float64(0.09),
            },
        )
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[1].items()},
            {
                0: np.float64(0.8),
                1: np.int_(8),
                2: np.int_(9),
                3: np.int_(10),
                4: np.float64(0.10),
                5: np.float64(0.11 + 0.13),
                6: np.float64(0.12 + 0.16),
                7: np.float64(0.14),
                8: np.float64(0.15 + 0.17),
                9: np.float64(0.18),
            },
        )

    def test_setup_dr_components_quadratic_full(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        quadratic_coeffs = [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            [[0.10, 0.11, 0.12], [0.13, 0.14, 0.15], [0.16, 0.17, 0.18]],
        ]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=quadratic_coeffs,
        )
        comp_list = dri.setup_dr_components(simplified=False)
        self.assertEqual(len(comp_list), 2)
        for comp in comp_list:
            self.assertTrue(comp.mutable)
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[0].items()},
            {
                0: np.float64(0.5),
                1: np.int_(2),
                2: np.int_(4),
                3: np.int_(6),
                4: np.float64(0.01),
                5: np.float64(0.02),
                6: np.float64(0.03),
                7: np.float64(0.04),
                8: np.float64(0.05),
                9: np.float64(0.06),
                10: np.float64(0.07),
                11: np.float64(0.08),
                12: np.float64(0.09),
            },
        )
        self.assertDictEqual(
            {idx: cd.value for idx, cd in comp_list[1].items()},
            {
                0: np.float64(0.8),
                1: np.int_(8),
                2: np.int_(9),
                3: np.int_(10),
                4: np.float64(0.10),
                5: np.float64(0.11),
                6: np.float64(0.12),
                7: np.float64(0.13),
                8: np.float64(0.14),
                9: np.float64(0.15),
                10: np.float64(0.16),
                11: np.float64(0.17),
                12: np.float64(0.18),
            },
        )


class TestSetDRComponentValues(unittest.TestCase):
    def test_set_values_length_mismatch(self):
        static_coeffs = [0.5, 0.8]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        comp1 = Param([0], initialize=0, mutable=True)
        comp2 = Param([0], initialize=0, mutable=True)
        comp3 = Param([0], initialize=0, mutable=True)
        exc_str = (
            r"`dr_components` should match second-stage dimension "
            r"\(got 3, expected 2\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            dri.set_dr_component_values([comp1, comp2, comp3])

    def test_set_values_num_coeffs_mismatch(self):
        static_coeffs = [0.5, 0.8]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        m = ConcreteModel()
        m.comp1 = Param([0, 1], initialize=0, mutable=True)
        m.comp2 = Param([0, 1], initialize=0, mutable=True)
        exc_str = "`dr_component`.*expected.*per dimension" ".*got 2, expected 1"
        with self.assertRaisesRegex(ValueError, exc_str):
            dri.set_dr_component_values([m.comp1, m.comp2])


class TestGenerateDRExprs(unittest.TestCase):
    @parameterized.parameterized.expand([[True], [False]])
    def test_generate_static_exprs(self, simplified):
        static_coeffs = [0.5, 0.8]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        m = ConcreteModel()
        m.dr0 = Param([0], initialize=0, mutable=True)
        m.dr1 = Param([0], initialize=0, mutable=True)
        m.q = Param([0, 1], initialize=0, mutable=True)
        exprs = dri.generate_dr_exprs(
            [m.dr0, m.dr1], uncertain_params=m.q, simplified=simplified
        )
        self.assertExpressionsEqual(exprs[0], m.dr0[0])
        self.assertExpressionsEqual(exprs[1], m.dr1[0])

    @parameterized.parameterized.expand([[True], [False]])
    def test_generate_affine_exprs(self, simplified):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        m = ConcreteModel()
        m.dr0 = Param(range(4), initialize=0, mutable=True)
        m.dr1 = Param(range(4), initialize=0, mutable=True)
        m.q = Param([0, 1, 2], initialize=0, mutable=True)
        exprs = dri.generate_dr_exprs(
            [m.dr0, m.dr1],
            uncertain_params=[m.q[0], m.q[1], m.q[2]],
            simplified=simplified,
        )
        self.assertExpressionsEqual(
            exprs[0],
            m.dr0[0] + m.q[0] * m.dr0[1] + m.q[1] * m.dr0[2] + m.q[2] * m.dr0[3],
        )
        self.assertExpressionsEqual(
            exprs[1],
            m.dr1[0] + m.q[0] * m.dr1[1] + m.q[1] * m.dr1[2] + m.q[2] * m.dr1[3],
        )

    def test_generate_quadratic_exprs_simplified(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        quadratic_coeffs = [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            [[0.10, 0.11, 0.12], [0.13, 0.14, 0.15], [0.16, 0.17, 0.18]],
        ]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=quadratic_coeffs,
        )
        m = ConcreteModel()
        m.dr0 = Param(range(10), initialize=0, mutable=True)
        m.dr1 = Param(range(10), initialize=0, mutable=True)
        m.q = Param([0, 1, 2], initialize=0, mutable=True)
        exprs = dri.generate_dr_exprs(
            [m.dr0, m.dr1], uncertain_params=m.q, simplified=True
        )
        self.assertExpressionsEqual(
            exprs[0],
            (
                m.dr0[0]
                + m.q[0] * m.dr0[1]
                + m.q[1] * m.dr0[2]
                + m.q[2] * m.dr0[3]
                + (m.q[0] * m.q[0]) * m.dr0[4]
                + (m.q[0] * m.q[1]) * m.dr0[5]
                + (m.q[0] * m.q[2]) * m.dr0[6]
                + (m.q[1] * m.q[1]) * m.dr0[7]
                + (m.q[1] * m.q[2]) * m.dr0[8]
                + (m.q[2] * m.q[2]) * m.dr0[9]
            ),
        )
        self.assertExpressionsEqual(
            exprs[1],
            (
                m.dr1[0]
                + m.q[0] * m.dr1[1]
                + m.q[1] * m.dr1[2]
                + m.q[2] * m.dr1[3]
                + (m.q[0] * m.q[0]) * m.dr1[4]
                + (m.q[0] * m.q[1]) * m.dr1[5]
                + (m.q[0] * m.q[2]) * m.dr1[6]
                + (m.q[1] * m.q[1]) * m.dr1[7]
                + (m.q[1] * m.q[2]) * m.dr1[8]
                + (m.q[2] * m.q[2]) * m.dr1[9]
            ),
        )

    def test_generate_quadratic_exprs_full(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        quadratic_coeffs = [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            [[0.10, 0.11, 0.12], [0.13, 0.14, 0.15], [0.16, 0.17, 0.18]],
        ]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=quadratic_coeffs,
        )
        m = ConcreteModel()
        m.dr0 = Param(range(13), initialize=0, mutable=True)
        m.dr1 = Param(range(13), initialize=0, mutable=True)
        m.q = Param([0, 1, 2], initialize=0, mutable=True)
        exprs = dri.generate_dr_exprs(
            [m.dr0, m.dr1], uncertain_params=m.q, simplified=False
        )
        self.assertExpressionsEqual(
            exprs[0],
            (
                m.dr0[0]
                + m.q[0] * m.dr0[1]
                + m.q[1] * m.dr0[2]
                + m.q[2] * m.dr0[3]
                + (m.q[0] * m.q[0]) * m.dr0[4]
                + (m.q[0] * m.q[1]) * m.dr0[5]
                + (m.q[0] * m.q[2]) * m.dr0[6]
                + (m.q[1] * m.q[0]) * m.dr0[7]
                + (m.q[1] * m.q[1]) * m.dr0[8]
                + (m.q[1] * m.q[2]) * m.dr0[9]
                + (m.q[2] * m.q[0]) * m.dr0[10]
                + (m.q[2] * m.q[1]) * m.dr0[11]
                + (m.q[2] * m.q[2]) * m.dr0[12]
            ),
        )
        self.assertExpressionsEqual(
            exprs[1],
            (
                m.dr1[0]
                + m.q[0] * m.dr1[1]
                + m.q[1] * m.dr1[2]
                + m.q[2] * m.dr1[3]
                + (m.q[0] * m.q[0]) * m.dr1[4]
                + (m.q[0] * m.q[1]) * m.dr1[5]
                + (m.q[0] * m.q[2]) * m.dr1[6]
                + (m.q[1] * m.q[0]) * m.dr1[7]
                + (m.q[1] * m.q[1]) * m.dr1[8]
                + (m.q[1] * m.q[2]) * m.dr1[9]
                + (m.q[2] * m.q[0]) * m.dr1[10]
                + (m.q[2] * m.q[1]) * m.dr1[11]
                + (m.q[2] * m.q[2]) * m.dr1[12]
            ),
        )

    def test_generate_exprs_num_comps_mismatch(self):
        static_coeffs = [0.5, 0.8]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        m = ConcreteModel()
        m.dr0 = Param(range(13), initialize=0, mutable=True)
        m.q = Param([0, 1, 2], initialize=0, mutable=True)
        exc_str = (
            "Length of `dr_components` should match "
            "second-stage dimension "
            r"\(got 1, expected 2\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            dri.generate_dr_exprs([m.dr0], uncertain_params=m.q, simplified=False)

    def test_generate_exprs_num_coeffs_per_dim_mismatch(self):
        static_coeffs = [0.5, 0.8]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs, affine_coeffs=None, quadratic_coeffs=None
        )
        m = ConcreteModel()
        m.dr0 = Param(range(13), initialize=0, mutable=True)
        m.dr1 = Param(range(13), initialize=0, mutable=True)
        m.q = Param([0, 1, 2], initialize=0, mutable=True)
        exc_str = "`dr_component`.*expected.*per dimension" ".*got 13, expected 1"
        with self.assertRaisesRegex(ValueError, exc_str):
            dri.generate_dr_exprs(
                [m.dr0, m.dr1], uncertain_params=m.q, simplified=False
            )

    def test_generate_exprs_num_params_mismatch(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        m = ConcreteModel()
        m.dr0 = Param(range(13), initialize=0, mutable=True)
        m.dr1 = Param(range(13), initialize=0, mutable=True)
        m.q = Param([0], initialize=0, mutable=True)
        exc_str = (
            "`uncertain_params`.*length along axis 1.*affine" ".*got 1, expected 3"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            dri.generate_dr_exprs(
                [m.dr0, m.dr1], uncertain_params=[m.q[0]], simplified=False
            )


class TestGenerateDREquations(unittest.TestCase):
    def test_generate_dr_equations_num_ss_vars_mismatch(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        m = ConcreteModel()
        m.dr0 = Param(range(4), initialize=0, mutable=True)
        m.dr1 = Param(range(4), initialize=0, mutable=True)
        m.q = Param([0, 1, 2], initialize=0, mutable=True)
        m.z = Var([0, 1, 2], initialize=0)

        exc_str = (
            "`second_stage_variables`.*match.*`dr_components" ".*got 3, expected 2"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            dri.generate_dr_equations(
                dr_components=[m.dr0, m.dr1],
                second_stage_variables=[m.z[0], m.z[1], m.z[2]],
                uncertain_params=[m.q[0], m.q[1], m.q[2]],
            )

    def test_generate_dr_equations(self):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs,
            quadratic_coeffs=None,
        )
        m = ConcreteModel()
        m.dr0 = Param(range(4), initialize=0, mutable=True)
        m.dr1 = Param(range(4), initialize=0, mutable=True)
        m.q = Param([0, 1, 2], initialize=0, mutable=True)
        m.z = Var([0, 1], initialize=0)
        conlist = dri.generate_dr_equations(
            dr_components=[m.dr0, m.dr1],
            second_stage_variables=[m.z[0], m.z[1]],
            uncertain_params=[m.q[0], m.q[1], m.q[2]],
        )
        self.assertExpressionsEqual(
            conlist[1].expr,
            (
                m.dr0[0]
                + m.q[0] * m.dr0[1]
                + m.q[1] * m.dr0[2]
                + m.q[2] * m.dr0[3]
                - m.z[0]
                == 0
            ),
        )
        self.assertExpressionsEqual(
            conlist[2].expr,
            (
                m.dr1[0]
                + m.q[0] * m.dr1[1]
                + m.q[1] * m.dr1[2]
                + m.q[2] * m.dr1[3]
                - m.z[1]
                == 0
            ),
        )


class TestEvaluateDR(unittest.TestCase):
    @parameterized.parameterized.expand([[0], [1], [2]])
    def test_evaluate_dr(self, dr_order):
        static_coeffs = [0.5, 0.8]
        affine_coeffs = [[2, 4, 6], [8, 9, 10]]
        quadratic_coeffs = [
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            [[0.10, 0.11, 0.12], [0.13, 0.14, 0.15], [0.16, 0.17, 0.18]],
        ]
        dri = DecisionRuleInterface(
            static_coeffs=static_coeffs,
            affine_coeffs=affine_coeffs if dr_order >= 1 else None,
            quadratic_coeffs=quadratic_coeffs if dr_order >= 2 else None,
        )
        dr_eval = dri.evaluate([2, 1, 0.5])
        expected_evals = {
            0: static_coeffs,
            1: static_coeffs + np.array([11, 30]),
            2: static_coeffs + np.array([11, 30]) + np.array([0.4025, 1.505]),
        }
        np.testing.assert_allclose(dr_eval, expected_evals[dr_order])


if __name__ == "__main__":
    unittest.main()
