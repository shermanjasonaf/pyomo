#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-07-17 Mon 15:55:57

@author: jasherma

Interface to PyROS Decision rule components.
"""


import numpy as np

from pyomo.core.base.param import _ParamData as ParamData
from pyomo.core.base.var import _GeneralVarData as VarData
from pyomo.core.expr.numeric_expr import (
    NPV_ProductExpression,
    NPV_PowExpression,
)
import pyomo.environ as pyo


class DecisionRuleInterface:
    """
    Interface to PyROS second-stage decision rules.

    Decision rule equations are polynomial (up to degree 2)
    equations of the form
    d[0] + d[1] * q[0] + d[2] * q[1] + ... - z == 0
    """

    def __init__(
            self,
            model,
            second_stage_vars,
            uncertain_params,
            decision_rule_vars,
            decision_rule_eqns,
            ):
        """Initialize self (see class docstring).

        """
        param_names = tuple(param.name for param in uncertain_params)

        self.num_ssv = len(second_stage_vars)
        self.num_uncertain_params = len(uncertain_params)
        self.orig_ssv_names = [ssv.name for ssv in second_stage_vars]
        self.orig_uncertain_param_names = [
            param.name for param in uncertain_params
        ]
        self.orig_dr_var_names = [
            dr_var.name for dr_var in decision_rule_vars
        ]

        # arrays for containing DR coefficients
        constant_dr_coeffs = np.zeros(self.num_ssv)
        affine_dr_coeffs = np.zeros((self.num_ssv, self.num_uncertain_params))
        quadratic_dr_coeffs = np.zeros((
            self.num_ssv,
            self.num_uncertain_params,
            self.num_uncertain_params,
        ))

        # now parse DR constraint expressions to assemble
        # the arrays
        ssv_dr_eq_zip = zip(
            second_stage_vars,
            decision_rule_eqns,
        )
        for ssv_idx, (ssv, dr_eq) in enumerate(ssv_dr_eq_zip):
            for term in dr_eq.body.args:
                is_constant_dr_term = (
                    isinstance(term.args[0], int)
                    and term.args[0] == 1
                    and isinstance(term.args[1], VarData)
                )
                is_ssv_term = (
                    isinstance(term.args[0], int)
                    and term.args[0] == -1
                    and isinstance(term.args[1], VarData)
                )
                is_linear_term = (
                    isinstance(term.args[0], ParamData)
                    and isinstance(term.args[1], VarData)
                )
                is_bilinear_term = (
                    isinstance(term.args[0], NPV_ProductExpression)
                    and isinstance(term.args[1], VarData)
                )
                is_squared_term = (
                    isinstance(term.args[0], NPV_PowExpression)
                    and isinstance(term.args[1], VarData)
                )
                if is_constant_dr_term:
                    uncertain_param_idxs = ()
                    dr_var_value = term.args[1].value
                    constant_dr_coeffs[ssv_idx] = dr_var_value
                elif is_linear_term:
                    uncertain_param_idxs = (
                        param_names.index(term.args[0].name),
                    )
                    dr_var_value = term.args[1].value
                    affine_dr_coeffs[ssv_idx, uncertain_param_idxs[0]] = (
                        dr_var_value
                    )
                elif is_bilinear_term:
                    param_idx_1, param_idx_2 = tuple(
                        param_names.index(param.name)
                        for param in term.args[0].args
                    )
                    uncertain_param_idxs = (param_idx_1, param_idx_2)
                    assert len(uncertain_param_idxs) == 2
                    dr_var_value = term.args[1].value
                    quadratic_dr_coeffs[ssv_idx, param_idx_1, param_idx_2] = (
                        dr_var_value / 2
                    )
                    quadratic_dr_coeffs[ssv_idx, param_idx_2, param_idx_1] = (
                        dr_var_value / 2
                    )
                elif is_squared_term:
                    param = term.args[0].args[0]
                    param_idx = param_names.index(param.name)
                    uncertain_param_idxs = (param_idx, param_idx)
                    dr_var_value = term.args[1].value
                    quadratic_dr_coeffs[ssv_idx, param_idx, param_idx] = (
                        dr_var_value
                    )
                elif is_ssv_term:
                    ssv_in_eq = term.args[1]
                    assert ssv_in_eq is ssv
                else:
                    raise ValueError(
                        f"Could not categorize expression term {term}"
                        f"of DR equation {dr_eq}."
                    )

        self.constant_dr_coeffs = constant_dr_coeffs
        self.affine_dr_coeffs = affine_dr_coeffs
        self.quadratic_dr_coeffs = quadratic_dr_coeffs

    @property
    def degree(self):
        """
        Evaluate polynomial degree of decision rule
        contained in self.
        """
        quadratic_coeffs_all_zero = np.all(self.quadratic_dr_coeffs == 0)
        affine_coeffs_all_zero = np.all(self.affine_dr_coeffs == 0)

        if not quadratic_coeffs_all_zero:
            degree = 2
        else:
            degree = int(not affine_coeffs_all_zero)

        return degree

    def get_new_dr_vars(self, degree):
        """
        Construct new decision rule variables for a single
        DR equation.
        """
        from scipy.special import comb
        num_of_monomials = comb(
            N=self.num_uncertain_params + degree,
            k=degree,
            exact=True,
            repetition=False,
        )
        return [pyo.Var(range(num_of_monomials)) for _ in range(self.num_ssv)]

    def evaluate_dr_at(self, param_values):
        """
        Evaluate second-stage variable values
        according to decision rules.
        """
        param_vals_arr = np.array(param_values)
        return (
            self.constant_dr_coeffs
            + self.affine_dr_coeffs @ param_vals_arr
            + param_vals_arr @ self.quadratic_dr_coeffs @ param_vals_arr
        )

    def get_param_idx_to_coeff_map(self):
        """
        Get mapping from tuples of uncertain parameter indexes
        to corresponding DR coefficient values.
        """
        dr_map = {}
        for ssv_idx, const_coef in enumerate(self.constant_dr_coeffs):
            map_for_this_ssv = {}
            map_for_this_ssv[()] = const_coef
            map_for_this_ssv.update({
                (param_idx,): coeff
                for param_idx, coeff
                in enumerate(self.affine_dr_coeffs[ssv_idx])
            })

            for param_idx_1 in range(self.num_uncertain_params):
                for param_idx_2 in range(param_idx_1, self.num_uncertain_params):
                    if param_idx_1 == param_idx_2:
                        map_for_this_ssv[param_idx_1, param_idx_2] = 1 * (
                            self.quadratic_dr_coeffs[
                                ssv_idx, param_idx_1, param_idx_2
                            ]
                        )
                    else:
                        map_for_this_ssv[param_idx_1, param_idx_2] = 2 * (
                            self.quadratic_dr_coeffs[
                                ssv_idx, param_idx_1, param_idx_2
                            ]
                        )
            dr_map[ssv_idx] = map_for_this_ssv

        return dr_map

    def generate_dr_eqns(
            self,
            second_stage_vars,
            uncertain_params,
            # decision_rule_vars=None,  # support later
            dr_order=None,
            ):
        """
        Generate decision rule constraints.
        """
        # standardize and validate DR order argument
        if dr_order is None:
            dr_order = self.degree
        assert self.degree <= dr_order <= 2

        dr_eqns = pyo.ConstraintList()
        dr_eqns.construct()
        for ssv_idx, ssv in enumerate(second_stage_vars):
            # constant terms
            dr_expr = 1 * self.constant_dr_coeffs[ssv_idx]

            # affine terms
            if dr_order >= 1:
                for param, coeff in zip(uncertain_params, self.affine_dr_coeffs[ssv_idx]):
                    dr_expr += param * coeff

            # quadratic terms
            if dr_order >= 2:
                for p_idx_1, row in enumerate(self.quadratic_dr_coeffs[ssv_idx]):
                    for p_idx_2 in range(p_idx_1, self.num_uncertain_params):
                        if p_idx_1 == p_idx_2:
                            coeff_val = row[p_idx_2]
                            dr_expr += (
                                uncertain_params[p_idx_1] ** 2
                                * coeff_val
                            )
                        else:
                            coeff_val = 2 * row[p_idx_2]
                            dr_expr += (
                                uncertain_params[p_idx_1]
                                * uncertain_params[p_idx_2]
                                * coeff_val
                            )

            dr_eqns.add(dr_expr - ssv == 0)

        return dr_eqns
