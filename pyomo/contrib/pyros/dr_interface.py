"""
Interface for analyzing polynomial decision rule coefficients.
"""

import numpy as np
import scipy

from pyomo.core.base import ConstraintList, Param
from pyomo.core.util import prod


class DecisionRuleInterface:
    """
    Interface to the coefficients of a polynomial decision rule.

    Parameters
    ----------
    static_coeffs : (M,) numpy.ndarray
        Static decision rule coefficients,
        where `M` is the dimension of the full second-stage
        variable space.
    affine_coeffs : None or (M, N) numpy.ndarray
        Affine decision rule coefficients,
        where `N` is the number of uncertain parameters.
        Note: if `affine_coeffs` is None, then `quadratic_coeffs`
        must also be None.
    quadratic_coeffs : None or (M, N, N) numpy.ndarray
        Quadratic decision rule coefficients.
        Typically, this is symmetric with respect to axes 1 and 2.
    """

    def __init__(self, static_coeffs, affine_coeffs, quadratic_coeffs):
        self._static_coeffs = np.asarray(static_coeffs)
        self._affine_coeffs = (
            None if affine_coeffs is None else np.asarray(affine_coeffs)
        )
        self._quadratic_coeffs = (
            None if quadratic_coeffs is None else np.asarray(quadratic_coeffs)
        )

        if self.affine_coeffs is not None:
            if self.affine_coeffs.shape[0] != self.static_coeffs.size:
                raise ValueError(
                    "Array-like value for `affine_coeffs` should have same "
                    "size along axis 0 as there are entries in "
                    f"`static_coeffs` "
                    f"(got {self.affine_coeffs.shape[0]}, "
                    f"expected {self.second_stage_var_dim})"
                )
        if self.quadratic_coeffs is not None:
            if self.affine_coeffs is None:
                # can't have quadratic and not have affine
                raise ValueError(
                    "Argument `affine_coeffs` must be an array-like, "
                    "rather than None, if "
                    "`quadratic_coeffs` is an array-like"
                )
            if self.quadratic_coeffs.shape[0] != self.static_coeffs.size:
                raise ValueError(
                    "Size along axis 0 of `quadratic_coeffs` should match "
                    "length of `static_coeffs` "
                    f"(got {self.quadratic_coeffs.shape[0]}, "
                    f"expected {self.static_coeffs.size})"
                )
            if self.affine_coeffs.shape[1] != self.quadratic_coeffs.shape[1]:
                raise ValueError(
                    "Size along axis 1 of `quadratic_coeffs` should match "
                    "that of `affine_coeffs` "
                    f"(got {self.quadratic_coeffs.shape[1]}, "
                    f"expected {self.affine_coeffs.shape[1]})"
                )
            if self.quadratic_coeffs.shape[1] != self.quadratic_coeffs.shape[2]:
                # must be square matrices
                raise ValueError(
                    "Sizes along axes 1 and 2 of `quadratic_coeffs` should match "
                    f"(axis 1 size is {self.quadratic_coeffs.shape[1]}, "
                    f"axis 2 size is {self.quadratic_coeffs.shape[2]})"
                )

    @property
    def static_coeffs(self):
        """
        (M,) numpy.ndarray : Static decision rule coefficients,
        where `M` is the dimension of the second-stage variable space.
        """
        return self._static_coeffs

    @property
    def affine_coeffs(self):
        """
        None or (M, N) numpy.ndarray : Affine decision rule coefficients,
        where `M` is the dimension of the second-stage variable space,
        and `N` is the dimension of the uncertain parameter space.
        """
        return self._affine_coeffs

    @property
    def quadratic_coeffs(self):
        """
        None or (M, N) numpy.ndarray : Quadratic decision rule
        coefficients,
        where `M` is the dimension of the second-stage variable space,
        and `N` is the dimension of the uncertain parameter space.
        """
        return self._quadratic_coeffs

    @property
    def order(self):
        """
        int : Order of the decision rule polynomial, according to
        whether there are affine or quadratic coefficients.
        """
        if self.affine_coeffs is None and self.quadratic_coeffs is None:
            return 0
        elif self.quadratic_coeffs is None:
            return 1
        else:
            return 2

    @property
    def second_stage_var_dim(self):
        """
        int : Dimension of the second-stage variable space,
        inferred from the size of ``self.static_coeffs``.
        """
        return self.static_coeffs.size

    @property
    def uncertain_param_dim(self):
        """
        int : Dimension of the uncertain parameter space,
        inferred from the size along axis 1 of ``self.affine_coeffs``.
        """
        return self.affine_coeffs.shape[1]

    def compute_polynomial_degrees(self, ss_idxs=None, tol=1e-10):
        """
        Compute degree of the decision rule polynomial specified by
        the coeffients contained in `self` for each specified
        dimension of the second-stage variable space.

        Parameters
        ----------
        ss_idxs : None or list of int, optional
            Dimensions in the second-stage variable space
            for which to compute the degrees.
            If `None` is passed, then all dimensions are considered.
        tol : float, optional
            Minimum value strictly beyond which
            absolute values of coefficients
            contained in self are considered nonzero.
            For quadratic coefficients that are not on the diagonals
            in axes 1 and 2, this tolerance is applied to the entries
            of the sum of the quadratic coefficients matrix
            and their tranpose in axes 1 and 2.

        Returns
        -------
        polynomial_degrees : numpy.ndarray
            Polynomial degrees in each of the specified second-stage
            variable space dimensions.
        """
        if ss_idxs is None:
            ss_idxs = list(range(self.second_stage_var_dim))
        is_static_coeff_nonzero = abs(self.static_coeffs[ss_idxs]) > tol

        is_affine_coeff_nonzero = np.full(len(ss_idxs), False)
        if self.order >= 1:
            is_affine_coeff_nonzero = (
                np.linalg.norm(self.affine_coeffs[ss_idxs], axis=-1, ord=float("inf"))
            ) > tol

        is_quadratic_coeff_nonzero = np.full(len(ss_idxs), False)
        if self.order == 2:
            quad_coeffs = self.quadratic_coeffs[ss_idxs]
            quad_diagonal = np.array(
                [np.diag(diag) for diag in np.diagonal(quad_coeffs, axis1=1, axis2=2)]
            )
            adjusted_quad_coeffs = (
                quad_coeffs + np.swapaxes(quad_coeffs, 1, 2) - quad_diagonal
            )
            is_quadratic_coeff_nonzero = (
                np.linalg.norm(
                    np.linalg.norm(adjusted_quad_coeffs, axis=-1, ord=float("inf")),
                    axis=-1,
                    ord=float("inf"),
                )
                > tol
            )

        polynomial_degrees = np.full(len(ss_idxs), -1, dtype=int)
        polynomial_degrees[is_quadratic_coeff_nonzero] = 2
        polynomial_degrees[
            np.logical_and(is_affine_coeff_nonzero, ~is_quadratic_coeff_nonzero)
        ] = 1
        polynomial_degrees[
            np.logical_and(
                is_static_coeff_nonzero,
                ~np.logical_or(is_affine_coeff_nonzero, is_quadratic_coeff_nonzero),
            )
        ] = 0
        return polynomial_degrees

    def get_num_coeffs_per_ss_dim(self, simplified=True):
        """
        Get number of coefficients in the decision rule polynomial
        for each second-stage variable.

        Parameters
        ----------
        simplified : bool, optional
            True if the quadratic coefficients are to be simplified,
            i.e., coefficient `(i, j)` is to be merged into coefficient
            `(j, i)`, False otherwise.

        Returns
        -------
        int
            Number of coefficients.
        """
        if self.order == 0:
            return 1

        num_uncertain_params = self.uncertain_param_dim
        if simplified:
            return scipy.special.comb(
                N=num_uncertain_params + self.order,
                k=self.order,
                exact=True,
                repetition=False,
            )
        else:
            return sum(num_uncertain_params**order for order in range(self.order + 1))

    def get_param_idx_to_coeff_map(self, ss_idx, simplified=True):
        """
        Cast the coefficients contained in `self` to a map
        from the dimension indices of the uncertain parameter
        space to the corresponding coefficients.

        Parameters
        ----------
        simplified : bool, optional
            True if the quadratic coefficients are to be simplified,
            i.e., coefficient `(i, j)` is to be merged into coefficient
            `(j, i)`, False otherwise.

        Returns
        -------
        dict
            Each entry maps a tuple of the parameter dimension
            indices to the coefficient of the monomial obtained
            by multiplying the uncertain parameters corresponding
            to those indices.
        """
        coeff_to_idx_map = {(): self.static_coeffs[ss_idx]}
        if self.order >= 1:
            coeff_to_idx_map.update(
                {
                    (param_idx,): val
                    for param_idx, val in enumerate(self.affine_coeffs[ss_idx])
                }
            )
        if self.order == 2:
            if simplified:
                quad_coeffs_simp = self.quadratic_coeffs[ss_idx].copy()
                quad_coeffs_simp += np.tril(quad_coeffs_simp, k=-1).T
                coeff_to_idx_map.update(
                    {
                        (idx1, idx2): quad_coeffs_simp[idx1, idx2]
                        for (idx1, idx2) in zip(*np.triu_indices_from(quad_coeffs_simp))
                    }
                )
            else:
                coeff_to_idx_map.update(
                    {
                        (idx1, idx2): val
                        for (idx1, idx2), val in np.ndenumerate(
                            self.quadratic_coeffs[ss_idx]
                        )
                    }
                )

        return coeff_to_idx_map

    def _set_dr_component_values(self, dr_component, ss_idx, simplified=True):
        dr_cdata_list = list(dr_component.values())
        expected_num_dr_coeffs = self.get_num_coeffs_per_ss_dim(simplified=simplified)
        if len(dr_cdata_list) != expected_num_dr_coeffs:
            raise ValueError(
                "Length of `dr_component` should match "
                "expected number of DR components per dimension"
                f"(got {len(dr_cdata_list)}, expected {expected_num_dr_coeffs})"
            )

        idx_to_coeff_map = self.get_param_idx_to_coeff_map(
            ss_idx, simplified=simplified
        )
        for dr_var, (idxs, coeff) in zip(dr_cdata_list, idx_to_coeff_map.items()):
            dr_var.set_value(coeff)

    def set_dr_component_values(self, dr_components, simplified=True):
        """
        Set values of Pyomo indexed components representing the
        DR coefficients.

        Parameters
        ----------
        dr_components : list of IndexedComponent
            Components of which to set values.
            The list must be of length equal to
            ``self.second_stage_var_dim``.
        simplified : bool
            True if the quadratic coefficients are to be simplified,
            i.e., coefficient `(i, j)` is to be merged into coefficient
            `(j, i)`, False otherwise.
        """
        if len(dr_components) != self.second_stage_var_dim:
            raise ValueError(
                "Length of `dr_components` should match "
                "second-stage dimension "
                f"(got {len(dr_components)}, expected {self.second_stage_var_dim})"
            )
        for ss_idx, indexed_comp in enumerate(dr_components):
            self._set_dr_component_values(
                indexed_comp, ss_idx=ss_idx, simplified=simplified
            )

    def setup_dr_components(self, ctype=Param, simplified=True, set_values=True):
        """
        Construct a list of indexed Pyomo component objects
        for representing the decision rule coefficients.

        Parameters
        ----------
        ctype : type, optional
            Pyomo Component type, such as ``Param`` or ``Var``.
            If ``Param`` is passed, then the components are
            instantiated with argument ``mutable=True``.
        simplified : bool, optional
            True if the quadratic coefficients are to be simplified,
            i.e., coefficient `(i, j)` is to be merged into coefficient
            `(j, i)`, False otherwise.
        set_values : bool, optional
            True if the values of the instantiated components
            are to be set according to the coefficient values contained
            in ``self``, False otherwise.

        Returns
        -------
        list of IndexedComponent
            The components of interest.
            Each entry is of type `ctype`.
        """
        num_dr_vars = self.get_num_coeffs_per_ss_dim(simplified=simplified)
        indexed_comp_list = []
        ctype_kwargs = dict(mutable=True) if ctype is Param else dict()
        for ss_idx in range(self.second_stage_var_dim):
            indexed_comp = ctype(list(range(num_dr_vars)), initialize=0, **ctype_kwargs)
            indexed_comp.construct()
            indexed_comp_list.append(indexed_comp)
        if set_values:
            self.set_dr_component_values(
                dr_components=indexed_comp_list, simplified=simplified
            )

        return indexed_comp_list

    def generate_dr_exprs(self, dr_components, uncertain_params, simplified=True):
        """
        Generate Pyomo expressions of the decision rule polynomials.

        Parameters
        ----------
        dr_components : list of IndexedComponent
            Components representing the polynomial decision rules.
            Should be of length ``self.second_stage_var_dim``.
        uncertain_params : list of ParamData
            Components representing the uncertain parameters.
            Should be of length ``self.uncertain_param_dim``.
        simplified : bool, optional
            True if the quadratic coefficients are to be simplified,
            i.e., coefficient `(i, j)` is to be merged into coefficient
            `(j, i)`, False otherwise.

        Returns
        -------
        list of NumericExpression
            The polynomial expressions of interest.
        """
        if self.order > 0 and len(uncertain_params) != self.uncertain_param_dim:
            raise ValueError(
                "Length of `uncertain_params` should match "
                "length along axis 1 of `self.affine_coeffs` "
                f"(got {len(uncertain_params)}, expected {self.uncertain_param_dim})"
            )
        if len(dr_components) != self.second_stage_var_dim:
            raise ValueError(
                "Length of `dr_components` should match "
                "second-stage dimension "
                f"(got {len(dr_components)}, expected {self.second_stage_var_dim})"
            )
        dr_expr_list = []
        expected_num_dr_coeffs = self.get_num_coeffs_per_ss_dim(simplified=simplified)
        for ss_idx, indexed_dr_comp in enumerate(dr_components):
            if len(indexed_dr_comp) != expected_num_dr_coeffs:
                raise ValueError(
                    "Length of `dr_component` should match "
                    "expected number of DR components per dimension"
                    f"(got {len(indexed_dr_comp)}, expected {expected_num_dr_coeffs})"
                )
            param_idx_to_coeff_map = self.get_param_idx_to_coeff_map(
                simplified=simplified, ss_idx=ss_idx
            )
            dr_zip = zip(param_idx_to_coeff_map.items(), indexed_dr_comp.values())
            dr_expr = 0
            for (param_idxs, coeff), cdata in dr_zip:
                param_combo = [uncertain_params[pidx] for pidx in param_idxs]
                dr_expr += prod(param_combo) * cdata
            dr_expr_list.append(dr_expr)

        return dr_expr_list

    def generate_dr_equations(
        self, dr_components, second_stage_variables, uncertain_params, simplified=True
    ):
        """
        Generate Pyomo equality constraints restricting
        the second-stage variables of a model to the corresponding
        decision rule polynomials.

        Parameters
        ----------
        dr_components : list of IndexedComponent
            Components representing the polynomial decision rules.
            Should be of length ``self.second_stage_var_dim``.
        second_stage_variables : list of VarData
            Components representing the second-stage variables.
            Should be of length ``self.second_stage_var_dim``.
        uncertain_params : list of ParamData
            Components representing the uncertain parameters.
            Should be of length ``self.uncertain_param_dim``.
        simplified : bool, optional
            True if the quadratic coefficients are to be simplified,
            i.e., coefficient `(i, j)` is to be merged into coefficient
            `(j, i)`, False otherwise.

        Returns
        -------
        conlist : ConstraintList
            Decision rule constraints.
        """
        if len(second_stage_variables) != len(dr_components):
            raise ValueError(
                "Length of `second_stage_variables` should match that of "
                "`dr_components` "
                f"(got {len(second_stage_variables)}, expected {len(dr_components)})"
            )
        dr_exprs = self.generate_dr_exprs(
            dr_components=dr_components,
            uncertain_params=uncertain_params,
            simplified=simplified,
        )
        dr_conlist = ConstraintList()
        dr_conlist.construct()
        for expr, ss_var in zip(dr_exprs, second_stage_variables):
            dr_conlist.add(expr - ss_var == 0)

        return dr_conlist
