"""
Functions for handling the construction and solving of the GRCS master problem via ROSolver
"""
from pyomo.core.base import (
    ConcreteModel,
    Block,
    Var,
    Objective,
    Constraint,
    ConstraintList,
    SortComponents,
)
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverResults
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (
    selective_clone,
    ObjectiveType,
    pyrosTerminationCondition,
    process_termination_condition_master_problem,
    adjust_solver_time_settings,
    revert_solver_max_time_adjustment,
    get_main_elapsed_time,
)
from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name

from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import TIC_TOC_SOLVE_TIME_ATTR
from pyomo.contrib.pyros.util import enforce_dr_degree


def initial_construct_master(model_data):
    """
    Constructs the iteration 0 master problem
    return: a MasterProblemData object containing the master_model object
    """
    m = ConcreteModel()
    m.scenarios = Block(NonNegativeIntegers, NonNegativeIntegers)

    master_data = MasterProblemData()
    master_data.original = model_data.working_model.clone()
    master_data.master_model = m
    master_data.timing = model_data.timing

    return master_data


def get_state_vars(model, iterations):
    """
    Obtain the state variables of a two-stage model
    for a given (sequence of) iterations corresponding
    to model blocks.

    Parameters
    ----------
    model : ConcreteModel
        PyROS model.
    iterations : iterable
        Iterations to consider.

    Returns
    -------
    iter_state_var_map : dict
        Mapping from iterations to list(s) of state vars.
    """
    iter_state_var_map = dict()
    for itn in iterations:
        state_vars = [
            var for blk in model.scenarios[itn, :] for var in blk.util.state_vars
        ]
        iter_state_var_map[itn] = state_vars

    return iter_state_var_map


def construct_master_feasibility_problem(model_data, config):
    """
    Construct a slack-variable based master feasibility model.
    Initialize all model variables appropriately, and scale slack variables
    as well.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver config.

    Returns
    -------
    model : ConcreteModel
        Slack variable model.
    """

    # clone master model. current state:
    # - variables for all but newest block are set to values from
    #   master solution from previous iteration
    # - variables for newest block are set to values from separation
    #   solution chosen in previous iteration
    model = model_data.master_model.clone()

    # obtain mapping from master problem to master feasibility
    # problem variables
    varmap_name = unique_component_name(model_data.master_model, 'pyros_var_map')
    setattr(
        model_data.master_model,
        varmap_name,
        list(model_data.master_model.component_data_objects(Var)),
    )
    model = model_data.master_model.clone()
    model_data.feasibility_problem_varmap = list(
        zip(getattr(model_data.master_model, varmap_name), getattr(model, varmap_name))
    )
    delattr(model_data.master_model, varmap_name)
    delattr(model, varmap_name)

    for obj in model.component_data_objects(Objective):
        obj.deactivate()
    iteration = model_data.iteration

    # add slacks only to inequality constraints for the newest
    # master block. these should be the only constraints which
    # may have been violated by the previous master and separation
    # solution(s)
    targets = []
    for blk in model.scenarios[iteration, :]:
        targets.extend([
            con
            for con in blk.component_data_objects(
                Constraint, active=True, descend_into=True
            )
            if not con.equality
        ])

    # retain original constraint expressions
    # (for slack initialization and scaling)
    pre_slack_con_exprs = ComponentMap((con, con.body - con.upper) for con in targets)

    # add slack variables and objective
    # inequalities g(v) <= b become g(v) -- s^- <= b
    TransformationFactory("core.add_slack_variables").apply_to(model, targets=targets)
    slack_vars = ComponentSet(
        model._core_add_slack_variables.component_data_objects(Var, descend_into=True)
    )

    # initialize and scale slack variables
    for con in pre_slack_con_exprs:
        # get mapping from slack variables to their (linear)
        # coefficients (+/-1) in the updated constraint expressions
        repn = generate_standard_repn(con.body)
        slack_var_coef_map = ComponentMap()
        for idx in range(len(repn.linear_vars)):
            var = repn.linear_vars[idx]
            if var in slack_vars:
                slack_var_coef_map[var] = repn.linear_coefs[idx]

        slack_substitution_map = dict()
        for slack_var in slack_var_coef_map:
            # coefficient determines whether the slack
            # is a +ve or -ve slack
            if slack_var_coef_map[slack_var] == -1:
                con_slack = max(0, value(pre_slack_con_exprs[con]))
            else:
                con_slack = max(0, -value(pre_slack_con_exprs[con]))

            # initialize slack variable, evaluate scaling coefficient
            slack_var.set_value(con_slack)
            scaling_coeff = 1

            # update expression replacement map for slack scaling
            slack_substitution_map[id(slack_var)] = scaling_coeff * slack_var

        # finally, scale slack(s)
        con.set_value(
            (
                replace_expressions(con.lower, slack_substitution_map),
                replace_expressions(con.body, slack_substitution_map),
                replace_expressions(con.upper, slack_substitution_map),
            )
        )

    return model


def solve_master_feasibility_problem(model_data, config):
    """
    Solve a slack variable-based feasibility model derived
    from the master problem. Initialize the master problem
    to the  solution found by the optimizer if solved successfully,
    or to the initial point provided to the solver otherwise.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Solver results.
    """
    model = construct_master_feasibility_problem(model_data, config)

    active_obj = next(model.component_data_objects(Objective, active=True))
    config.progress_logger.debug("Solving master feasibility problem")
    config.progress_logger.debug(
        f" Initial objective (total slack): {value(active_obj)}"
    )

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    timer = TicTocTimer()
    orig_setting, custom_setting_present = adjust_solver_time_settings(
        model_data.timing, solver, config
    )
    model_data.timing.start_timer("main.master_feasibility")
    timer.tic(msg=None)
    try:
        results = solver.solve(model, tee=config.tee, load_solutions=False)
    except ApplicationError:
        # account for possible external subsolver errors
        # (such as segmentation faults, function evaluation
        # errors, etc.)
        config.progress_logger.error(
            f"Optimizer {repr(solver)} encountered exception "
            "attempting to solve master feasibility problem in iteration "
            f"{model_data.iteration}"
        )
        raise
    else:
        setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
        model_data.timing.stop_timer("main.master_feasibility")
    finally:
        revert_solver_max_time_adjustment(
            solver, orig_setting, custom_setting_present, config
        )

    feasible_terminations = {
        tc.optimal,
        tc.locallyOptimal,
        tc.globallyOptimal,
        tc.feasible,
    }
    if results.solver.termination_condition in feasible_terminations:
        model.solutions.load_from(results)
        config.progress_logger.debug(
            f" Final objective (total slack): {value(active_obj)}"
        )
        config.progress_logger.debug(
            f" Termination condition: {results.solver.termination_condition}"
        )
        config.progress_logger.debug(
            f" Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s"
        )
    else:
        config.progress_logger.warning(
            "Could not successfully solve master feasibility problem "
            f"of iteration {model_data.iteration} with primary subordinate "
            f"{'global' if config.solve_master_globally else 'local'} solver "
            "to acceptable level. "
            f"Termination stats:\n{results.solver}\n"
            "Maintaining unoptimized point for master problem initialization."
        )

    # load master feasibility point to master model
    for master_var, feas_var in model_data.feasibility_problem_varmap:
        master_var.set_value(feas_var.value, skip_validation=True)

    return results


def enforce_dr_polishing_efficiencies(
        polishing_model,
        config,
        ):
    """
    Enforce decision rule variable efficiencies
    in DR polishing model.
    """
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    # add efficiency for "free" DR variables
    from pyomo.core.util import prod
    import numpy as np
    DR_POLISHING_EFFICIENCY_TOL = 1e-10
    num_ssv = len(nominal_polishing_block.util.second_stage_variables)

    # nested iterable of second-stage variables.
    # level 1: second-stage variable
    # level 2: master problem block
    all_ssv_duplicates = (
        [
            blk.util.second_stage_variables[idx]
            for blk in polishing_model.scenarios.values()
        ]
        for idx in range(num_ssv)
    )
    # nested iterable of decision rule equations, similar structure
    all_dr_eq_duplicates = (
        [
            blk.util.decision_rule_eqns[idx]
            for blk in polishing_model.scenarios.values()
        ]
        for idx in range(num_ssv)
    )
    duplicates_zip = zip(
        nominal_polishing_block.util.decision_rule_vars,
        all_ssv_duplicates,
        all_dr_eq_duplicates,
    )
    for indexed_dr_var, ssv_copies, dr_eq_copies in duplicates_zip:
        # for each scenario block, evaluate
        # product of uncertain parameters in each monomial
        dr_monomial_param_product_vals = []
        for idx, dr_eq in enumerate(dr_eq_copies):
            dr_monomial_param_product_vals.append([
                value(prod(monomial.args[:-1]))
                for monomial in dr_eq.body.args[:-1]
            ])

        # get decision rule variables participating in each
        # monomial for the first block.
        # assumes DR polynomial structure is independent of
        # block, since blocks were created by cloning
        dr_vars_by_monomial_order = np.array([
            monomial.args[-1]
            for monomial in dr_eq_copies[0].body.args[:-1]
        ])

        # cast to array; flip levels. thus:
        # - row corresponds to monomial of interest
        # - column corresponds to scenario block
        dr_monomial_param_product_vals_arr = np.array(
            dr_monomial_param_product_vals
        )

        # evaluate value of second-stage variable copy for each
        # scenario block
        # ssv_val_per_scenario = np.array([value(ssv) for ssv in ssv_copies])

        # for each scenario block,
        # we want to check how uncertain parameter products
        # in each monomial compare to the second-stage
        # variable value
        rel_dr_monomial_copy_vals = abs(
            dr_monomial_param_product_vals_arr
            # / ssv_val_per_scenario[:, np.newaxis]
        )

        # now check: for each DR monomial, is uncertain
        # parameter product relative to second-stage variable
        # value smaller than tolerance for all scenarios?
        # if so, DR variable is considered to be free, so we fix
        # value to 0
        max_param_product_vals = np.max(rel_dr_monomial_copy_vals, axis=0)
        free_dr_vars = dr_vars_by_monomial_order[
            max_param_product_vals < DR_POLISHING_EFFICIENCY_TOL
        ]
        for var in free_dr_vars:
            var.fix(0)


def get_dr_vars_for_norm(polishing_model, config):
    """
    For each second-stage variable, get DR coefficients
    to be considered in polishing objective.
    """
    nominal_polishing_block = polishing_model.scenarios[0, 0]
    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars
    include_static_term_in_norm = config.dr_polishing_options[
        "include_static_term_in_norm"
    ]

    # for each DR equation, get list of nonstatic DR variables
    norm_idx_to_dr_var_maps = {}
    for idx, indexed_dr_var in enumerate(decision_rule_vars):
        norm_idx_to_dr_var_maps[idx] = {}
        for dr_var_idx, dr_var in indexed_dr_var.items():
            monomial_degree = (
                nominal_polishing_block.util.dr_var_to_exponent_map[dr_var]
            )
            include_term_in_norm = (
                include_static_term_in_norm or monomial_degree > 0
            )
            if include_term_in_norm:
                norm_idx_to_dr_var_maps[idx][dr_var_idx] = dr_var

    return norm_idx_to_dr_var_maps


def add_1_norm_polishing_components(polishing_model, config):
    """
    Add DR polishing components for minimization of 1-norm
    of the DR polynomial terms.
    """
    # for each second-stage variable, get DR variables to be considered
    # in polishing objective
    norm_idx_to_dr_var_maps = get_dr_vars_for_norm(polishing_model, config)

    nominal_polishing_block = polishing_model.scenarios[0, 0]
    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars
    nominal_polishing_block.util.polishing_vars = polishing_vars = []

    # add polishing variables
    for idx, _ in enumerate(decision_rule_vars):
        indexed_polishing_var = Var(
            list(norm_idx_to_dr_var_maps[idx].keys()),
            domain=NonNegativeReals,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"dr_polishing_var_{idx}"
            ),
            indexed_polishing_var,
        )
        polishing_vars.append(indexed_polishing_var)

    # set up scaling factors for polishing objective per
    # second-stage variable
    by_z_for_scaling = {
        "z_nom": True,
        "none": False,
    }
    second_stage_scaling_factors = get_second_stage_scaling_factors(
        second_stage_variables=nominal_polishing_block.util.second_stage_variables,
        by_z=by_z_for_scaling[config.dr_polishing_options["polishing_norm_scaling"]],
    )

    # add polishing constraints
    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        polishing_vars,
        nominal_polishing_block.util.second_stage_variables,
    )
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    for idx, (dr_eq, indexed_polishing_var, ss_var) in enumerate(dr_eq_var_zip):
        # set up polishing constraint components
        polishing_absolute_value_lb_cons = Constraint(
            indexed_polishing_var.index_set(),
        )
        polishing_absolute_value_ub_cons = Constraint(
            indexed_polishing_var.index_set(),
        )

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )

        # update list attributes
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        # get monomials of the nominal DR expression,
        # ensuring second-stage variable is not included
        dr_expr_terms = dr_eq.body.args[:-1]

        for dr_monomial in dr_expr_terms:
            # retrieve DR variable in monomial
            dr_var_in_term = dr_monomial.args[-1]
            idx_of_dr_var_in_term = dr_var_in_term.index()
            if idx_of_dr_var_in_term not in norm_idx_to_dr_var_maps[idx]:
                continue

            # get corresponding polishing variable
            polishing_var = indexed_polishing_var[idx_of_dr_var_in_term]

            # add polishing constraints
            scale_factor = second_stage_scaling_factors[ss_var]
            polishing_absolute_value_lb_cons[idx_of_dr_var_in_term] = (
                -polishing_var - scale_factor * dr_monomial <= 0
            )
            polishing_absolute_value_ub_cons[idx_of_dr_var_in_term] = (
                scale_factor * dr_monomial - polishing_var <= 0
            )

            # if DR var is fixed, then fix polishing variable as well.
            # also, deactivate coresponding polishing constraints
            if dr_var_in_term.fixed:
                polishing_var.fix()
                polishing_absolute_value_lb_cons[idx_of_dr_var_in_term].deactivate()
                polishing_absolute_value_ub_cons[idx_of_dr_var_in_term].deactivate()

            # initialize auxiliary polishing variable
            polishing_var.set_value(abs(value(scale_factor * dr_monomial)))

    # finally, declare polishing objective
    polishing_model.polishing_obj = Objective(
        expr=sum(
            sum(polishing_var.values())
            for polishing_var in polishing_vars
        )
    )


def add_inf_norm_polishing_components(polishing_model, config):
    """
    Add DR polishing components for minimization of infinity-norm
    of the DR polynomial terms.
    """
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    # for each second-stage variable, get DR variables to be considered
    # in polishing objective
    norm_idx_to_dr_var_maps = get_dr_vars_for_norm(polishing_model, config)

    # declare variable for representing infinity norm
    inf_norm_var = polishing_model.inf_norm_var = Var(
        initialize=0,  # will be updated later
        domain=NonNegativeReals,
    )

    # set up scaling factors for polishing objective per
    # second-stage variable
    by_z_for_scaling = {
        "z_nom": True,
        "none": False,
    }
    second_stage_scaling_factors = get_second_stage_scaling_factors(
        second_stage_variables=nominal_polishing_block.util.second_stage_variables,
        by_z=by_z_for_scaling[config.dr_polishing_options["polishing_norm_scaling"]],
    )

    # now add polishing constraints
    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        nominal_polishing_block.util.second_stage_variables,
    )
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    for idx, (dr_eq, ss_var) in enumerate(dr_eq_var_zip):
        polishing_absolute_value_lb_cons = Constraint(
            list(norm_idx_to_dr_var_maps[idx].keys()),
        )
        polishing_absolute_value_ub_cons = Constraint(
            list(norm_idx_to_dr_var_maps[idx].keys()),
        )

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        # ensure second-stage variable term excluded
        dr_expr_terms = dr_eq.body.args[:-1]

        for dr_eq_term in dr_expr_terms:
            # extract DR var in term
            dr_var_in_term = dr_eq_term.args[-1]
            idx_of_dr_var_in_term = dr_var_in_term.index()
            if idx_of_dr_var_in_term not in norm_idx_to_dr_var_maps[idx]:
                continue

            # add polishing constraints
            scale_factor = second_stage_scaling_factors[ss_var]
            polishing_absolute_value_lb_cons[idx_of_dr_var_in_term] = (
                -inf_norm_var - scale_factor * dr_eq_term <= 0
            )
            polishing_absolute_value_ub_cons[idx_of_dr_var_in_term] = (
                scale_factor * dr_eq_term - inf_norm_var <= 0
            )

            # initialize/update value of infinity norm variable,
            # such that initial value is equal to initial infinity
            # norm of nonstatic DR terms
            inf_norm_var.set_value(max(
                value(inf_norm_var),
                abs(value(scale_factor * dr_eq_term)),
            ))

    # declare polishing objective
    polishing_model.polishing_obj = Objective(expr=inf_norm_var)


def add_sum_inf_norm_polishing_components(polishing_model, config):
    """
    Add DR polishing components for minimization of sum of
    the infinity-norm of the DR polynomial terms for each second-stage
    variable.
    """
    # for each second-stage variable, get DR variables to be considered
    # in polishing objective
    norm_idx_to_dr_var_maps = get_dr_vars_for_norm(polishing_model, config)

    nominal_polishing_block = polishing_model.scenarios[0, 0]
    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars

    # declare variable for representing infinity norm of
    # DR monomials for each second-stage variable
    polishing_model.infinity_norm_vars = Var(
        range(len(decision_rule_vars)),
        initialize=0,  # will be updated later
        domain=NonNegativeReals,
    )

    # set up scaling factors for polishing objective per
    # second-stage variable
    by_z_for_scaling = {
        "z_nom": True,
        "none": False,
    }
    second_stage_scaling_factors = get_second_stage_scaling_factors(
        second_stage_variables=nominal_polishing_block.util.second_stage_variables,
        by_z=by_z_for_scaling[config.dr_polishing_options["polishing_norm_scaling"]],
    )

    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        polishing_model.infinity_norm_vars.values(),
        nominal_polishing_block.util.second_stage_variables,
    )
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    for idx, (dr_eq, inf_norm_var, ss_var) in enumerate(dr_eq_var_zip):
        # components for absolute value and infinity norm constraints
        polishing_absolute_value_lb_cons = Constraint(
            list(norm_idx_to_dr_var_maps[idx].keys()),
        )
        polishing_absolute_value_ub_cons = Constraint(
            list(norm_idx_to_dr_var_maps[idx].keys()),
        )

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )

        # update list attributes
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        # get monomials of nominal DR expression,
        # ensuring second-stage variable is excluded
        dr_expr_terms = dr_eq.body.args[:-1]

        for dr_monomial in dr_expr_terms:
            # extract DR var in term
            dr_var_in_monomial = dr_monomial.args[-1]
            idx_of_dr_var = dr_var_in_monomial.index()
            if idx_of_dr_var not in norm_idx_to_dr_var_maps[idx]:
                continue

            # add polishing constraints
            scale_factor = second_stage_scaling_factors[ss_var]
            polishing_absolute_value_lb_cons[idx_of_dr_var] = (
                -inf_norm_var - scale_factor * dr_monomial <= 0
            )
            polishing_absolute_value_ub_cons[idx_of_dr_var] = (
                scale_factor * dr_monomial - inf_norm_var <= 0
            )

            # update initial value of infinity norm var, so
            # that it is set to infinity norm of initial DR
            # monomial values
            inf_norm_var.set_value(max(
                abs(value(scale_factor * dr_monomial)),
                value(inf_norm_var),
            ))

    # declare polishing objective
    polishing_model.polishing_obj = Objective(
        expr=sum(polishing_model.infinity_norm_vars.values()),
    )


def scale_decision_rule_eqns(polishing_model, config):
    """
    Scale decision rule equations, as desired.
    """
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    # get scaling factors based on nominal second-stage variable values
    by_z_map = {
        "none": False,
        "z_nom": True,
    }
    second_stage_scaling_factors = get_second_stage_scaling_factors(
        nominal_polishing_block.util.second_stage_variables,
        by_z=by_z_map[config.dr_polishing_options["dr_eq_scaling"]],
    )

    # scale decision rule equations as desired
    from itertools import chain
    dr_eq_scaling_zip = zip(
        nominal_polishing_block.util.second_stage_variables,
        *tuple(chain([
            blk.util.decision_rule_eqns
            for blk in polishing_model.scenarios.values()
        ])),
    )
    for nom_ss_var, *dr_eq_copies in dr_eq_scaling_zip:
        dr_eq_scale_factor = second_stage_scaling_factors[nom_ss_var]
        for dr_eq_copy in dr_eq_copies:
            dr_eq_copy.set_value((
                dr_eq_scale_factor * dr_eq_copy.lower,
                sum(dr_eq_scale_factor * arg for arg in dr_eq_copy.body.args),
                dr_eq_scale_factor * dr_eq_copy.upper,
            ))


def create_dr_polishing_nlp(model_data, config):
    """
    Create NLP formulation of the decision rule polishing problem,
    given master problem and its solution.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    polishing_model : ConcreteModel
        Polishing model. The specific formulation is based
        on ``config.dr_polishing_options``.
    """
    # get DR polishing options
    # clone master problem
    master_model = model_data.master_model
    polishing_model = master_model.clone()
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    # fix first-stage variables
    first_stage_vars = nominal_polishing_block.util.first_stage_variables
    for var in first_stage_vars:
        var.fix()

    # enforce optimality of the first-stage and DR variables
    polishing_model.obj.deactivate()
    if config.objective_focus == ObjectiveType.worst_case:
        polishing_model.zeta.fix()
    else:
        optimal_master_obj_value = value(polishing_model.obj)
        polishing_model.nominal_optimality_con = Constraint(
            expr=(
                nominal_polishing_block.first_stage_objective
                + nominal_polishing_block.second_stage_objective
                <= optimal_master_obj_value
            ),
        )

    # enforce DR effiencies according to iteration number
    # TODO: shouldn't have to do this here, ensure this is called before
    #       master problem is solved
    enforce_dr_degree(
        blk=polishing_model.scenarios[0, 0],
        config=config,
        degree=get_master_dr_degree(model_data, config),
    )

    # enforce additional DR efficiencies, as desired
    include_extra_dr_efficiency = (
        config.dr_polishing_options["include_extra_dr_efficiency"]
    )
    if include_extra_dr_efficiency:
        enforce_dr_polishing_efficiencies(polishing_model, config)

    # add polishing-specific components
    # TODO: make whether or not norm should consider static
    #       term an optional argument to each of these
    #       functions
    polishing_norm = config.dr_polishing_options["polishing_norm"]
    polishing_component_func_map = {
        "1_norm": add_1_norm_polishing_components,
        "inf_norm": add_inf_norm_polishing_components,
        "sum_inf_norms": add_sum_inf_norm_polishing_components,
    }
    polishing_component_func = polishing_component_func_map[polishing_norm]
    polishing_component_func(polishing_model, config)

    # scale DR equations as desired
    scale_decision_rule_eqns(polishing_model, config)

    # DEBUGGING CHECKS
    # unused_vars = [
    #     var for var in polishing_model.component_data_objects(Var)
    #     if not var.fixed and all(
    #         var not in ComponentSet(identify_variables(con.body))
    #         for con in polishing_model.component_data_objects(Constraint, active=True)
    #     )
    # ]
    # for var in unused_vars:
    #     print(var.name)
    # unused_cons = [
    #     con
    #     for con in polishing_model.component_data_objects(Constraint, active=True)
    #     if all(var.fixed for var in ComponentSet(identify_variables(con.body)))
    # ]
    # for con in unused_cons:
    #     print(con.name)
    # import pdb
    # pdb.set_trace()

    return polishing_model


def minimize_dr_vars_nlp(model_data, config):
    """
    Polish the PyROS decision rule determined for the most
    recently solved master problem by minimizing the collective
    L1 norm of the vector of all decision rule variables.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Subordinate solver results for the polishing problem.
    polishing_successful : bool
        True if polishing model was solved to acceptable level,
        False otherwise.
    """
    # create polishing NLP
    polishing_model = create_dr_polishing_nlp(
        model_data=model_data,
        config=config,
    )

    # from pyomo.contrib.pyros.util import SolverWithBackup

    if config.solve_master_globally:
        solvers = [config.global_solver] + config.backup_global_solvers
    else:
        solvers = [config.local_solver] + config.backup_local_solvers
    solver = solvers[0]

    config.progress_logger.debug("Solving DR polishing problem")

    # NOTE: this objective evalaution may not be accurate, due
    #       to the current initialization scheme for the auxiliary
    #       variables. new initialization will be implemented in the
    #       near future.
    polishing_obj = polishing_model.polishing_obj
    config.progress_logger.debug(f" Initial DR norm: {value(polishing_obj)}")

    # === Solve the polishing model
    timer = TicTocTimer()
    orig_setting, custom_setting_present = adjust_solver_time_settings(
        model_data.timing, solver, config
    )
    model_data.timing.start_timer("main.dr_polishing")
    timer.tic(msg=None)
    try:
        results = solver.solve(polishing_model, tee=config.tee, load_solutions=False)
    except ApplicationError:
        config.progress_logger.error(
            f"Optimizer {repr(solver)} encountered an exception "
            "attempting to solve decision rule polishing problem "
            f"in iteration {model_data.iteration}"
        )
        raise
    else:
        setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
        model_data.timing.stop_timer("main.dr_polishing")
    finally:
        revert_solver_max_time_adjustment(
            solver, orig_setting, custom_setting_present, config
        )

    # interested in the time and termination status for debugging
    # purposes
    config.progress_logger.debug(" Done solving DR polishing problem")
    config.progress_logger.debug(
        f"  Termination condition: {results.solver.termination_condition} "
    )
    config.progress_logger.debug(
        f"  Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)} s"
    )

    # === Process solution by termination condition
    acceptable = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal}
    if results.solver.termination_condition not in acceptable:
        # continue with "unpolished" master model solution
        return results, False
    # update master model second-stage, state, and decision rule
    # variables to polishing model solution
    polishing_model.solutions.load_from(results)

    for idx, blk in model_data.master_model.scenarios.items():
        ssv_zip = zip(
            blk.util.second_stage_variables,
            polishing_model.scenarios[idx].util.second_stage_variables,
        )
        sv_zip = zip(
            blk.util.state_vars, polishing_model.scenarios[idx].util.state_vars
        )
        for master_ssv, polish_ssv in ssv_zip:
            master_ssv.set_value(value(polish_ssv))
        for master_sv, polish_sv in sv_zip:
            master_sv.set_value(value(polish_sv))

        # update master problem decision rule variables
        dr_var_zip = zip(
            blk.util.decision_rule_vars,
            polishing_model.scenarios[idx].util.decision_rule_vars,
        )
        for master_dr, polish_dr in dr_var_zip:
            for mvar, pvar in zip(master_dr.values(), polish_dr.values()):
                mvar.set_value(value(pvar), skip_validation=True)

    config.progress_logger.debug(f" Optimized DR norm: {value(polishing_obj)}")
    config.progress_logger.debug(" Polished master objective:")

    # print master solution
    if config.objective_focus == ObjectiveType.worst_case:
        worst_blk_idx = max(
            model_data.master_model.scenarios.keys(),
            key=lambda idx: value(
                model_data.master_model.scenarios[idx].second_stage_objective
            ),
        )
    else:
        worst_blk_idx = (0, 0)

    # debugging: summarize objective breakdown
    worst_master_blk = model_data.master_model.scenarios[worst_blk_idx]
    config.progress_logger.debug(
        "  First-stage objective " f"{value(worst_master_blk.first_stage_objective)}"
    )
    config.progress_logger.debug(
        "  Second-stage objective " f"{value(worst_master_blk.second_stage_objective)}"
    )
    polished_master_obj = value(
        worst_master_blk.first_stage_objective + worst_master_blk.second_stage_objective
    )
    config.progress_logger.debug(f"  Objective {polished_master_obj}")

    return results, True


def get_second_stage_scaling_factors(second_stage_variables, by_z):
    """
    Get second-stage scaling factors.
    """
    if by_z:
        return ComponentMap(
            (var, 1 / max(1, abs(value(var))))
            for var in second_stage_variables
        )
    else:
        return ComponentMap((var, 1) for var in second_stage_variables)


def add_1_norm_lps_polishing_components(polishing_model, config):
    """
    Add components for DR polishing LPs: 1-norm.
    """
    # for each second-stage variable, get DR variables to be considered
    # in polishing objective
    norm_idx_to_dr_var_maps = get_dr_vars_for_norm(polishing_model, config)

    nominal_polishing_block = polishing_model.scenarios[0, 0]
    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars

    # declare polishing variables
    nominal_polishing_block.util.polishing_vars = polishing_vars = []
    for idx, _ in enumerate(decision_rule_vars):
        indexed_polishing_var = Var(
            list(norm_idx_to_dr_var_maps[idx].keys()),
            domain=NonNegativeReals,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"dr_polishing_var_{idx}"
            ),
            indexed_polishing_var,
        )
        polishing_vars.append(indexed_polishing_var)

    # initialize list attributes for keeping track of model components
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    nominal_polishing_block.util.unfixed_dr_vars = []
    nominal_polishing_block.util.unfixed_polishing_vars = []
    nominal_polishing_block.util.active_polishing_lb_cons = []
    nominal_polishing_block.util.active_polishing_ub_cons = []

    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        polishing_vars,
        nominal_polishing_block.util.second_stage_variables,
    )
    for idx, (dr_eq, indexed_polishing_var, ss_var) in enumerate(dr_eq_var_zip):
        polishing_absolute_value_lb_cons = Constraint(
            indexed_polishing_var.index_set(),
        )
        polishing_absolute_value_ub_cons = Constraint(
            indexed_polishing_var.index_set(),
        )

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )

        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        # keep track of unfixed DR variables and polishing components
        unfixed_dr_vars = []
        unfixed_polishing_vars = []
        active_polishing_lb_cons = []
        active_polishing_ub_cons = []

        # update list attributes
        nominal_polishing_block.util.unfixed_dr_vars.append(unfixed_dr_vars)
        nominal_polishing_block.util.unfixed_polishing_vars.append(unfixed_polishing_vars)
        nominal_polishing_block.util.active_polishing_lb_cons.append(
            active_polishing_lb_cons
        )
        nominal_polishing_block.util.active_polishing_ub_cons.append(
            active_polishing_ub_cons
        )

        # ensure second-stage variable term excluded
        dr_expr_terms = dr_eq.body.args[:-1]

        for dr_eq_term in dr_expr_terms:
            # extract DR var in term
            dr_var_in_term = dr_eq_term.args[-1]
            idx_of_dr_var_in_term = dr_var_in_term.index()
            if idx_of_dr_var_in_term not in norm_idx_to_dr_var_maps[idx]:
                if not dr_var_in_term.fixed:
                    unfixed_dr_vars.append(dr_var_in_term)
                continue

            # get corresponding polishing variable
            polishing_var = indexed_polishing_var[idx_of_dr_var_in_term]

            # add polishing constraints
            polishing_absolute_value_lb_cons[idx_of_dr_var_in_term] = (
                -polishing_var - dr_eq_term <= 0
            )
            polishing_absolute_value_ub_cons[idx_of_dr_var_in_term] = (
                dr_eq_term - polishing_var <= 0
            )

            # if DR var is fixed, then fix polishing variable as well.
            # also, deactivate coresponding polishing constraints
            if dr_var_in_term.fixed:
                polishing_var.fix()
                polishing_absolute_value_lb_cons[idx_of_dr_var_in_term].deactivate()
                polishing_absolute_value_ub_cons[idx_of_dr_var_in_term].deactivate()
            else:
                # take note of unfixed components
                unfixed_dr_vars.append(dr_var_in_term)
                unfixed_polishing_vars.append(polishing_var)
                active_polishing_lb_cons.append(
                    polishing_absolute_value_lb_cons[idx_of_dr_var_in_term]
                )
                active_polishing_ub_cons.append(
                    polishing_absolute_value_ub_cons[idx_of_dr_var_in_term]
                )

            # initialize auxiliary polishing variable
            polishing_var.set_value(abs(value(dr_eq_term)))

    # declare polishing objective
    @polishing_model.Objective(norm_idx_to_dr_var_maps)
    def polishing_obj(m, idx):
        return sum(polishing_vars[idx].values())


def add_inf_norm_lps_polishing_components(polishing_model, config):
    """
    Add components for DR polishing LPs: infinity-norm.
    """
    # for each second-stage variable, get DR variables to be considered
    # in polishing objective
    norm_idx_to_dr_var_maps = get_dr_vars_for_norm(polishing_model, config)

    nominal_polishing_block = polishing_model.scenarios[0, 0]
    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars

    # declare polishing variables
    nominal_polishing_block.util.polishing_vars = polishing_vars = []
    for idx, _ in enumerate(decision_rule_vars):
        indexed_polishing_var = Var(
            list(norm_idx_to_dr_var_maps[idx].keys()),
            domain=NonNegativeReals,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"dr_polishing_var_{idx}"
            ),
            indexed_polishing_var,
        )
        polishing_vars.append(indexed_polishing_var)

    polishing_model.infinity_norm_vars = Var(
        norm_idx_to_dr_var_maps,
        domain=NonNegativeReals,
        initialize=0,  # will be updated later
    )

    # initialize list attributes for keeping track of model components
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    nominal_polishing_block.util.unfixed_dr_vars = []
    nominal_polishing_block.util.unfixed_polishing_vars = []
    nominal_polishing_block.util.active_polishing_lb_cons = []
    nominal_polishing_block.util.active_polishing_ub_cons = []

    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        polishing_vars,
        nominal_polishing_block.util.second_stage_variables,
        polishing_model.infinity_norm_vars.values()
    )
    for idx, (dr_eq, indexed_polishing_var, ss_var, inf_norm_var) in enumerate(dr_eq_var_zip):
        polishing_absolute_value_lb_cons = Constraint(
            indexed_polishing_var.index_set(),
        )
        polishing_absolute_value_ub_cons = Constraint(
            indexed_polishing_var.index_set(),
        )

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                nominal_polishing_block,
                f"polishing_abs_val_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )

        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        # keep track of unfixed DR variables and polishing components
        unfixed_dr_vars = []
        unfixed_polishing_vars = [inf_norm_var]
        active_polishing_lb_cons = list(polishing_absolute_value_lb_cons.values())
        active_polishing_ub_cons = list(polishing_absolute_value_ub_cons.values())

        # update list attributes
        nominal_polishing_block.util.unfixed_dr_vars.append(unfixed_dr_vars)
        nominal_polishing_block.util.unfixed_polishing_vars.append(
            unfixed_polishing_vars
        )
        nominal_polishing_block.util.active_polishing_lb_cons.append(
            active_polishing_lb_cons
        )
        nominal_polishing_block.util.active_polishing_ub_cons.append(
            active_polishing_ub_cons
        )

        # ensure second-stage variable term excluded
        dr_expr_terms = dr_eq.body.args[:-1]

        for dr_eq_term in dr_expr_terms:
            # extract DR var in term
            dr_var_in_term = dr_eq_term.args[-1]
            idx_of_dr_var_in_term = dr_var_in_term.index()
            if idx_of_dr_var_in_term not in norm_idx_to_dr_var_maps[idx]:
                if not dr_var_in_term.fixed:
                    unfixed_dr_vars.append(dr_var_in_term)
                continue

            # add polishing constraints
            polishing_absolute_value_lb_cons[idx_of_dr_var_in_term] = (
                -inf_norm_var - dr_eq_term <= 0
            )
            polishing_absolute_value_ub_cons[idx_of_dr_var_in_term] = (
                dr_eq_term - inf_norm_var <= 0
            )

            # if DR var is fixed, then fix polishing variable as well.
            # also, deactivate coresponding polishing constraints
            if not dr_var_in_term.fixed:
                # take note of unfixed components
                unfixed_dr_vars.append(dr_var_in_term)

            # update infinity norm variable
            inf_norm_var.set_value(
                max(value(inf_norm_var), abs(value(dr_eq_term)))
            )

    # declare polishing objective
    @polishing_model.Objective(norm_idx_to_dr_var_maps)
    def polishing_obj(m, idx):
        return m.infinity_norm_vars[idx]


def create_dr_polishing_lp_problem(model_data, config):
    """
    Create model for DR polishing LPs.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    polishing_model : ConcreteModel
        Polishing model.
    """
    # clone master problem
    master_model = model_data.master_model
    polishing_model = master_model.clone()

    # fix all but the DR variables
    for blk in polishing_model.scenarios.values():
        model_vars = (
            blk.util.first_stage_variables
            + blk.util.second_stage_variables
            + blk.util.state_vars
        )
        for var in model_vars:
            var.fix()

    # deactivate all constraints
    for con in polishing_model.component_data_objects(Constraint, active=True):
        con.deactivate()

    # deactivate master objective
    polishing_model.obj.deactivate()

    # enforce DR effiencies according to iteration number
    # TODO: shouldn't have to do this here, ensure this is called before
    #       master problem is solved
    enforce_dr_degree(
        blk=polishing_model.scenarios[0, 0],
        config=config,
        degree=get_master_dr_degree(model_data, config),
    )

    # enforce additional efficiencies based on uncertain
    # parameter products per polynomial term
    include_extra_dr_efficiency = config.dr_polishing_options[
        "include_extra_dr_efficiency"
    ]
    if include_extra_dr_efficiency:
        enforce_dr_polishing_efficiencies(polishing_model, config)

    # add polishing model components based on norm of choice
    polishing_norm = config.dr_polishing_options["polishing_norm"]
    polishing_component_func_map = {
        "1_norm": add_1_norm_lps_polishing_components,
        "inf_norm": add_inf_norm_lps_polishing_components,
    }
    polishing_component_func = polishing_component_func_map[polishing_norm]
    polishing_component_func(polishing_model, config)

    # deactivate all polishing components
    nom_util_blk = polishing_model.scenarios[0, 0].util

    # fix DR and polishing variables
    for dr_var_list in nom_util_blk.unfixed_dr_vars:
        for drvar in dr_var_list:
            drvar.fix()

    # fix auxiliary DR polishing variables
    for pol_var_list in nom_util_blk.unfixed_polishing_vars:
        for polvar in pol_var_list:
            polvar.fix()

    # deactivate polishing norm constraints
    for lb_conlist in nom_util_blk.active_polishing_lb_cons:
        for lb_con in lb_conlist:
            lb_con.deactivate()
    for ub_conlist in nom_util_blk.active_polishing_ub_cons:
        for ub_con in ub_conlist:
            ub_con.deactivate()

    polishing_model.polishing_obj.deactivate()

    # all model components should be fixed or deactivated.
    # relevant components will be unfixed/reactivated when
    # polishing problems are to be solved
    return polishing_model


def minimize_dr_vars_lps(model_data, config):
    """
    Polish the PyROS decision rule determined for the most
    recently solved master problem by minimizing the collective
    L1 norm of the vector of all decision rule variables.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Subordinate solver results for the polishing problem.
    polishing_successful : bool
        True if polishing model was solved to acceptable level,
        False otherwise.
    """
    polishing_model = create_dr_polishing_lp_problem(model_data, config)

    # get solver
    solver = config.global_solver
    acceptable_terminations = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal}

    tt_timer = TicTocTimer()
    polishing_successful = True

    nominal_polishing_block = polishing_model.scenarios[0, 0]
    polishing_zip = zip(
        nominal_polishing_block.util.second_stage_variables,
        nominal_polishing_block.util.unfixed_dr_vars,
        nominal_polishing_block.util.unfixed_polishing_vars,
        nominal_polishing_block.util.active_polishing_lb_cons,
        nominal_polishing_block.util.active_polishing_ub_cons,
        polishing_model.polishing_obj.values(),
    )
    for tup in polishing_zip:
        # one DR polishing problem (LP) per second-stage variable.
        (
            second_stage_var,
            dr_var_list,
            pol_var_list,
            lb_con_list,
            ub_con_list,
            obj,
        ) = tup

        dr_var_idx = nominal_polishing_block.util.decision_rule_vars.index(
            dr_var_list[0].parent_component()
        )

        # unfix relevant variables, activate relevant objectives
        # and constraints
        for var in dr_var_list + pol_var_list:
            var.unfix()
        for con in lb_con_list + ub_con_list:
            con.activate()
        obj.activate()
        for blk in polishing_model.scenarios.values():
            blk.util.decision_rule_eqns[dr_var_idx].activate()

        # check correct components unfixed/activated
        # print("*" * 80)
        # print(list(ComponentSet(
        #     var.name
        #     for var in polishing_model.component_data_objects(Var)
        #     if not var.fixed
        # )))
        # print(list(ComponentSet(
        #     con.name
        #     for con in polishing_model.component_data_objects(
        #         Constraint, active=True
        #     )
        # )))
        # print([
        #     obj.name
        #     for obj in polishing_model.component_data_objects(
        #         Objective, active=True
        #     )
        # ])
        # print("Unused components:")
        # unused_vars = [
        #     var for var in polishing_model.component_data_objects(Var)
        #     if not var.fixed and all(
        #         var not in ComponentSet(identify_variables(con.body))
        #         for con in polishing_model.component_data_objects(Constraint, active=True)
        #     )
        # ]
        # for var in unused_vars:
        #     print(var.name)
        # unused_cons = [
        #     con
        #     for con in polishing_model.component_data_objects(Constraint, active=True)
        #     if all(var.fixed for var in ComponentSet(identify_variables(con.body)))
        # ]
        # for con in unused_cons:
        #     print(con.name)
        # import pdb
        # pdb.set_trace()

        # model now set up for current second-stage variable.
        # attempt to solve
        # print("Obj before", value(obj))
        ssv_name = second_stage_var.getname(
            relative_to=nominal_polishing_block,
            fully_qualified=True,
        )
        config.progress_logger.debug(f" DR polishing for second-stage var {ssv_name!r}")
        config.progress_logger.debug(f" Initial DR norm: {value(obj)}")

        # attempt solve
        orig_setting, custom_setting_present = adjust_solver_time_settings(
            model_data.timing, solver, config
        )
        model_data.timing.start_timer("main.dr_polishing")
        tt_timer.tic(msg=None)
        try:
            results = solver.solve(
                polishing_model,
                tee=config.tee,
                load_solutions=False,
            )
        except ApplicationError:
            config.progress_logger.error(
                f"Optimizer {repr(solver)} encountered an exception "
                "attempting to solve decision rule polishing problem "
                f"in iteration {model_data.iteration}"
            )
            raise
        else:
            setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, tt_timer.toc(msg=None))
            model_data.timing.stop_timer("main.dr_polishing")
        finally:
            revert_solver_max_time_adjustment(
                solver, orig_setting, custom_setting_present, config
            )

        # interested in the time and termination status for debugging
        # purposes
        config.progress_logger.debug(" Done solving DR polishing problem")
        config.progress_logger.debug(
            f"  Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)} s"
        )
        config.progress_logger.debug(
            f"  Termination status: {results.solver.termination_condition} "
        )

        # take action based on solver termination status
        if results.solver.termination_condition not in acceptable_terminations:
            polishing_successful = False
            config.progress_logger.debug(" Polishing unsuccessful this step")
        else:
            # load polished DR to model
            polishing_model.solutions.load_from(results)
            # print("Obj after", value(obj))
            config.progress_logger.debug(f" Optimized DR norm: {value(obj)}")

        # fix variables, objective, and constraints that were
        # unfixed/activated in this step
        for var in dr_var_list + pol_var_list:
            var.fix()
        for con in lb_con_list + ub_con_list:
            con.deactivate()
        obj.deactivate()
        for blk in polishing_model.scenarios.values():
            blk.util.decision_rule_eqns[dr_var_idx].deactivate()

    # update master model second-stage, state, and DR
    # variables based on polishing results
    for idx, blk in model_data.master_model.scenarios.items():
        ssv_zip = zip(
            blk.util.second_stage_variables,
            polishing_model.scenarios[idx].util.second_stage_variables,
        )
        sv_zip = zip(
            blk.util.state_vars, polishing_model.scenarios[idx].util.state_vars
        )
        for master_ssv, polish_ssv in ssv_zip:
            master_ssv.set_value(value(polish_ssv))
        for master_sv, polish_sv in sv_zip:
            master_sv.set_value(value(polish_sv))

        # update master problem decision rule variables
        dr_var_zip = zip(
            blk.util.decision_rule_vars,
            polishing_model.scenarios[idx].util.decision_rule_vars,
        )
        for master_dr, polish_dr in dr_var_zip:
            for mvar, pvar in zip(master_dr.values(), polish_dr.values()):
                mvar.set_value(value(pvar), skip_validation=True)

    return results, polishing_successful


def get_default_dr_polishing_options():
    """
    Return dict of default PyROS DR polishing options.
    """
    return dict(
        formulation="single_NLP",
        polishing_norm="inf_norm",
        polishing_norm_scaling="z_nom",
        dr_eq_scaling="none",
        include_extra_dr_efficiency=True,
        include_static_term_in_norm=False,
    )


def standardize_dr_polishing_options(polishing_options):
    """
    Standardize DR polishing options.
    """
    default_options = get_default_dr_polishing_options()
    return {
        option: polishing_options.get(option, val)
        for option, val in default_options.items()
    }


def minimize_dr_vars(model_data, config):
    """
    Solve decision rule polishing problem and update recourse policy
    accordingly.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Subordinate solver results for the polishing problem.
    polishing_successful : bool
        True if polishing model was solved to acceptable level,
        False otherwise.

    Note
    ----
    The decision rule polishing problem is solved with
    the primary optimizer used for the master problems.
    """
    # determine polishing formulation to use
    polishing_formulation = config.dr_polishing_options["formulation"]

    # now perform polishing, based on formulation
    polishing_formulation_to_func_map = {
        "LPs": minimize_dr_vars_lps,
        "single_NLP": minimize_dr_vars_nlp,
    }
    if polishing_formulation in polishing_formulation_to_func_map:
        return polishing_formulation_to_func_map[polishing_formulation](
            model_data=model_data,
            config=config,
        )
    else:
        raise ValueError(
            f"Invalid value {polishing_formulation!r} "
            "for DR polishing setting 'formulation'."
            f"Ensure value is one of: "
            f"{','.join(repr(form) for form in polishing_formulation_to_func_map)}"
        )


def add_p_robust_constraint(model_data, config):
    """
    p-robustness--adds constraints to the master problem ensuring that the
    optimal k-th iteration solution is within (1+rho) of the nominal
    objective. The parameter rho is specified by the user and should be between.
    """
    rho = config.p_robustness['rho']
    model = model_data.master_model
    block_0 = model.scenarios[0, 0]
    frac_nom_cost = (1 + rho) * (
        block_0.first_stage_objective + block_0.second_stage_objective
    )

    for block_k in model.scenarios[model_data.iteration, :]:
        model.p_robust_constraints.add(
            block_k.first_stage_objective + block_k.second_stage_objective
            <= frac_nom_cost
        )
    return


def add_scenario_to_master(model_data, violations):
    """
    Add block to master, without cloning the master_model.first_stage_variables
    """

    m = model_data.master_model
    i = max(m.scenarios.keys())[0] + 1

    # === Add a block to master for each violation
    idx = 0  # Only supporting adding single violation back to master in v1
    new_block = selective_clone(
        m.scenarios[0, 0], m.scenarios[0, 0].util.first_stage_variables
    )
    m.scenarios[i, idx].transfer_attributes_from(new_block)

    # === Set uncertain params in new block(s) to correct value(s)
    for j, p in enumerate(m.scenarios[i, idx].util.uncertain_params):
        p.set_value(violations[j])

    return


def get_master_dr_degree(model_data, config):
    """
    Determine DR order to enforce based on iteration
    number and model data.
    """
    if model_data.iteration == 0:
        return 0
    elif model_data.iteration <= len(config.uncertain_params):
        return min(1, config.decision_rule_order)
    else:
        return min(2, config.decision_rule_order)


def higher_order_decision_rule_efficiency(model_data, config):
    # === Efficiencies for decision rules
    #  if iteration <= |q| then all d^n where n > 1 are fixed to 0
    #  if iteration == 0, all d^n, n > 0 are fixed to 0
    #  These efficiencies should be carried through as d* to polishing
    order_to_enforce = get_master_dr_degree(model_data, config)
    enforce_dr_degree(
        blk=model_data.master_model.scenarios[0, 0],
        config=config,
        degree=order_to_enforce,
    )


def solver_call_master(model_data, config, solver, solve_data):
    """
    Invoke subsolver(s) on PyROS master problem.

    Parameters
    ----------
    model_data : MasterProblemData
        Container for current master problem and related data.
    config : ConfigDict
        PyROS solver settings.
    solver : solver type
        Primary subordinate optimizer with which to solve
        the master problem. This may be a local or global
        NLP solver.
    solve_data : MasterResult
        Master problem results object. May be empty or contain
        master feasibility problem results.

    Returns
    -------
    master_soln : MasterResult
        Master problem results object, containing master
        model and subsolver results.
    """
    nlp_model = model_data.master_model
    master_soln = solve_data
    solver_term_cond_dict = {}

    if config.solve_master_globally:
        solvers = [solver] + config.backup_global_solvers
    else:
        solvers = [solver] + config.backup_local_solvers

    solve_mode = "global" if config.solve_master_globally else "local"
    config.progress_logger.debug("Solving master problem")

    higher_order_decision_rule_efficiency(model_data, config)

    timer = TicTocTimer()
    for idx, opt in enumerate(solvers):
        if idx > 0:
            config.progress_logger.warning(
                f"Invoking backup solver {opt!r} "
                f"(solver {idx + 1} of {len(solvers)}) for "
                f"master problem of iteration {model_data.iteration}."
            )
        orig_setting, custom_setting_present = adjust_solver_time_settings(
            model_data.timing, opt, config
        )
        model_data.timing.start_timer("main.master")
        timer.tic(msg=None)
        try:
            results = opt.solve(
                nlp_model,
                tee=config.tee,
                load_solutions=False,
                symbolic_solver_labels=True,
            )
        except ApplicationError:
            # account for possible external subsolver errors
            # (such as segmentation faults, function evaluation
            # errors, etc.)
            config.progress_logger.error(
                f"Optimizer {repr(opt)} ({idx + 1} of {len(solvers)}) "
                "encountered exception attempting to "
                f"solve master problem in iteration {model_data.iteration}"
            )
            raise
        else:
            setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
            model_data.timing.stop_timer("main.master")
        finally:
            revert_solver_max_time_adjustment(
                solver, orig_setting, custom_setting_present, config
            )

        optimal_termination = check_optimal_termination(results)
        infeasible = results.solver.termination_condition == tc.infeasible

        if optimal_termination:
            nlp_model.solutions.load_from(results)

        # record master problem termination conditions
        # for this particular subsolver
        # pyros termination condition is determined later in the
        # algorithm
        solver_term_cond_dict[str(opt)] = str(results.solver.termination_condition)
        master_soln.termination_condition = results.solver.termination_condition
        master_soln.pyros_termination_condition = None
        (
            try_backup,
            _,
        ) = (
            master_soln.master_subsolver_results
        ) = process_termination_condition_master_problem(config=config, results=results)

        master_soln.nominal_block = nlp_model.scenarios[0, 0]
        master_soln.results = results
        master_soln.master_model = nlp_model

        # if model was solved successfully, update/record the results
        # (nominal block DOF variable and objective values)
        if not try_backup and not infeasible:
            master_soln.fsv_vals = list(
                v.value for v in nlp_model.scenarios[0, 0].util.first_stage_variables
            )
            if config.objective_focus is ObjectiveType.nominal:
                master_soln.ssv_vals = list(
                    v.value
                    for v in nlp_model.scenarios[0, 0].util.second_stage_variables
                )
                master_soln.second_stage_objective = value(
                    nlp_model.scenarios[0, 0].second_stage_objective
                )
            else:
                idx = max(nlp_model.scenarios.keys())[0]
                master_soln.ssv_vals = list(
                    v.value
                    for v in nlp_model.scenarios[idx, 0].util.second_stage_variables
                )
                master_soln.second_stage_objective = value(
                    nlp_model.scenarios[idx, 0].second_stage_objective
                )
            master_soln.first_stage_objective = value(
                nlp_model.scenarios[0, 0].first_stage_objective
            )

            # debugging: log breakdown of master objective
            config.progress_logger.debug(" Optimized master objective breakdown:")
            config.progress_logger.debug(
                f"  First-stage objective {master_soln.first_stage_objective}"
            )
            config.progress_logger.debug(
                f"  Second-stage objective {master_soln.second_stage_objective}"
            )
            master_obj = (
                master_soln.first_stage_objective + master_soln.second_stage_objective
            )
            config.progress_logger.debug(f"  Objective {master_obj}")
            config.progress_logger.debug(
                f" Termination condition: {results.solver.termination_condition}"
            )
            config.progress_logger.debug(
                f" Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s"
            )

            master_soln.nominal_block = nlp_model.scenarios[0, 0]
            master_soln.results = results
            master_soln.master_model = nlp_model

        # if PyROS time limit exceeded, exit loop and return solution
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                try_backup = False
                master_soln.master_subsolver_results = (
                    None,
                    pyrosTerminationCondition.time_out,
                )
                master_soln.pyros_termination_condition = (
                    pyrosTerminationCondition.time_out
                )

        if not try_backup:
            return master_soln

    # all solvers have failed to return an acceptable status.
    # we will terminate PyROS with subsolver error status.
    # at this point, export subproblem to file, if desired.
    # NOTE: subproblem is written with variables set to their
    #       initial values (not the final subsolver iterate)
    save_dir = config.subproblem_file_directory
    serialization_msg = ""
    if save_dir and config.keepfiles:
        output_problem_path = os.path.join(
            save_dir,
            (
                config.uncertainty_set.type
                + "_"
                + model_data.original.name
                + "_master_"
                + str(model_data.iteration)
                + ".bar"
            ),
        )
        nlp_model.write(
            output_problem_path, io_options={'symbolic_solver_labels': True}
        )
        serialization_msg = (
            " For debugging, problem has been serialized to the file "
            f"{output_problem_path!r}."
        )

    deterministic_model_qual = (
        " (i.e., the deterministic model)" if model_data.iteration == 0 else ""
    )
    deterministic_msg = (
        (
            " Please ensure your deterministic model "
            f"is solvable by at least one of the subordinate {solve_mode} "
            "optimizers provided."
        )
        if model_data.iteration == 0
        else ""
    )
    master_soln.pyros_termination_condition = pyrosTerminationCondition.subsolver_error
    config.progress_logger.warning(
        f"Could not successfully solve master problem of iteration "
        f"{model_data.iteration}{deterministic_model_qual} with any of the "
        f"provided subordinate {solve_mode} optimizers. "
        f"(Termination statuses: "
        f"{[term_cond for term_cond in solver_term_cond_dict.values()]}.)"
        f"{deterministic_msg}"
        f"{serialization_msg}"
    )

    return master_soln


def solve_master(model_data, config):
    """
    Solve the master problem
    """
    master_soln = MasterResult()

    # no master feas problem for iteration 0
    if model_data.iteration > 0:
        if not config.bypass_master_feasibility:
            results = solve_master_feasibility_problem(model_data, config)
        else:
            results = SolverResults()
            setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, 0)
        master_soln.feasibility_problem_results = results

        # if pyros time limit reached, load time out status
        # to master results and return to caller
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                # load master model
                master_soln.master_model = model_data.master_model
                master_soln.nominal_block = model_data.master_model.scenarios[0, 0]

                # empty results object, with master solve time of zero
                master_soln.results = SolverResults()
                setattr(master_soln.results.solver, TIC_TOC_SOLVE_TIME_ATTR, 0)

                # PyROS time out status
                master_soln.pyros_termination_condition = (
                    pyrosTerminationCondition.time_out
                )
                master_soln.master_subsolver_results = (
                    None,
                    pyrosTerminationCondition.time_out,
                )
                return master_soln

    solver = (
        config.global_solver if config.solve_master_globally else config.local_solver
    )

    return solver_call_master(
        model_data=model_data, config=config, solver=solver, solve_data=master_soln
    )
