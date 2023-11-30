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

    # load master feasibility point to master model
    for master_var, feas_var in model_data.feasibility_problem_varmap:
        master_var.set_value(feas_var.value, skip_validation=True)

    return results


def create_dr_polishing_nlp(model_data, config):
    """
    Create decision rule polishing problem instance.

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
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    # fix first-stage variables (including epigraph, where applicable)
    first_stage_vars = nominal_polishing_block.util.first_stage_variables
    for var in first_stage_vars:
        var.fix()

    # enforce DR effiencies according to iteration number
    # TODO: shouldn't have to do this here, ensure this is called before
    #       master problem is solved
    enforce_dr_degree(
        blk=polishing_model.scenarios[0, 0],
        config=config,
        degree=get_master_dr_degree(model_data, config),
    )

    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars
    nominal_polishing_block.util.polishing_vars = polishing_vars = []
    for idx, indexed_dr_var in enumerate(decision_rule_vars):
        # compile nonstatic DR variables into iterable object
        nonstatic_dr_var_map = {}
        for dr_var_idx, dr_var in indexed_dr_var.items():
            if nominal_polishing_block.util.dr_var_to_exponent_map[dr_var] > 0:
                nonstatic_dr_var_map[dr_var_idx] = dr_var

        # declare DR polishing auxiliary variables
        # (for representing abs values of DR variables)
        indexed_polishing_var = Var(
            list(nonstatic_dr_var_map.keys()),
            domain=NonNegativeReals,
        )
        nominal_polishing_block.add_component(
            unique_component_name(nominal_polishing_block, f"dr_polishing_var_{idx}"),
            indexed_polishing_var,
        )
        polishing_vars.append(indexed_polishing_var)

    # declare variable for representing infinity norm of polishing variables
    polishing_model.infinity_norm_var = Var(domain=NonNegativeReals)

    # ensure master optimality constraint enforced
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

    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        polishing_vars,
        nominal_polishing_block.util.second_stage_variables,
    )
    nominal_polishing_block.util.polishing_vars = all_polishing_vars = []
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    nominal_polishing_block.util.polishing_inf_norm_cons = all_infnorm_cons = []
    for idx, (dr_eq, indexed_polishing_var, ss_var) in enumerate(dr_eq_var_zip):
        all_polishing_vars.append(indexed_polishing_var)

        # scale the DR equation? (for all blocks)
        # dr_eq_scale_factor = max(1, abs(ss_var.value))
        # for blk in polishing_model.scenarios.values():
        #     blk_dr_eq = blk.util.decision_rule_eqns[idx]
        #     blk_dr_eq.set_value((
        #         blk_dr_eq.lower / dr_eq_scale_factor,
        #         sum(arg / dr_eq_scale_factor for arg in blk_dr_eq.body.args),
        #         blk_dr_eq.upper / dr_eq_scale_factor,
        #     ))

        # components for absolute value and infinity norm constraints
        polishing_infinity_norm_cons = Constraint(
            indexed_polishing_var.index_set(),
        )
        polishing_absolute_value_lb_cons = Constraint(
            indexed_polishing_var.index_set(),
        )
        polishing_absolute_value_ub_cons = Constraint(
            indexed_polishing_var.index_set(),
        )

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_abs_val_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_abs_val_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_infinity_norm_con_{idx}",
            ),
            polishing_infinity_norm_cons,
        )

        all_infnorm_cons.append(polishing_infinity_norm_cons)
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        # ensure second-stage variable term excluded
        dr_expr_terms = dr_eq.body.args[:-1]

        # from pyomo.environ import prod
        for dr_eq_term in dr_expr_terms:
            # extract DR var in term
            dr_var_in_term = dr_eq_term.args[-1]
            # term_coeff = prod(dr_eq_term.args[:-1])

            # TODO here: conditions under which to fix DR variable:
            # (1) it has already been fixed from master due to
            #     DR efficiencies
            # (2) coefficient of term in DR expression is 0
            ...

            dr_var_in_term_idx = dr_var_in_term.index()
            if dr_var_in_term_idx not in indexed_polishing_var.index_set():
                continue

            # get corresponding polishing variable
            polishing_var = indexed_polishing_var[dr_var_in_term_idx]

            # add polishing constraints
            polishing_absolute_value_lb_cons[dr_var_in_term_idx] = (
                -polishing_var - dr_eq_term <= 0
            )
            polishing_absolute_value_ub_cons[dr_var_in_term_idx] = (
                dr_eq_term - polishing_var <= 0
            )
            # add infinity norm constraint
            polishing_infinity_norm_cons[dr_var_in_term_idx] = (
                polishing_var - polishing_model.infinity_norm_var <= 0
            )

            # if DR var is fixed, then fix polishing variable as well.
            # also, deactivate coresponding polishing constraints
            if dr_var_in_term.fixed:
                polishing_var.fix()
                polishing_absolute_value_lb_cons[dr_var_in_term_idx].deactivate()
                polishing_absolute_value_ub_cons[dr_var_in_term_idx].deactivate()

            # initialize auxiliary polishing variable
            polishing_var.set_value(abs(value(dr_eq_term)))

    # now initialize infinity norm var
    polishing_model.infinity_norm_var.set_value(max(
        value(var)
        for indexedvar in all_polishing_vars
        for var in indexedvar.values()
    ))

    # deactivate objective
    polishing_model.obj.deactivate()

    # declare polishing objective
    polishing_model.polishing_obj = Objective(
        expr=polishing_model.infinity_norm_var,
        # expr=sum(
        #     sum(polishing_var.values())
        #     for polishing_var in polishing_vars
        # )
    )

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


def create_dr_polishing_lp_problem(model_data, config):
    """
    Create DR polishing problem instance.

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
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    # fix all but the DR variables
    for blk in polishing_model.scenarios.values():
        model_vars = (
            blk.util.first_stage_variables
            + blk.util.second_stage_variables
            + blk.util.state_vars
        )
        for var in model_vars:
            var.fix()

    # enforce DR effiencies according to iteration number
    # TODO: shouldn't have to do this here, ensure this is called before
    #       master problem is solved
    enforce_dr_degree(
        blk=polishing_model.scenarios[0, 0],
        config=config,
        degree=get_master_dr_degree(model_data, config),
    )

    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars
    nominal_polishing_block.util.polishing_vars = polishing_vars = []
    nominal_polishing_block.util.polishing_infinity_norm_vars = []
    for idx, indexed_dr_var in enumerate(decision_rule_vars):
        # compile nonstatic DR variables into iterable object
        nonstatic_dr_var_map = {}
        for dr_var_idx, dr_var in indexed_dr_var.items():
            if nominal_polishing_block.util.dr_var_to_exponent_map[dr_var] > 0:
                nonstatic_dr_var_map[dr_var_idx] = dr_var

        # declare DR polishing auxiliary variables
        # (for representing abs values of DR variables)
        indexed_polishing_var = Var(
            list(nonstatic_dr_var_map.keys()),
            domain=NonNegativeReals,
        )
        nominal_polishing_block.add_component(
            unique_component_name(nominal_polishing_block, f"dr_polishing_var_{idx}"),
            indexed_polishing_var,
        )
        polishing_vars.append(indexed_polishing_var)

    # declare variables for representing infinity norm of polishing variables
    polishing_model.infinity_norm_var = Var(
        range(len(nominal_polishing_block.util.decision_rule_vars)),
        domain=NonNegativeReals,
    )

    # deactivate all but the DR constraints
    for blk in polishing_model.scenarios.values():
        for con in blk.component_data_objects(Constraint, active=True):
            con.deactivate()
        for con in blk.util.decision_rule_eqns:
            con.activate()

    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        polishing_vars,
        nominal_polishing_block.util.second_stage_variables,
        polishing_model.infinity_norm_var.values(),
    )
    nominal_polishing_block.util.polishing_vars = all_polishing_vars = []
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    nominal_polishing_block.util.polishing_inf_norm_cons = all_infnorm_cons = []
    for idx, (dr_eq, indexed_polishing_var, ss_var, inf_var) in enumerate(dr_eq_var_zip):
        all_polishing_vars.append(indexed_polishing_var)

        # scale the DR equation? (for all blocks)
        # dr_eq_scale_factor = max(1, abs(ss_var.value))
        # for blk in polishing_model.scenarios.values():
        #     blk_dr_eq = blk.util.decision_rule_eqns[idx]
        #     blk_dr_eq.set_value((
        #         blk_dr_eq.lower / dr_eq_scale_factor,
        #         sum(arg / dr_eq_scale_factor for arg in blk_dr_eq.body.args),
        #         blk_dr_eq.upper / dr_eq_scale_factor,
        #     ))

        # components for absolute value and infinity norm constraints
        polishing_infinity_norm_cons = Constraint(
            indexed_polishing_var.index_set(),
        )
        polishing_absolute_value_lb_cons = Constraint(
            indexed_polishing_var.index_set(),
        )
        polishing_absolute_value_ub_cons = Constraint(
            indexed_polishing_var.index_set(),
        )

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_abs_val_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_abs_val_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_infinity_norm_con_{idx}",
            ),
            polishing_infinity_norm_cons,
        )

        all_infnorm_cons.append(polishing_infinity_norm_cons)
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        # ensure second-stage variable term excluded
        dr_expr_terms = dr_eq.body.args[:-1]

        # from pyomo.environ import prod
        for dr_eq_term in dr_expr_terms:
            # extract DR var in term
            dr_var_in_term = dr_eq_term.args[-1]
            # term_coeff = prod(dr_eq_term.args[:-1])

            # TODO here: conditions under which to fix DR variable:
            # (1) it has already been fixed from master due to
            #     DR efficiencies
            # (2) coefficient of term in DR expression is 0
            ...

            dr_var_in_term_idx = dr_var_in_term.index()
            if dr_var_in_term_idx not in indexed_polishing_var.index_set():
                continue

            # get corresponding polishing variable
            polishing_var = indexed_polishing_var[dr_var_in_term_idx]

            # initialize auxiliary polishing variable
            polishing_var.set_value(abs(value(dr_eq_term)))

            # add polishing constraints
            polishing_absolute_value_lb_cons[dr_var_in_term_idx] = (
                -polishing_var - dr_eq_term <= 0
            )
            polishing_absolute_value_ub_cons[dr_var_in_term_idx] = (
                dr_eq_term - polishing_var <= 0
            )
            # add infinity norm constraint
            polishing_infinity_norm_cons[dr_var_in_term_idx] = (
                polishing_var - inf_var <= 0
            )

            # if DR var is fixed, then fix polishing variable as well.
            # also, deactivate coresponding polishing constraints
            if dr_var_in_term.fixed:
                polishing_var.fix()
                polishing_absolute_value_lb_cons[dr_var_in_term_idx].deactivate()
                polishing_absolute_value_ub_cons[dr_var_in_term_idx].deactivate()

            # initialize auxiliary polishing variable
            polishing_var.set_value(abs(value(dr_eq_term)))

        # initialize infinity norm variable
        inf_var.set_value(max(
            value(var) for var in indexed_polishing_var.values()
        ))

    # deactivate objective copied from master model
    polishing_model.obj.deactivate()

    # declare polishing objective
    polishing_model.polishing_obj = Objective(
        polishing_model.infinity_norm_var.index_set(),
        expr={
            idx: norm_var
            for idx, norm_var in polishing_model.infinity_norm_var.items()
        },
    )

    # take note of which DR and polishing variables were not fixed
    # and which polishing constraints were not deactivated
    nominal_polishing_block.util.unfixed_dr_vars = [
        [var for var in indexed_dr_var.values() if not var.fixed]
        for indexed_dr_var in nominal_polishing_block.util.decision_rule_vars
    ]
    nominal_polishing_block.util.unfixed_polishing_vars = [
        [var for var in indexed_polishing_var.values() if not var.fixed]
        for indexed_polishing_var in all_polishing_vars
    ]
    nominal_polishing_block.util.active_polishing_lb_cons = [
        [con for con in indexed_con.values() if con.active]
        for indexed_con in all_lb_cons
    ]
    nominal_polishing_block.util.active_polishing_ub_cons = [
        [con for con in indexed_con.values() if con.active]
        for indexed_con in all_ub_cons
    ]
    nominal_polishing_block.util.active_infnorm_cons = [
        [con for con in indexed_con.values() if con.active]
        for indexed_con in all_infnorm_cons
    ]
    for var in decision_rule_vars:
        var.fix()
    for var in all_polishing_vars:
        var.fix()
    for var in polishing_model.infinity_norm_var.values():
        var.fix()
    for con in all_lb_cons:
        con.deactivate()
    for con in all_ub_cons:
        con.deactivate()
    for con in all_infnorm_cons:
        con.deactivate()
    for blk in polishing_model.scenarios.values():
        for eq in blk.util.decision_rule_eqns:
            eq.deactivate()
    polishing_model.polishing_obj.deactivate()

    # # debugging
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
    # for indcon in all_lb_cons:
    #     for con in indcon.values():
    #         if not (con.lslack(), con.uslack()) >= (-1e-4, -1e-4):
    #             print(con.name, con.lslack(), con.uslack())
    # for indcon in all_ub_cons:
    #     for con in indcon.values():
    #         if not (con.lslack(), con.uslack()) >= (-1e-4, -1e-4):
    #             print(con.name, con.lslack(), con.uslack())
    # for indcon in all_infnorm_cons:
    #     for con in indcon.values():
    #         if not (con.lslack(), con.uslack()) >= (-1e-4, -1e-4):
    #             print(con.name, con.lslack(), con.uslack())
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
    polishing_model = create_dr_polishing_nlp(model_data, config)

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

    # diag_model = polishing_model.clone()
    # from idaes.core.util.model_diagnostics import DiagnosticsToolbox
    # dt = DiagnosticsToolbox(model=diag_model)

    # for blk in diag_model.scenarios.values():
    #     for var in blk.util.decision_rule_vars:
    #         var.fix()
    # import pdb
    # pdb.set_trace()

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
        f"  Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)} s"
    )
    config.progress_logger.debug(
        f"  Termination status: {results.solver.termination_condition} "
    )

    # === Process solution by termination condition
    acceptable = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal}
    if results.solver.termination_condition not in acceptable:
        # continue with "unpolished" master model solution
        return results, False
    # update master model second-stage, state, and decision rule
    # variables to polishing model solution
    polishing_model.solutions.load_from(results)

    # print out the variable values
    # nominal_polishing_block = polishing_model.scenarios[0, 0]
    # nominal_polishing_block.util.polishing_vars
    # conlist = [
    #     nominal_polishing_block.util.polishing_inf_norm_cons,
    #     nominal_polishing_block.util.polishing_abs_val_lb_cons,
    #     nominal_polishing_block.util.polishing_abs_val_ub_cons,
    # ]
    # for conattr in conlist:
    #     print("---")
    #     for indcon in conattr:
    #         for con in indcon.values():
    #             print(con.name, con.lslack(), con.uslack())
    # print("Polishing variable values")
    # for indvar in nominal_polishing_block.util.polishing_vars:
    #     for pvar in indvar.values():
    #         print(pvar.name, pvar.value)
    # print("DR term values")
    # for dreq in nominal_polishing_block.util.decision_rule_eqns:
    #     for term in dreq.body.args[1:-1]:
    #         print(term, value(term))
    # eg_pol_var = nominal_polishing_block.dr_polishing_var_0[2]
    # print("Cons with polishing var:", eg_pol_var.name)
    # for con in polishing_model.component_data_objects(Constraint, active=True):
    #     if eg_pol_var in ComponentSet(identify_variables(con.body)):
    #         print(con.name)
    #         con.pprint()
    # import pdb
    # pdb.set_trace()

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
    config.progress_logger.debug("Polished Master objective:")

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
        " First-stage objective " f"{value(worst_master_blk.first_stage_objective)}"
    )
    config.progress_logger.debug(
        " Second-stage objective " f"{value(worst_master_blk.second_stage_objective)}"
    )
    polished_master_obj = value(
        worst_master_blk.first_stage_objective + worst_master_blk.second_stage_objective
    )
    config.progress_logger.debug(f" Objective {polished_master_obj}")

    return results, True


def minimize_dr_vars(model_data, config):
    return minimize_dr_vars_nlp(model_data, config)


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
    solver = config.local_solver
    if config.solve_master_globally:
        solver = config.global_solver
    acceptable_terminations = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal}

    tt_timer = TicTocTimer()
    polishing_successful = True

    nominal_polishing_block = polishing_model.scenarios[0, 0]
    polishing_zip = zip(
        polishing_model.infinity_norm_var.values(),
        nominal_polishing_block.util.unfixed_dr_vars,
        nominal_polishing_block.util.unfixed_polishing_vars,
        nominal_polishing_block.util.active_polishing_lb_cons,
        nominal_polishing_block.util.active_polishing_ub_cons,
        nominal_polishing_block.util.active_infnorm_cons,
        polishing_model.polishing_obj.values(),
    )
    for tup in polishing_zip:
        # one DR polishing problem (LP) per second-stage variable.
        (
            inf_norm_var,
            dr_var_list,
            pol_var_list,
            lb_con_list,
            ub_con_list,
            infnorm_con_list,
            obj,
        ) = tup

        dr_var_idx = nominal_polishing_block.util.decision_rule_vars.index(
            dr_var_list[0].parent_component()
        )

        # unfix relevant variables, activate relevant objectives
        # and constraints
        inf_norm_var.unfix()
        for var in dr_var_list + pol_var_list:
            var.unfix()
        for con in lb_con_list + ub_con_list + infnorm_con_list:
            con.activate()
        obj.activate()
        for blk in polishing_model.scenarios.values():
            blk.util.decision_rule_eqns[dr_var_idx].activate()

        # check correct components unfixed/activated
        # print([
        #     var.name
        #     for var in polishing_model.component_data_objects(Var)
        #     if not var.fixed
        # ])
        # print([
        #     con.name
        #     for con in polishing_model.component_data_objects(
        #         Constraint, active=True
        #     )
        # ])
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
        inf_norm_var.fix()
        for var in dr_var_list + pol_var_list:
            var.fix()
        for con in lb_con_list + ub_con_list + infnorm_con_list:
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
            config.progress_logger.debug("Master objective")
            config.progress_logger.debug(
                f" First-stage objective {master_soln.first_stage_objective}"
            )
            config.progress_logger.debug(
                f" Second-stage objective {master_soln.second_stage_objective}"
            )
            master_obj = (
                master_soln.first_stage_objective + master_soln.second_stage_objective
            )
            config.progress_logger.debug(f" Objective {master_obj}")

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
        results = solve_master_feasibility_problem(model_data, config)
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
