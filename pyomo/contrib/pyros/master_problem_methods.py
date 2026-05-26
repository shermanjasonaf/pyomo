# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""
Functions for construction and solution of the PyROS master problem.
"""

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core import TransformationFactory
from pyomo.core.base import ConcreteModel, Block, Var, Objective, Constraint
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.core.expr import identify_variables, value
from pyomo.core.util import prod
from pyomo.opt import TerminationCondition as tc
from pyomo.repn.standard_repn import generate_standard_repn

from pyomo.contrib.pyros.solve_data import MasterResults
from pyomo.contrib.pyros.util import (
    call_solver,
    call_solver_and_backups,
    DR_POLISHING_PARAM_PRODUCT_ZERO_TOL,
    enforce_dr_degree,
    get_all_first_stage_eq_cons,
    get_dr_expression,
    check_time_limit_reached,
    generate_all_decision_rule_var_data_objects,
    ObjectiveType,
    pyrosTerminationCondition,
    SolverCallStatus,
    TIC_TOC_SOLVE_TIME_ATTR,
    write_subproblem,
)


def construct_initial_master_problem(model_data):
    """
    Construct the initial master problem model object
    from the preprocessed working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object,
        containing the preprocessed working model.

    Returns
    -------
    master_model : ConcreteModel
        Initial master problem model object.
        Contains a single scenario block fully cloned from
        the working model.
    """
    master_model = ConcreteModel()
    master_model.scenarios = Block(NonNegativeIntegers, NonNegativeIntegers)
    add_scenario_block_to_master_problem(
        master_model=master_model,
        scenario_idx=(0, 0),
        param_realization=model_data.config.nominal_uncertain_param_vals,
        from_block=model_data.working_model,
        clone_first_stage_components=True,
    )

    # epigraph Objective was not added during preprocessing,
    # as we wanted to add it to the root block of the master
    # model rather than to the model to prevent
    # duplication across scenario sub-blocks
    master_model.epigraph_obj = Objective(
        expr=master_model.scenarios[0, 0].first_stage.epigraph_var
    )

    return master_model


def add_scenario_block_to_master_problem(
    master_model,
    scenario_idx,
    param_realization,
    from_block,
    clone_first_stage_components,
):
    """
    Add new scenario block to the master model.

    Parameters
    ----------
    master_model : ConcreteModel
        Master model.
    scenario_idx : tuple
        Index of ``master_model.scenarios`` for the new block.
    param_realization : Iterable of numeric type
        Uncertain parameter realization for new block.
    from_block : BlockData
        Block from which to transfer attributes.
        This can be an existing scenario block, or a block
        with the same hierarchical structure as the
        preprocessed working model.
    clone_first_stage_components : bool
        True to clone first-stage variables
        when transferring attributes to the new block
        to the new block (as opposed to using the objects as
        they are in `from_block`), False otherwise.
    """
    # Note for any of the Vars not copied:
    # - if Var is not a member of an indexed var, then
    #   the 'name' attribute changes from
    #   '{from_block.name}.{var.name}'
    #   to 'scenarios[{scenario_idx}].{var.name}'
    # - otherwise, the name stays the same
    memo = dict()
    if not clone_first_stage_components:
        nonadjustable_comps = from_block.all_nonadjustable_variables
        memo = {id(comp): comp for comp in nonadjustable_comps}

        # we will clone the first-stage constraints
        # (mostly to prevent symbol map name clashes).
        # the duplicate constraints are redundant.
        # consider deactivating these constraints in the
        # off-nominal blocks?

    new_block = from_block.clone(memo=memo)
    master_model.scenarios[scenario_idx].transfer_attributes_from(new_block)

    # update uncertain parameter values in new block
    new_uncertain_params = master_model.scenarios[scenario_idx].uncertain_params
    for param, val in zip(new_uncertain_params, param_realization):
        param.set_value(val)

    # deactivate the first-stage constraints: they are duplicate
    if scenario_idx != (0, 0):
        new_blk = master_model.scenarios[scenario_idx]
        for con in new_blk.first_stage.inequality_cons.values():
            con.deactivate()
        for con in get_all_first_stage_eq_cons(new_blk):
            con.deactivate()


def construct_master_feasibility_problem(master_data):
    """
    Construct slack variable minimization problem from the master
    model.

    Slack variables are added only to the seconds-stage
    inequality constraints of the blocks added for the
    current PyROS iteration.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.

    Returns
    -------
    slack_model : ConcreteModel
        Slack variable model.
    """
    slack_model = master_data.master_model.clone()

    iteration = master_data.iteration

    # add slacks only to second-stage inequality constraints for the
    # newest master block(s).
    # these should be the only constraints that
    # may have been violated by the previous master and separation
    # solution(s)
    targets = []
    for blk in slack_model.scenarios[iteration, :]:
        targets.extend(blk.second_stage.inequality_cons.values())

    # retain original constraint expressions before adding slacks
    # (to facilitate slack initialization and scaling)
    pre_slack_con_exprs = ComponentMap((con, con.body - con.upper) for con in targets)

    # add slack variables and objective
    # inequalities g(v) <= b become g(v) - s^- <= b
    # note: this automatically deactivates all hitherto active
    #       objectives
    TransformationFactory("core.add_slack_variables").apply_to(
        slack_model, targets=targets
    )
    slack_vars = ComponentSet(
        slack_model._core_add_slack_variables.component_data_objects(
            Var, descend_into=True
        )
    )

    # initialize slack variables
    for con in pre_slack_con_exprs:
        # get mapping from slack variables to their (linear)
        # coefficients (+/-1) in the updated constraint expressions
        repn = generate_standard_repn(con.body)
        slack_var_coef_map = ComponentMap()
        for idx in range(len(repn.linear_vars)):
            var = repn.linear_vars[idx]
            if var in slack_vars:
                slack_var_coef_map[var] = repn.linear_coefs[idx]

        for slack_var in slack_var_coef_map:
            # coefficient determines whether the slack
            # is a +ve or -ve slack
            if slack_var_coef_map[slack_var] == -1:
                con_slack = max(0, value(pre_slack_con_exprs[con]))
            else:
                con_slack = max(0, -value(pre_slack_con_exprs[con]))

            slack_var.set_value(con_slack)

    return slack_model


def solve_master_feasibility_problem(master_data):
    """
    Solve a slack variable-based feasibility model derived
    from the master problem. Initialize the master problem
    to the  solution found by the optimizer if solved successfully,
    or to the initial point provided to the solver otherwise.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.

    Returns
    -------
    results : SolverResults
        Solver results.
    """
    model = construct_master_feasibility_problem(master_data)

    active_obj = next(model.component_data_objects(Objective, active=True))

    config = master_data.config
    config.progress_logger.debug("Solving master feasibility problem")
    config.progress_logger.debug(
        f" Initial objective (total slack): {value(active_obj)}"
    )

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    results = call_solver(
        model=model,
        solver=solver,
        config=config,
        timing_obj=master_data.timing,
        timer_name="main.master_feasibility",
        err_msg=(
            f"Optimizer {repr(solver)} encountered exception "
            "attempting to solve master feasibility problem in iteration "
            f"{master_data.iteration}."
        ),
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

        master_model = master_data.master_model

        # update the nonadjustable variables of the master model
        master_nadj_vars = master_model.scenarios[0, 0].all_nonadjustable_variables
        mfeas_nadj_vars = model.scenarios[0, 0].all_nonadjustable_variables
        for master_nadj_var, mfeas_nadj_var in zip(master_nadj_vars, mfeas_nadj_vars):
            master_nadj_var.set_value(mfeas_nadj_var.value, skip_validation=True)

        # update the adjustable variables of the master model
        scenario_blk_zip = zip(
            master_model.scenarios.values(), model.scenarios.values()
        )
        for master_blk, mfeas_blk in scenario_blk_zip:
            adj_var_zip = zip(
                master_blk.all_adjustable_variables, mfeas_blk.all_adjustable_variables
            )
            for master_adj_var, mfeas_adj_var in adj_var_zip:
                master_adj_var.set_value(mfeas_adj_var.value, skip_validation=True)
    else:
        config.progress_logger.debug(
            "Could not successfully solve master feasibility problem "
            f"of iteration {master_data.iteration} with primary subordinate "
            f"{'global' if config.solve_master_globally else 'local'} solver "
            "to acceptable level. "
            f"Termination stats:\n{results.solver}\n"
            "Maintaining unoptimized point for master problem initialization."
        )

    return results


def construct_dr_polishing_problem(master_data):
    """
    Construct DR polishing problem from the master problem.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.

    Returns
    -------
    polishing_model : ConcreteModel
        Polishing model.

    Note
    ----
    Polishing problem is to minimize the L1-norm of the vector of
    all decision rule polynomial terms, subject to the original
    master problem constraints, with all first-stage variables
    (including epigraph) fixed. Optimality of the polished
    DR with respect to the master objective is also enforced.
    """
    master_model = master_data.master_model
    polishing_model = master_model.clone()
    nominal_polishing_block = polishing_model.scenarios[0, 0]

    nominal_eff_var_partitioning = nominal_polishing_block.effective_var_partitioning

    nondr_nonadjustable_vars = (
        nominal_eff_var_partitioning.first_stage_variables
        # fixing epigraph variable constrains the problem
        # to the optimal master problem solution set
        + [nominal_polishing_block.first_stage.epigraph_var]
    )
    for var in nondr_nonadjustable_vars:
        var.fix()

    # deactivate original constraints that involved
    # only vars that have been fixed.
    # we do this mostly to ensure that the active equality constraints
    # do not grossly outnumber the unfixed Vars
    fixed_dr_vars = [
        var
        for var in generate_all_decision_rule_var_data_objects(nominal_polishing_block)
        if var.fixed
    ]
    fixed_nonadjustable_vars = ComponentSet(nondr_nonadjustable_vars + fixed_dr_vars)
    for blk in polishing_model.scenarios.values():
        for con in blk.component_data_objects(Constraint, active=True):
            vars_in_con = ComponentSet(identify_variables(con.body))
            if not (vars_in_con - fixed_nonadjustable_vars):
                con.deactivate()

    # we will add the polishing objective later
    polishing_model.epigraph_obj.deactivate()

    polishing_model.polishing_vars = polishing_vars = []
    indexed_dr_var_list = nominal_polishing_block.first_stage.decision_rule_vars
    for idx, indexed_dr_var in enumerate(indexed_dr_var_list):
        # auxiliary 'polishing' variables.
        # these are meant to represent the absolute values
        # of the terms of DR polynomial;
        # we need these for the L1-norm
        indexed_polishing_var = Var(
            list(indexed_dr_var.keys()), domain=NonNegativeReals
        )
        polishing_model.add_component(f"dr_polishing_var_{idx}", indexed_polishing_var)
        polishing_vars.append(indexed_polishing_var)

    # we need the DR expressions to set up the
    # absolute value constraints and initialize the
    # auxiliary polishing variables
    eff_ss_var_to_dr_expr_pairs = [
        (ss_var, get_dr_expression(nominal_polishing_block, ss_var))
        for ss_var in nominal_eff_var_partitioning.second_stage_variables
    ]

    dr_eq_var_zip = zip(polishing_vars, eff_ss_var_to_dr_expr_pairs)
    polishing_model.polishing_abs_val_lb_cons = all_lb_cons = []
    polishing_model.polishing_abs_val_ub_cons = all_ub_cons = []
    for idx, (indexed_polishing_var, (ss_var, dr_expr)) in enumerate(dr_eq_var_zip):
        # set up absolute value constraint components
        polishing_absolute_value_lb_cons = Constraint(indexed_polishing_var.index_set())
        polishing_absolute_value_ub_cons = Constraint(indexed_polishing_var.index_set())

        # add indexed constraints to polishing model
        polishing_model.add_component(
            f"polishing_abs_val_lb_con_{idx}", polishing_absolute_value_lb_cons
        )
        polishing_model.add_component(
            f"polishing_abs_val_ub_con_{idx}", polishing_absolute_value_ub_cons
        )

        # update list of absolute value (i.e., polishing) cons
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)

        for dr_monomial in dr_expr.args:
            is_a_nonstatic_dr_term = dr_monomial.is_expression_type()
            if is_a_nonstatic_dr_term:
                # degree >= 1 monomial expression of form
                # (product of uncertain params) * dr variable
                dr_var_in_term = dr_monomial.args[-1]
            else:
                # the static term (intercept)
                dr_var_in_term = dr_monomial

            # we want the DR variable and corresponding polishing
            # constraints to have the same index in the indexed
            # components
            dr_var_in_term_idx = dr_var_in_term.index()
            polishing_var = indexed_polishing_var[dr_var_in_term_idx]

            # Fix DR variable if:
            # (1) it has already been fixed from master due to
            #     DR efficiencies (already done)
            # (2) coefficient of term
            #     (i.e. product of uncertain parameter values)
            #     in DR expression is 0
            #     across all master blocks
            dr_term_copies = [
                (
                    scenario_blk.second_stage.decision_rule_eqns[idx].body.args[
                        dr_var_in_term_idx
                    ]
                )
                for scenario_blk in master_model.scenarios.values()
            ]
            all_copy_coeffs_zero = is_a_nonstatic_dr_term and all(
                abs(value(prod(term.args[:-1]))) <= DR_POLISHING_PARAM_PRODUCT_ZERO_TOL
                for term in dr_term_copies
            )
            if all_copy_coeffs_zero:
                # increment static DR variable value
                # to maintain feasibility of the initial point
                # as much as possible
                static_dr_var_in_expr = dr_expr.args[0]
                static_dr_var_in_expr.set_value(
                    value(static_dr_var_in_expr) + value(dr_monomial)
                )
                dr_var_in_term.fix(0)

            # add polishing constraints
            polishing_absolute_value_lb_cons[dr_var_in_term_idx] = (
                -polishing_var - dr_monomial <= 0
            )
            polishing_absolute_value_ub_cons[dr_var_in_term_idx] = (
                dr_monomial - polishing_var <= 0
            )

            # some DR variables may be fixed,
            # due to the PyROS DR order efficiency instituted
            # in the first few iterations.
            # these need not be polished
            if dr_var_in_term.fixed or not is_a_nonstatic_dr_term:
                polishing_var.fix()
                polishing_absolute_value_lb_cons[dr_var_in_term_idx].deactivate()
                polishing_absolute_value_ub_cons[dr_var_in_term_idx].deactivate()

            # ensure polishing var properly initialized
            polishing_var.set_value(abs(value(dr_monomial)))

    # L1-norm objective
    # TODO: if dropping nonstatic terms, ensure the
    #       corresponding polishing variables are excluded
    #       from this expression
    polishing_model.polishing_obj = Objective(
        expr=sum(sum(polishing_var.values()) for polishing_var in polishing_vars)
    )

    return polishing_model


def minimize_dr_vars(master_data):
    """
    Polish decision rule of most recent master problem solution.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.

    Returns
    -------
    results : SolverResults
        Subordinate solver results for the polishing problem.
    polishing_successful : bool
        True if polishing model was solved to acceptable level,
        False otherwise.
    """
    config = master_data.config

    # create polishing NLP
    polishing_model = construct_dr_polishing_problem(master_data)

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

    config.progress_logger.debug("Solving DR polishing problem")

    # polishing objective should be consistent with value of sum
    # of absolute values of polynomial DR terms provided
    # auxiliary variables initialized correctly
    polishing_obj = polishing_model.polishing_obj
    config.progress_logger.debug(f" Initial DR norm: {value(polishing_obj)}")

    # === Solve the polishing model
    results = call_solver(
        model=polishing_model,
        solver=solver,
        config=config,
        timing_obj=master_data.timing,
        timer_name="main.dr_polishing",
        err_msg=(
            f"Optimizer {repr(solver)} encountered an exception "
            "attempting to solve decision rule polishing problem "
            f"in iteration {master_data.iteration}"
        ),
    )

    # interested in the time and termination status for debugging
    # purposes
    config.progress_logger.debug(" Done solving DR polishing problem")

    # === Process solution by termination condition
    acceptable = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal}
    if results.solver.termination_condition not in acceptable:
        # continue with "unpolished" master model solution
        config.progress_logger.debug(
            "Could not successfully solve DR polishing problem "
            f"of iteration {master_data.iteration} with primary subordinate "
            f"{'global' if config.solve_master_globally else 'local'} solver "
            "to acceptable level. "
            f"Termination stats:\n{results.solver}\n"
            "Maintaining unpolished master problem solution."
        )
        return results, False

    # update master model second-stage, state, and decision rule
    # variables to polishing model solution
    polishing_model.solutions.load_from(results)

    # update master problem variable values
    for idx, blk in master_data.master_model.scenarios.items():
        master_adjustable_vars = blk.all_adjustable_variables
        polishing_adjustable_vars = polishing_model.scenarios[
            idx
        ].all_adjustable_variables
        adjustable_vars_zip = zip(master_adjustable_vars, polishing_adjustable_vars)
        for master_var, polish_var in adjustable_vars_zip:
            master_var.set_value(value(polish_var))
        dr_var_zip = zip(
            blk.first_stage.decision_rule_vars,
            polishing_model.scenarios[idx].first_stage.decision_rule_vars,
        )
        for master_dr, polish_dr in dr_var_zip:
            for mvar, pvar in zip(master_dr.values(), polish_dr.values()):
                mvar.set_value(value(pvar), skip_validation=True)

    config.progress_logger.debug(f" Optimized DR norm: {value(polishing_obj)}")
    log_master_solve_results(polishing_model, config, results, desc="polished")

    return results, True


def get_master_dr_degree(master_data):
    """
    Determine DR polynomial degree to enforce based on
    the iteration number and/or the presence of first-stage
    equality constraints that depend on the decision rule variables.

    If there are first-stage equality constraints that depend
    on the decision rule variables, such as equalities derived
    from coefficient matching or discretization of state-variable
    independent equalities, then the degree is set to
    ``config.decision_rule_order``.

    Otherwise, the degree is set to:

    - 0 if iteration number is 0
    - min(1, config.decision_rule_order) if iteration number
      otherwise does not exceed number of effective
      uncertain parameters
    - min(2, config.decision_rule_order) otherwise.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.

    Returns
    -------
    int
        DR order, or polynomial degree, to enforce.
    """
    nom_scenario_blk = master_data.master_model.scenarios[0, 0]
    if nom_scenario_blk.first_stage.dr_dependent_equality_cons:
        return master_data.config.decision_rule_order

    if master_data.iteration == 0:
        return 0
    elif master_data.iteration <= len(nom_scenario_blk.effective_uncertain_params):
        return min(1, master_data.config.decision_rule_order)
    else:
        return min(2, master_data.config.decision_rule_order)


def higher_order_decision_rule_efficiency(master_data):
    """
    Enforce DR coefficient variable efficiencies for
    master problem-like formulation.

    Parameters
    ----------
    master_data : MasterProblemData
        Master problem data.

    Note
    ----
    The DR coefficient variable efficiencies consist of
    setting the degree of the DR polynomial expressions
    by fixing the appropriate variables to 0. The degree
    to be set depends on the iteration number;
    see ``get_master_dr_degree``.
    """
    order_to_enforce = get_master_dr_degree(master_data)
    enforce_dr_degree(
        working_blk=master_data.master_model.scenarios[0, 0],
        config=master_data.config,
        degree=order_to_enforce,
    )


def log_master_solve_results(master_model, config, results, desc="Optimized"):
    """
    Log master problem solve results.
    """
    if config.objective_focus == ObjectiveType.worst_case:
        eval_obj_blk_idx = max(
            master_model.scenarios.keys(),
            key=lambda idx: value(master_model.scenarios[idx].second_stage_objective),
        )
    else:
        eval_obj_blk_idx = (0, 0)

    eval_obj_blk = master_model.scenarios[eval_obj_blk_idx]
    config.progress_logger.debug(f" {desc.capitalize()} master objective breakdown:")
    config.progress_logger.debug(
        f"  First-stage objective: {value(eval_obj_blk.first_stage_objective)}"
    )
    config.progress_logger.debug(
        f"  Second-stage objective: {value(eval_obj_blk.second_stage_objective)}"
    )
    master_obj = eval_obj_blk.full_objective
    config.progress_logger.debug(f"  Overall Objective: {value(master_obj)}")
    config.progress_logger.debug(
        f" Termination condition: {results.solver.termination_condition}"
    )
    config.progress_logger.debug(
        f" Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s"
    )


def solver_call_master(master_data):
    """
    Invoke subsolver(s) on PyROS master problem,
    and update the MasterResults object accordingly.

    Parameters
    ----------
    master_data : MasterProblemData
        Container for current master problem and related data.

    Returns
    -------
    master_soln : MasterResults
        Master solution results object.
    """
    config = master_data.config
    master_model = master_data.master_model
    master_soln = MasterResults(
        master_model=master_model, pyros_termination_condition=None
    )

    config.progress_logger.debug("Solving master problem")

    higher_order_decision_rule_efficiency(master_data)

    acceptable_optimal_terminations = {tc.optimal, tc.globallyOptimal}
    if not config.solve_master_globally:
        acceptable_optimal_terminations.add(tc.locallyOptimal)

    solver_results_list, solve_status = call_solver_and_backups(
        model=master_model,
        config=config,
        solve_globally=config.solve_master_globally,
        timing_obj=master_data.timing,
        timer_name="main.master",
        problem_desc=f"master problem of iteration {master_data.iteration}",
        acceptable_terminations=acceptable_optimal_terminations | {tc.infeasible},
        backup_solver_terminations={
            tc.feasible,
            tc.maxTimeLimit,
            tc.maxIterations,
            tc.maxEvaluations,
            tc.minStepLength,
            tc.minFunctionValue,
            tc.other,
            tc.solverFailure,
            tc.internalSolverError,
            tc.error,
            tc.unbounded,
            tc.infeasibleOrUnbounded,
            tc.invalidProblem,
            tc.intermediateNonInteger,
            tc.noSolution,
            tc.unknown,
        },
    )

    master_soln = MasterResults(
        master_model=master_model, master_results_list=solver_results_list
    )

    final_solve_res = solver_results_list[-1]
    if solve_status == SolverCallStatus.SUCCESSFUL:
        infeasible = final_solve_res.solver.termination_condition == tc.infeasible
        optimal = (
            final_solve_res.solver.termination_condition
            in acceptable_optimal_terminations
        )
        if optimal:
            master_soln.pyros_termination_condition = None
            master_model.solutions.load_from(final_solve_res)
            log_master_solve_results(master_model, config, final_solve_res)
        elif infeasible:
            master_soln.pyros_termination_condition = (
                pyrosTerminationCondition.robust_infeasible
            )

    if solve_status == SolverCallStatus.TIME_OUT:
        master_soln.pyros_termination_condition = pyrosTerminationCondition.time_out

    if solve_status == SolverCallStatus.SUBSOLVER_ERROR:
        solve_mode = "global" if config.solve_master_globally else "local"
        master_soln.pyros_termination_condition = (
            pyrosTerminationCondition.subsolver_error
        )
        # log subproblem solve failure warning
        deterministic_model_qual = (
            " (i.e., the deterministic model)" if master_data.iteration == 0 else ""
        )
        deterministic_msg = (
            (
                " Please ensure that your deterministic model, "
                "subject to the nominal uncertain parameter realization "
                "you have provided, "
                f"is solvable by at least one of the subordinate {solve_mode} "
                "optimizers provided."
            )
            if master_data.iteration == 0
            else ""
        )
        config.progress_logger.warning(
            f"Could not successfully solve master problem of iteration "
            f"{master_data.iteration}{deterministic_model_qual} with any of the "
            f"provided subordinate {solve_mode} optimizers. "
            f"(Termination statuses: "
            f"{[res.solver.termination_condition for res in solver_results_list]}.)"
            f"{deterministic_msg}"
        )

        # at this point, export subproblem to file, if desired.
        # NOTE: subproblem is written with variables set to their
        #       initial values (not the final subsolver iterate)
        if config.keepfiles and config.subproblem_file_directory is not None:
            write_subproblem(
                model=master_model,
                fname=(
                    f"{config.uncertainty_set.type}"
                    f"_{master_data.original_model_name}"
                    f"_master_{master_data.iteration}"
                ),
                config=config,
            )

    return master_soln


def solve_master(master_data):
    """
    Solve the master problem.

    Returns
    -------
    master_soln : MasterResults
        Master problem solve results.
    """
    feasibility_problem_results = None
    time_out_after_feasibility = False
    # if master_data.iteration > 0:
    #     feasibility_problem_results = solve_master_feasibility_problem(master_data)
    #     time_out_after_feasibility = check_time_limit_reached(
    #         master_data.timing, master_data.config
    #     )

    if time_out_after_feasibility:
        master_soln = MasterResults(
            master_model=master_data.master_model,
            feasibility_problem_results=feasibility_problem_results,
            master_results_list=None,
            pyros_termination_condition=pyrosTerminationCondition.time_out,
        )
    else:
        master_soln = solver_call_master(master_data)
        master_soln.feasibility_problem_results = feasibility_problem_results

    return master_soln


class MasterProblemData:
    """
    Container for objects pertaining to the PyROS master problem.

    Parameters
    ----------
    model_data : ModelData
        PyROS model data object, equipped with the
        fully preprocessed working model.

    Attributes
    ----------
    master_model : BlockData
        Master problem model object.
    original_model_name : str
        Name of the user-provided deterministic model object.
    iteration : int
        Index of the current PyROS cutting set iteration.
    timing : TimingData
        Main timer for the current problem being solved.
    config : ConfigDict
        PyROS solver options.
    """

    def __init__(self, model_data):
        """Initialize self (see docstring)."""
        self.master_model = construct_initial_master_problem(model_data)
        # we track the original model name for serialization purposes
        self.original_model_name = model_data.original_model.name
        self.iteration = 0
        self.timing = model_data.timing
        self.config = model_data.config

    def solve_master(self):
        """
        Solve the master problem.
        """
        return solve_master(self)

    def solve_dr_polishing(self):
        """
        Solve the DR polishing problem.
        """
        return minimize_dr_vars(self)
