'''
Methods for the execution of the grcs algorithm
'''

from pyomo.core.base import Objective, ConstraintList, Var, Constraint, Block
from pyomo.opt.results import TerminationCondition
from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData, MasterResult
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
    ObjectiveType,
    get_time_from_solver,
    pyrosTerminationCondition,
    IterationLogRecord,
    generate_all_decision_rule_vars,
    generate_all_decision_rule_eqns,
    get_decision_rule_to_second_stage_var_map,
)
from pyomo.contrib.pyros.util import get_main_elapsed_time
from pyomo.core.base import value
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.var import _VarData as VarData
from itertools import chain
from pyomo.common.dependencies import numpy as np


def update_grcs_solve_data(
    pyros_soln, term_cond, nominal_data, timing_data, separation_data, master_soln, k
):
    '''
    This function updates the results data container object to return to the user so that they have all pertinent
    information from the PyROS run.
    :param grcs_soln: PyROS solution data container object
    :param term_cond: PyROS termination condition
    :param nominal_data: Contains information on all nominal data (var values, objective)
    :param timing_data: Contains timing information on subsolver calls in PyROS
    :param separation_data: Separation model data container
    :param master_problem_subsolver_statuses: All master problem sub-solver termination conditions from the PyROS run
    :param separation_problem_subsolver_statuses: All separation problem sub-solver termination conditions from the PyROS run
    :param k: Iteration counter
    :return: None
    '''
    pyros_soln.pyros_termination_condition = term_cond
    pyros_soln.total_iters = k
    pyros_soln.nominal_data = nominal_data
    pyros_soln.timing_data = timing_data
    pyros_soln.separation_data = separation_data
    pyros_soln.master_soln = master_soln

    return


def get_dr_var_to_scaled_expr_map(
    decision_rule_eqns, second_stage_vars, uncertain_params, decision_rule_vars
):
    """
    Generate mapping from decision rule variables
    to their terms in a model's DR expression.
    """
    var_to_scaled_expr_map = ComponentMap()
    ssv_dr_eq_zip = zip(second_stage_vars, decision_rule_eqns)
    for ssv_idx, (ssv, dr_eq) in enumerate(ssv_dr_eq_zip):
        for term in dr_eq.body.args:
            is_ssv_term = (
                isinstance(term.args[0], int)
                and term.args[0] == -1
                and isinstance(term.args[1], VarData)
            )
            if not is_ssv_term:
                dr_var = term.args[1]
                var_to_scaled_expr_map[dr_var] = term

    return var_to_scaled_expr_map


def evaluate_first_stage_var_shift(
    current_master_fsv_vals, previous_master_fsv_vals, first_iter_master_fsv_vals
):
    """
    Evaluate first-stage variable "shift": the maximum relative
    difference between first-stage variable values from the current
    and previous master iterations.

    Parameters
    ----------
    current_master_fsv_vals : ComponentMap
        First-stage variable values from the current master
        iteration.
    previous_master_fsv_vals : ComponentMap
        First-stage variable values from the previous master
        iteration.
    first_iter_master_fsv_vals : ComponentMap
        First-stage variable values from the first master
        iteration.

    Returns
    -------
    None
        Returned only if `current_master_fsv_vals` is empty,
        which should occur only if the problem has no first-stage
        variables.
    float
        The maximum relative difference
        Returned only if `current_master_fsv_vals` is not empty.
    """
    if not current_master_fsv_vals:
        # there are no first-stage variables
        return None
    else:
        return max(
            abs(current_master_fsv_vals[var] - previous_master_fsv_vals[var])
            / max((abs(first_iter_master_fsv_vals[var]), 1))
            for var in previous_master_fsv_vals
        )


def evaluate_second_stage_var_shift(
    current_master_nom_ssv_vals,
    previous_master_nom_ssv_vals,
    first_iter_master_nom_ssv_vals,
):
    """
    Evaluate second-stage variable "shift": the maximum relative
    difference between second-stage variable values from the current
    and previous master iterations as evaluated subject to the
    nominal uncertain parameter realization.

    Parameters
    ----------
    current_master_nom_ssv_vals : ComponentMap
        Second-stage variable values from the current master
        iteration, evaluated subject to the nominal uncertain
        parameter realization.
    previous_master_nom_ssv_vals : ComponentMap
        Second-stage variable values from the previous master
        iteration, evaluated subject to the nominal uncertain
        parameter realization.
    first_iter_master_nom_ssv_vals : ComponentMap
        Second-stage variable values from the first master
        iteration, evaluated subject to the nominal uncertain
        parameter realization.

    Returns
    -------
    None
        Returned only if `current_master_nom_ssv_vals` is empty,
        which should occur only if the problem has no second-stage
        variables.
    float
        The maximum relative difference.
        Returned only if `current_master_nom_ssv_vals` is not empty.
    """
    if not current_master_nom_ssv_vals:
        return None
    else:
        return max(
            abs(current_master_nom_ssv_vals[ssv] - previous_master_nom_ssv_vals[ssv])
            / max((abs(first_iter_master_nom_ssv_vals[ssv]), 1))
            for ssv in previous_master_nom_ssv_vals
        )


def evaluate_dr_var_shift(
    current_master_dr_var_vals,
    previous_master_dr_var_vals,
    first_iter_master_nom_ssv_vals,
    dr_var_to_ssv_map,
):
    """
    Evaluate decision rule variable "shift": the maximum relative
    difference between scaled decision rule (DR) variable expressions
    (terms in the DR equations) from the current
    and previous master iterations.

    Parameters
    ----------
    current_master_dr_var_vals : ComponentMap
        DR variable values from the current master
        iteration.
    previous_master_dr_var_vals : ComponentMap
        DR variable values from the previous master
        iteration.
    first_iter_master_nom_ssv_vals : ComponentMap
        Second-stage variable values (evaluated subject to the
        nominal uncertain parameter realization)
        from the first master iteration.
    dr_var_to_ssv_map : ComponentMap
        Mapping from each DR variable to the
        second-stage variable whose value is a function of the
        DR variable.

    Returns
    -------
    None
        Returned only if `current_master_dr_var_vals` is empty,
        which should occur only if the problem has no decision rule
        (or equivalently, second-stage) variables.
    float
        The maximum relative difference.
        Returned only if `current_master_dr_var_vals` is not empty.
    """
    if not current_master_dr_var_vals:
        return None
    else:
        return max(
            abs(current_master_dr_var_vals[drvar] - previous_master_dr_var_vals[drvar])
            / max((1, abs(first_iter_master_nom_ssv_vals[dr_var_to_ssv_map[drvar]])))
            for drvar in previous_master_dr_var_vals
        )


def ROSolver_iterative_solve(model_data, config):
    '''
    GRCS algorithm implementation
    :model_data: ROSolveData object with deterministic model information
    :config: ConfigBlock for the instance being solved
    '''
    # === Build the master problem and master problem data container object
    master_data = master_problem_methods.initial_construct_master(model_data, config)

    # === If using p_robustness, add ConstraintList for additional constraints
    if config.p_robustness:
        master_data.master_model.p_robust_constraints = ConstraintList()

    # set up nominal scenario block
    nominal_master_block = master_data.master_model.scenarios[0, 0]

    # === Make separation problem model once before entering the solve loop
    separation_model = separation_problem_methods.make_separation_problem(
        model_data=model_data,
        config=config,
    )

    # === Create separation problem data container object and add information to catalog during solve
    separation_data = SeparationProblemData()
    separation_data.separation_model = separation_model
    separation_data.points_separated = (
        []
    )  # contains last point separated in the separation problem
    separation_data.points_added_to_master = [
        config.nominal_uncertain_param_vals
    ]  # explicitly robust against in master
    separation_data.constraint_violations = (
        []
    )  # list of constraint violations for each iteration
    separation_data.total_global_separation_solves = (
        0  # number of times global solve is used
    )
    separation_data.timing = master_data.timing  # timing object

    # === Keep track of subsolver termination statuses from each iteration
    separation_data.separation_problem_subsolver_statuses = []

    # for discrete set types, keep track of scenarios added to master
    if config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS:
        separation_data.idxs_of_master_scenarios = [
            config.uncertainty_set.scenarios.index(
                tuple(config.nominal_uncertain_param_vals)
            )
        ]
    else:
        separation_data.idxs_of_master_scenarios = None

    # === Nominal information
    nominal_data = Block()
    nominal_data.nom_fsv_vals = []
    nominal_data.nom_ssv_vals = []
    nominal_data.nom_first_stage_cost = 0
    nominal_data.nom_second_stage_cost = 0
    nominal_data.nom_obj = 0

    # === Time information
    timing_data = Block()
    timing_data.total_master_solve_time = 0
    timing_data.total_separation_local_time = 0
    timing_data.total_separation_global_time = 0
    timing_data.total_dr_polish_time = 0

    dr_var_lists_original = []
    dr_var_lists_polished = []

    # set up first-stage variable and DR variable sets
    master_dr_var_set = ComponentSet(
        generate_all_decision_rule_vars(nominal_master_block)
    )
    master_fsv_set = ComponentSet(
        var for var in nominal_master_block.util.first_stage_variables
    )
    master_nom_ssv_set = ComponentSet(
        nominal_master_block.util.second_stage_variables
    )
    previous_master_fsv_vals = ComponentMap((var, None) for var in master_fsv_set)
    previous_master_dr_var_vals = ComponentMap((var, None) for var in master_dr_var_set)
    previous_master_nom_ssv_vals = ComponentMap(
        (var, None) for var in master_nom_ssv_set
    )

    first_iter_master_fsv_vals = ComponentMap((var, None) for var in master_fsv_set)
    first_iter_master_nom_ssv_vals = ComponentMap(
        (var, None) for var in master_nom_ssv_set
    )
    first_iter_dr_var_vals = ComponentMap((var, None) for var in master_dr_var_set)
    dr_var_scaled_expr_map = get_dr_var_to_scaled_expr_map(
        decision_rule_vars=nominal_master_block.util.decision_rule_vars,
        decision_rule_eqns=nominal_master_block.util.decision_rule_eqns,
        second_stage_vars=nominal_master_block.util.second_stage_variables,
        uncertain_params=nominal_master_block.util.uncertain_params,
    )
    dr_var_to_ssv_map = ComponentMap(
        get_decision_rule_to_second_stage_var_map(
            nominal_master_block,
            indexed_dr_vars=False,
        )
    )

    IterationLogRecord.log_header(config.progress_logger.info)
    k = 0
    master_statuses = []
    while config.max_iter == -1 or k < config.max_iter:
        master_data.iteration = k
        master_problem_methods.enforce_objective_focus(master_data, config)

        # === Add p-robust constraint if iteration > 0
        if k > 0 and config.p_robustness:
            master_problem_methods.add_p_robust_constraint(
                model_data=master_data, config=config
            )

        # === Solve Master Problem
        config.progress_logger.debug(f"PyROS working on iteration {k}...")
        master_soln = master_problem_methods.solve_master(
            model_data=master_data, config=config
        )
        # config.progress_logger.info("Done solving Master Problem!")

        # === Keep track of total time and subsolver termination conditions
        timing_data.total_master_solve_time += get_time_from_solver(master_soln.results)

        if k > 0:  # master feas problem not solved for iteration 0
            timing_data.total_master_solve_time += get_time_from_solver(
                master_soln.feasibility_problem_results
            )

        master_statuses.append(master_soln.results.solver.termination_condition)
        master_soln.master_problem_subsolver_statuses = master_statuses

        # === Check for robust infeasibility or error or time-out in master problem solve
        if (
            master_soln.master_subsolver_results[1]
            is pyrosTerminationCondition.robust_infeasible
        ):
            term_cond = pyrosTerminationCondition.robust_infeasible
        elif (
            master_soln.pyros_termination_condition
            is pyrosTerminationCondition.subsolver_error
        ):
            term_cond = pyrosTerminationCondition.subsolver_error
        elif (
            master_soln.pyros_termination_condition
            is pyrosTerminationCondition.time_out
        ):
            term_cond = pyrosTerminationCondition.time_out
        else:
            term_cond = None
        if term_cond in {
            pyrosTerminationCondition.subsolver_error,
            pyrosTerminationCondition.time_out,
            pyrosTerminationCondition.robust_infeasible,
        }:
            log_record = IterationLogRecord(
                iteration=k,
                objective=None,
                first_stage_var_shift=None,
                second_stage_var_shift=None,
                dr_var_shift=None,
                num_violated_cons=None,
                max_violation=None,
                dr_polishing_success=None,
                all_sep_problems_solved=None,
                global_separation=None,
                elapsed_time=get_main_elapsed_time(model_data.timing),
            )
            log_record.log(config.progress_logger.info)
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=term_cond,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            return model_data, []

        # === Save nominal information
        if k == 0:
            for val in master_soln.fsv_vals:
                nominal_data.nom_fsv_vals.append(val)

            for val in master_soln.ssv_vals:
                nominal_data.nom_ssv_vals.append(val)

            nominal_data.nom_first_stage_cost = master_soln.first_stage_objective
            nominal_data.nom_second_stage_cost = master_soln.second_stage_objective
            nominal_data.nom_obj = value(master_data.master_model.epigraph_obj)

        polishing_successful = True
        if (
            config.decision_rule_order != 0
            and len(config.second_stage_variables) > 0
            and k != 0
        ):
            # === Save initial values of DR vars to file
            for varslist in master_data.master_model.scenarios[
                0, 0
            ].util.decision_rule_vars:
                vals = []
                for dvar in varslist.values():
                    vals.append(dvar.value)
                dr_var_lists_original.append(vals)

            (
                polishing_results,
                polishing_successful,
            ) = master_problem_methods.minimize_dr_vars(
                model_data=master_data, config=config
            )
            timing_data.total_dr_polish_time += get_time_from_solver(polishing_results)

            # === Save after polish
            for varslist in master_data.master_model.scenarios[
                0, 0
            ].util.decision_rule_vars:
                vals = []
                for dvar in varslist.values():
                    vals.append(dvar.value)
                dr_var_lists_polished.append(vals)

        # get current first-stage and DR variable values
        # and compare with previous first-stage and DR variable
        # values
        current_master_fsv_vals = ComponentMap(
            (var, value(var)) for var in master_fsv_set
        )
        current_master_nom_ssv_vals = ComponentMap(
            (var, value(var)) for var in master_nom_ssv_set
        )
        current_master_dr_var_vals = ComponentMap(
            (var, value(expr)) for var, expr in dr_var_scaled_expr_map.items()
        )
        if k > 0:
            first_stage_var_shift = evaluate_first_stage_var_shift(
                current_master_fsv_vals=current_master_fsv_vals,
                previous_master_fsv_vals=previous_master_fsv_vals,
                first_iter_master_fsv_vals=first_iter_master_fsv_vals,
            )
            second_stage_var_shift = evaluate_second_stage_var_shift(
                current_master_nom_ssv_vals=current_master_nom_ssv_vals,
                previous_master_nom_ssv_vals=previous_master_nom_ssv_vals,
                first_iter_master_nom_ssv_vals=first_iter_master_nom_ssv_vals,
            )
            dr_var_shift = evaluate_dr_var_shift(
                current_master_dr_var_vals=current_master_dr_var_vals,
                previous_master_dr_var_vals=previous_master_dr_var_vals,
                first_iter_master_nom_ssv_vals=first_iter_master_nom_ssv_vals,
                dr_var_to_ssv_map=dr_var_to_ssv_map,
            )
        else:
            for fsv in first_iter_master_fsv_vals:
                first_iter_master_fsv_vals[fsv] = value(fsv)
            for ssv in first_iter_master_nom_ssv_vals:
                first_iter_master_nom_ssv_vals[ssv] = value(ssv)
            for drvar in first_iter_dr_var_vals:
                first_iter_dr_var_vals[drvar] = value(dr_var_scaled_expr_map[drvar])
            first_stage_var_shift = None
            second_stage_var_shift = None
            dr_var_shift = None

        # === Check if time limit reached after polishing
        if config.time_limit:
            elapsed = get_main_elapsed_time(model_data.timing)
            if elapsed >= config.time_limit:
                iter_log_record = IterationLogRecord(
                    iteration=k,
                    objective=value(master_data.master_model.epigraph_obj),
                    first_stage_var_shift=first_stage_var_shift,
                    second_stage_var_shift=second_stage_var_shift,
                    dr_var_shift=dr_var_shift,
                    num_violated_cons=None,
                    max_violation=None,
                    dr_polishing_success=polishing_successful,
                    all_sep_problems_solved=None,
                    global_separation=None,
                    elapsed_time=elapsed,
                )
                update_grcs_solve_data(
                    pyros_soln=model_data,
                    k=k,
                    term_cond=pyrosTerminationCondition.time_out,
                    nominal_data=nominal_data,
                    timing_data=timing_data,
                    separation_data=separation_data,
                    master_soln=master_soln,
                )
                iter_log_record.log(config.progress_logger.info)
                return model_data, []

        # === Set up for the separation problem
        separation_data.opt_fsv_vals = [
            v.value
            for v in master_soln.master_model.scenarios[0, 0].util.first_stage_variables
        ]
        separation_data.opt_ssv_vals = master_soln.ssv_vals

        # === Provide master model scenarios to separation problem for initialization options
        separation_data.master_scenarios = master_data.master_model.scenarios

        # === Solve Separation Problem
        separation_data.iteration = k
        separation_data.master_nominal_scenario = master_data.master_model.scenarios[
            0, 0
        ]

        separation_data.master_model = master_data.master_model

        separation_results = separation_problem_methods.solve_separation_problem(
            model_data=separation_data, config=config
        )

        separation_data.separation_problem_subsolver_statuses.extend(
            [
                res.solver.termination_condition
                for res in separation_results.generate_subsolver_results()
            ]
        )

        if separation_results.solved_globally:
            separation_data.total_global_separation_solves += 1

        # make updates based on separation results
        timing_data.total_separation_local_time += (
            separation_results.evaluate_local_solve_time(get_time_from_solver)
        )
        timing_data.total_separation_global_time += (
            separation_results.evaluate_global_solve_time(get_time_from_solver)
        )
        if separation_results.found_violation:
            scaled_violations = separation_results.scaled_violations
            if scaled_violations is not None:
                # can be None if time out or subsolver error
                # reported in separation
                separation_data.constraint_violations.append(scaled_violations.values())
        separation_data.points_separated = (
            separation_results.violating_param_realization
        )

        scaled_violations = [
            solve_call_res.scaled_violations[con]
            for con, solve_call_res in separation_results.main_loop_results.solver_call_results.items()
            if solve_call_res.scaled_violations is not None
        ]
        if scaled_violations:
            max_sep_con_violation = max(scaled_violations)
        else:
            max_sep_con_violation = None
        num_violated_cons = len(separation_results.violated_performance_constraints)

        all_sep_problems_solved = (
            len(scaled_violations) == len(separation_model.util.performance_constraints)
            and not separation_results.subsolver_error
            and not separation_results.time_out
        )

        iter_log_record = IterationLogRecord(
            iteration=k,
            objective=value(master_data.master_model.epigraph_obj),
            first_stage_var_shift=first_stage_var_shift,
            second_stage_var_shift=second_stage_var_shift,
            dr_var_shift=dr_var_shift,
            num_violated_cons=num_violated_cons,
            max_violation=max_sep_con_violation,
            dr_polishing_success=polishing_successful,
            all_sep_problems_solved=all_sep_problems_solved,
            global_separation=separation_results.solved_globally,
            elapsed_time=get_main_elapsed_time(model_data.timing),
        )

        # terminate on time limit
        elapsed = get_main_elapsed_time(model_data.timing)
        if separation_results.time_out:
            termination_condition = pyrosTerminationCondition.time_out
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=termination_condition,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            iter_log_record.log(config.progress_logger.info)
            return model_data, separation_results

        # terminate on separation subsolver error
        if separation_results.subsolver_error:
            termination_condition = pyrosTerminationCondition.subsolver_error
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=termination_condition,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            iter_log_record.log(config.progress_logger.info)
            return model_data, separation_results

        # === Check if we terminate due to robust optimality or feasibility,
        #     or in the event of bypassing global separation, no violations
        robustness_certified = separation_results.robustness_certified
        if robustness_certified:
            if config.bypass_global_separation:
                config.progress_logger.warning(
                    "Option to bypass global separation was chosen. "
                    "Robust feasibility and optimality of the reported "
                    "solution are not guaranteed."
                )
            robust_optimal = (
                config.solve_master_globally
                and config.objective_focus is ObjectiveType.worst_case
            )
            if robust_optimal:
                termination_condition = pyrosTerminationCondition.robust_optimal
            else:
                termination_condition = pyrosTerminationCondition.robust_feasible
            update_grcs_solve_data(
                pyros_soln=model_data,
                k=k,
                term_cond=termination_condition,
                nominal_data=nominal_data,
                timing_data=timing_data,
                separation_data=separation_data,
                master_soln=master_soln,
            )
            iter_log_record.log(config.progress_logger.info)
            return model_data, separation_results

        # === Add block to master at violation
        master_problem_methods.add_scenario_to_master(
            master_data=master_data,
            scenario_idx=(k + 1, 0),
            param_realization=separation_results.violating_param_realization,
        )
        separation_data.points_added_to_master.append(
            separation_results.violating_param_realization
        )

        config.progress_logger.debug("Points added to master:")
        config.progress_logger.debug(
            np.array([pt for pt in separation_data.points_added_to_master])
        )

        # initialize second-stage and state variables
        # for new master block to separation
        # solution chosen by heuristic. consequently,
        # equality constraints should all be satisfied (up to tolerances).
        for var, val in separation_results.violating_separation_variable_values.items():
            master_var = master_data.master_model.scenarios[k + 1, 0].find_component(
                var
            )
            master_var.set_value(val)

        k += 1

        iter_log_record.log(config.progress_logger.info)
        previous_master_fsv_vals = current_master_fsv_vals
        previous_master_nom_ssv_vals = current_master_nom_ssv_vals
        previous_master_dr_var_vals = current_master_dr_var_vals

    # Iteration limit reached
    update_grcs_solve_data(
        pyros_soln=model_data,
        k=k - 1,  # remove last increment to fix iteration count
        term_cond=pyrosTerminationCondition.max_iter,
        nominal_data=nominal_data,
        timing_data=timing_data,
        separation_data=separation_data,
        master_soln=master_soln,
    )
    return model_data, separation_results
