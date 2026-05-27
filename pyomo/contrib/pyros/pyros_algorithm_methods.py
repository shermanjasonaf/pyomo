# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""
Methods for execution of the main PyROS cutting set algorithm.
"""

from collections import namedtuple

from pyomo.common.dependencies import numpy as np
from pyomo.common.collections import ComponentMap
from pyomo.core.base import value

import pyomo.contrib.pyros.master_problem_methods as mp_methods
import pyomo.contrib.pyros.separation_problem_methods as sp_methods
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
    check_time_limit_reached,
    ObjectiveType,
    pyrosTerminationCondition,
    IterationLogRecord,
    get_main_elapsed_time,
    get_dr_var_to_monomial_map,
)


class GRCSResults:
    """
    Cutting set RO algorithm solve results.

    Attributes
    ----------
    master_results : MasterResults
        Solve results for most recent master problem.
    separation_results : SeparationResults or None
        Solve results for separation problem(s) of last iteration.
        If the separation subroutine was not invoked in the last
        iteration, then None.
    pyros_termination_condition : pyrosTerminationCondition
        PyROS termination condition.
    iterations : int
        Number of iterations required.
    """

    def __init__(
        self,
        master_results,
        separation_results,
        pyros_termination_condition,
        iterations,
    ):
        self.master_results = master_results
        self.separation_results = separation_results
        self.pyros_termination_condition = pyros_termination_condition
        self.iterations = iterations


def _evaluate_shift(current, prev, initial, norm=None):
    if current.size == 0:
        return None
    else:
        normalizers = np.max(
            np.vstack((np.ones(initial.size), np.abs(initial))), axis=0
        )
        return np.max(np.abs(current - prev) / normalizers)


VariableValueData = namedtuple(
    "VariableValueData",
    ("first_stage_variables", "second_stage_variables", "decision_rule_monomials"),
)


def get_variable_value_data(working_blk, dr_var_to_monomial_map):
    """
    Get variable value data.
    """
    ep = working_blk.effective_var_partitioning

    first_stage_data = ComponentMap(
        (var, var.value) for var in ep.first_stage_variables
    )
    second_stage_data = ComponentMap(
        (var, var.value) for var in ep.second_stage_variables
    )
    dr_term_data = ComponentMap(
        (dr_var, value(monomial))
        for dr_var, monomial in get_dr_var_to_monomial_map(working_blk).items()
    )

    return VariableValueData(
        first_stage_variables=first_stage_data,
        second_stage_variables=second_stage_data,
        decision_rule_monomials=dr_term_data,
    )


def evaluate_variable_shifts(current_var_data, previous_var_data, initial_var_data):
    """
    Evaluate relative changes in the variable values
    across solutions to a working model block, such as the
    nominal master block.
    """
    if previous_var_data is None:
        return None, None, None
    else:
        var_shifts = []
        for attr in current_var_data._fields:
            var_shifts.append(
                _evaluate_shift(
                    current=np.array(list(getattr(current_var_data, attr).values())),
                    prev=np.array(list(getattr(previous_var_data, attr).values())),
                    initial=np.array(list(getattr(initial_var_data, attr).values())),
                )
            )

    return tuple(var_shifts)


def ROSolver_iterative_solve(model_data):
    """
    Solve an RO problem with the iterative GRCS algorithm.

    Parameters
    ----------
    model_data : model data object
        Model data object, equipped with the
        fully preprocessed working model.

    Returns
    -------
    GRCSResults
        Iterative solve results.
    """
    config = model_data.config
    master_data = mp_methods.MasterProblemData(model_data)
    separation_data = sp_methods.SeparationProblemData(model_data)

    # set up first-stage variable and DR variable sets
    nominal_master_blk = master_data.master_model.scenarios[0, 0]
    dr_var_monomial_map = get_dr_var_to_monomial_map(nominal_master_blk)

    # keep track of variable values for iteration logging
    first_iter_var_data = None
    previous_iter_var_data = None
    current_iter_var_data = None

    num_second_stage_ineq_cons = len(
        separation_data.separation_model.second_stage.inequality_cons
    )
    IterationLogRecord.log_header(config.progress_logger.info)
    k = 0
    while config.max_iter == -1 or k < config.max_iter:
        master_data.iteration = k
        config.progress_logger.debug(f"PyROS working on iteration {k}...")

        master_soln = master_data.solve_master()
        master_termination_not_acceptable = master_soln.pyros_termination_condition in {
            pyrosTerminationCondition.robust_infeasible,
            pyrosTerminationCondition.time_out,
            pyrosTerminationCondition.subsolver_error,
        }
        if master_termination_not_acceptable:
            iter_log_record = IterationLogRecord(
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
                master_backup_solver=master_soln.backup_solver_used,
                master_feasibility_success=master_soln.feasibility_problem_success,
                separation_backup_local_solver=None,
                separation_backup_global_solver=None,
            )
            iter_log_record.log(config.progress_logger.info)
            return GRCSResults(
                master_results=master_soln,
                separation_results=None,
                pyros_termination_condition=master_soln.pyros_termination_condition,
                iterations=k + 1,
            )

        polishing_successful = True
        polish_master_solution = (
            mp_methods.get_master_dr_degree(master_data) > 0
            and nominal_master_blk.first_stage.decision_rule_vars
            and k != 0
            and False
        )
        if polish_master_solution:
            _, polishing_successful = master_data.solve_dr_polishing()

        # track variable values
        current_iter_var_data = get_variable_value_data(
            nominal_master_blk, dr_var_monomial_map
        )
        if k == 0:
            first_iter_var_data = current_iter_var_data
            previous_iter_var_data = None

        fsv_shift, ssv_shift, dr_var_shift = evaluate_variable_shifts(
            current_var_data=current_iter_var_data,
            previous_var_data=previous_iter_var_data,
            initial_var_data=first_iter_var_data,
        )

        # === Check if time limit reached after polishing
        if check_time_limit_reached(model_data.timing, config):
            iter_log_record = IterationLogRecord(
                iteration=k,
                objective=value(master_data.master_model.epigraph_obj),
                first_stage_var_shift=fsv_shift,
                second_stage_var_shift=ssv_shift,
                dr_var_shift=dr_var_shift,
                num_violated_cons=None,
                max_violation=None,
                dr_polishing_success=polishing_successful,
                all_sep_problems_solved=None,
                global_separation=None,
                elapsed_time=model_data.timing.get_main_elapsed_time(),
                master_backup_solver=master_soln.backup_solver_used,
                master_feasibility_success=master_soln.feasibility_problem_success,
                separation_backup_local_solver=None,
                separation_backup_global_solver=None,
            )
            iter_log_record.log(config.progress_logger.info)
            return GRCSResults(
                master_results=master_soln,
                separation_results=None,
                pyros_termination_condition=pyrosTerminationCondition.time_out,
                iterations=k + 1,
            )

        # === Solve Separation Problem
        separation_data.iteration = k
        separation_data.master_model = master_data.master_model
        separation_results = separation_data.solve_separation(master_data)

        scaled_violations = [
            solve_call_res.scaled_violations[con]
            for con, solve_call_res in separation_results.main_loop_results.solver_call_results.items()
            if solve_call_res.scaled_violations is not None
        ]
        if scaled_violations:
            max_sep_con_violation = max(scaled_violations)
        else:
            max_sep_con_violation = None
        num_violated_cons = len(separation_results.violated_second_stage_ineq_cons)

        all_sep_problems_solved = (
            len(scaled_violations) == num_second_stage_ineq_cons
            and not separation_results.subsolver_error
            and not separation_results.time_out
        ) or separation_results.all_discrete_scenarios_exhausted

        iter_log_record = IterationLogRecord(
            iteration=k,
            objective=value(master_data.master_model.epigraph_obj),
            first_stage_var_shift=fsv_shift,
            second_stage_var_shift=ssv_shift,
            dr_var_shift=dr_var_shift,
            num_violated_cons=num_violated_cons,
            max_violation=max_sep_con_violation,
            dr_polishing_success=polishing_successful,
            all_sep_problems_solved=all_sep_problems_solved,
            global_separation=separation_results.solved_globally,
            elapsed_time=get_main_elapsed_time(model_data.timing),
            master_backup_solver=master_soln.backup_solver_used,
            master_feasibility_success=master_soln.feasibility_problem_success,
            separation_backup_local_solver=separation_results.backup_local_solver_used,
            separation_backup_global_solver=(
                separation_results.backup_global_solver_used
            ),
        )

        # terminate on time limit
        if separation_results.time_out or separation_results.subsolver_error:
            if separation_results.subsolver_error and not separation_results.time_out:
                # report PyROS failure to find violated
                # constraint for subsolver error
                config.progress_logger.warning(
                    "PyROS failed to find a constraint violation and "
                    "will terminate with sub-solver error."
                )

            pyros_term_cond = (
                pyrosTerminationCondition.time_out
                if separation_results.time_out
                else pyrosTerminationCondition.subsolver_error
            )
            iter_log_record.log(config.progress_logger.info)
            return GRCSResults(
                master_results=master_soln,
                separation_results=separation_results,
                pyros_termination_condition=pyros_term_cond,
                iterations=k + 1,
            )

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
            iter_log_record.log(config.progress_logger.info)
            return GRCSResults(
                master_results=master_soln,
                separation_results=separation_results,
                pyros_termination_condition=termination_condition,
                iterations=k + 1,
            )

        def gen_uniq_scenario_ineq_cons(sep_res):
            idx_to_ss_ineq_map = {
                ssidx: ss_ineq
                for ssidx, ss_ineq in enumerate(sep_res.sorted_ss_ineq_cons)
            }
            scenarios = np.array([
                sep_res
                .main_loop_results
                .solver_call_results[ss_ineq]
                .violating_param_realization
                for ss_ineq in sep_res.sorted_ss_ineq_cons
            ])
            for pidx, pt in enumerate(scenarios):
                is_already_found = any([
                    np.allclose(pt, other_pt) for other_pt in scenarios[:pidx]
                ])
                if not is_already_found:
                    yield idx_to_ss_ineq_map[pidx]

        # add scenario blocks to the master problem according to
        # separation problem solutions
        max_scenarios_per_iter = 5
        for idx, ss_ineq_con in enumerate(
            # filter essentially duplicate uncertain parameter realizations
            gen_uniq_scenario_ineq_cons(separation_results)
        ):
            solver_call_res = separation_results.main_loop_results.solver_call_results[
                ss_ineq_con
            ]
            violating_realization = solver_call_res.violating_param_realization
            new_master_scenario_idx = (k + 1, idx)

            mp_methods.add_scenario_block_to_master_problem(
                master_model=master_data.master_model,
                scenario_idx=new_master_scenario_idx,
                param_realization=violating_realization,
                from_block=nominal_master_blk,
                clone_first_stage_components=False,
            )

            separation_data.points_added_to_master[new_master_scenario_idx] = (
                violating_realization
            )
            if config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS:
                separation_data.idxs_of_master_scenarios.append(
                    solver_call_res.discrete_set_scenario_index
                )
            separation_data.auxiliary_values_for_master_points[
                new_master_scenario_idx
            ] = solver_call_res.auxiliary_param_values

            # initialize second-stage and state variables
            # for new master block to separation
            # solution chosen by heuristic. consequently,
            # equality constraints should all be satisfied (up to tolerances).
            for var, val in solver_call_res.variable_values.items():
                master_var = master_data.master_model.scenarios[
                    new_master_scenario_idx
                ].find_component(var)
                master_var.set_value(val)
            config.progress_logger.debug(
                f"Appended to the master model a block with index "
                f"{new_master_scenario_idx} for "
                f"scenario {violating_realization} "
                "from solution to separation problem mapped to inequality "
                f"{ss_ineq_con.index()!r}."
            )

            if idx + 1 == max_scenarios_per_iter:
                break

        config.progress_logger.debug("All scenarios of master model so far:")
        for blk_idx, pt in separation_data.points_added_to_master.items():
            config.progress_logger.debug(f" {blk_idx}, {pt}")

        k += 1

        iter_log_record.log(config.progress_logger.info)
        previous_iter_var_data = current_iter_var_data

    # Iteration limit reached
    return GRCSResults(
        master_results=master_soln,
        separation_results=separation_results,
        pyros_termination_condition=pyrosTerminationCondition.max_iter,
        iterations=k,  # iteration count was already incremented
    )
