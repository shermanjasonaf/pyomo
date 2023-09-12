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


def create_dr_polishing_problem(model_data, config):
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

    # deactivate objective
    polishing_model.obj.deactivate()

    # fix first-stage variables (including epigraph, where applicable)
    first_stage_vars = nominal_polishing_block.util.first_stage_variables

    # fix first-stage and epigraph variables
    for var in first_stage_vars:
        var.fix()

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

    # unfix second-stage variables
    for ss_var in nominal_polishing_block.util.second_stage_variables:
        ss_var.unfix()

    # unfix decision rule variables, taking DR efficiencies into account
    num_uncertain_params = len(nominal_polishing_block.util.uncertain_params)
    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars
    for indexed_var in decision_rule_vars:
        num_dr_vars = len(indexed_var)
        indexed_var.unfix()
        if polishing_model.const_efficiency_applied:
            for i in range(1, num_dr_vars):
                indexed_var[i].fix(0)
        if polishing_model.linear_efficiency_applied:
            for i in range(num_uncertain_params + 1, num_dr_vars):
                indexed_var[i].fix(0)

    # declare auxiliary polishing variables
    polishing_vars = []
    for idx, indexed_dr_var in enumerate(decision_rule_vars):
        dr_var_index_set = indexed_dr_var.index_set()
        indexed_polishing_var = Var(dr_var_index_set, domain=NonNegativeReals)
        nominal_polishing_block.add_component(
            unique_component_name(nominal_polishing_block, f"dr_polishing_var_{idx}"),
            indexed_polishing_var,
        )
        polishing_vars.append(indexed_polishing_var)

    # add polishing absolute value constraints,
    # and initialize auxiliary polishing vars in the process
    dr_eq_var_zip = zip(
        nominal_polishing_block.util.decision_rule_eqns,
        polishing_vars,
    )
    for idx, (dr_eq, indexed_var) in enumerate(dr_eq_var_zip):
        # components for absolute value constraints
        polishing_absolute_value_lb_cons = ConstraintList(starting_index=0)
        polishing_absolute_value_ub_cons = ConstraintList(starting_index=0)
        polishing_absolute_value_lb_cons.construct()
        polishing_absolute_value_ub_cons.construct()

        # ensure second-stage variable term excluded
        dr_expr_terms = dr_eq.body.args[:-1]

        for dr_eq_term, polishing_var in zip(dr_expr_terms, indexed_var.values()):
            # initialize auxiliary variable
            polishing_var.set_value(abs(value(dr_eq_term)))

            # declare absolute value constraints
            polishing_absolute_value_lb_cons.add(-polishing_var <= dr_eq_term)
            polishing_absolute_value_ub_cons.add(dr_eq_term <= polishing_var)

        # add constraints to polishing model
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_absolute_value_lb_con_{idx}",
            ),
            polishing_absolute_value_lb_cons,
        )
        nominal_polishing_block.add_component(
            unique_component_name(
                polishing_model,
                f"polishing_absolute_value_ub_con_{idx}",
            ),
            polishing_absolute_value_ub_cons,
        )

    # declare polishing objective
    polishing_model.polishing_obj = Objective(
        expr=sum(
            sum(polishing_var.values())
            for polishing_var in polishing_vars
        )
    )

    return polishing_model


def minimize_dr_vars(model_data, config):
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
    polishing_model = create_dr_polishing_problem(model_data, config)

    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver

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
        f"  Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)} s"
    )
    config.progress_logger.debug(
        f"  Termination status: {results.solver.termination_condition} "
    )

    # === Process solution by termination condition
    acceptable = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal, tc.feasible}
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


def higher_order_decision_rule_efficiency(config, model_data):
    # === Efficiencies for decision rules
    #  if iteration <= |q| then all d^n where n > 1 are fixed to 0
    #  if iteration == 0, all d^n, n > 0 are fixed to 0
    #  These efficiencies should be carried through as d* to polishing
    nlp_model = model_data.master_model
    if config.decision_rule_order != None and len(config.second_stage_variables) > 0:
        #  Ensure all are unfixed unless next conditions are met...
        for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
            dr_var.unfix()
        num_dr_vars = len(
            nlp_model.scenarios[0, 0].util.decision_rule_vars[0]
        )  # there is at least one dr var
        num_uncertain_params = len(config.uncertain_params)
        nlp_model.const_efficiency_applied = False
        nlp_model.linear_efficiency_applied = False
        if model_data.iteration == 0:
            nlp_model.const_efficiency_applied = True
            for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
                for i in range(1, num_dr_vars):
                    dr_var[i].fix(0)
        elif (
            model_data.iteration <= num_uncertain_params
            and config.decision_rule_order > 1
        ):
            # Only applied in DR order > 1 case
            for dr_var in nlp_model.scenarios[0, 0].util.decision_rule_vars:
                for i in range(num_uncertain_params + 1, num_dr_vars):
                    nlp_model.linear_efficiency_applied = True
                    dr_var[i].fix(0)
    return


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

    higher_order_decision_rule_efficiency(config, model_data)

    solve_mode = "global" if config.solve_master_globally else "local"
    config.progress_logger.debug("Solving master problem")

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
