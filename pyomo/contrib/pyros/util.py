'''
Utility functions for the PyROS solver
'''
import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
    Constraint,
    Var,
    ConstraintList,
    Objective,
    minimize,
    Expression,
    ConcreteModel,
    maximize,
    Block,
    Param,
)
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
    identify_variables,
    identify_mutable_parameters,
    replace_expressions,
)
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory

import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted


# Tolerances used in the code
PARAM_IS_CERTAIN_REL_TOL = 1e-4
PARAM_IS_CERTAIN_ABS_TOL = 0
COEFF_MATCH_REL_TOL = 1e-6
COEFF_MATCH_ABS_TOL = 0
ABS_CON_CHECK_FEAS_TOL = 1e-5
TIC_TOC_SOLVE_TIME_ATTR = "pyros_tic_toc_time"
DEFAULT_LOGGER_NAME = "pyomo.contrib.pyros"


class TimingData:
    """
    PyROS solver timing data object.

    Implemented as a wrapper around `common.timing.HierarchicalTimer`,
    with added functionality for enforcing a standardized
    hierarchy of identifiers.

    Attributes
    ----------
    hierarchical_timer_full_ids : set of str
        (Class attribute.) Valid identifiers for use with
        the encapsulated hierarchical timer.
    """

    hierarchical_timer_full_ids = {
        "main",
        "main.preprocessing",
        "main.master_feasibility",
        "main.master",
        "main.dr_polishing",
        "main.local_separation",
        "main.global_separation",
    }

    def __init__(self):
        """Initialize self (see class docstring)."""
        self._hierarchical_timer = HierarchicalTimer()

    def __str__(self):
        """
        String representation of `self`. Currently
        returns the string representation of `self.hierarchical_timer`.

        Returns
        -------
        str
            String representation.
        """
        return self._hierarchical_timer.__str__()

    def _validate_full_identifier(self, full_identifier):
        """
        Validate identifier for hierarchical timer.

        Parameters
        ----------
        full_identifier : str
            Identifier to validate.

        Raises
        ------
        ValueError
            If identifier not in `TimingData.hierarchical_timer_full_ids`.
        """
        if full_identifier not in self.hierarchical_timer_full_ids:
            raise ValueError(
                "PyROS timing data object does not support timing ID: "
                f"{full_identifier}."
            )

    def start_timer(self, full_identifier):
        """
        Start timer for `self.hierarchical_timer`.

        Parameters
        ----------
        full_identifier : str
            Full identifier for the timer to be started.
            Must be an entry of
            `TimingData.hierarchical_timer_full_ids`.
        """
        self._validate_full_identifier(full_identifier)
        identifier = full_identifier.split(".")[-1]
        return self._hierarchical_timer.start(identifier=identifier)

    def stop_timer(self, full_identifier):
        """
        Stop timer for `self.hierarchical_timer`.

        Parameters
        ----------
        full_identifier : str
            Full identifier for the timer to be stopped.
            Must be an entry of
            `TimingData.hierarchical_timer_full_ids`.
        """
        self._validate_full_identifier(full_identifier)
        identifier = full_identifier.split(".")[-1]
        return self._hierarchical_timer.stop(identifier=identifier)

    def get_total_time(self, full_identifier):
        """
        Get total time spent with identifier active.

        Parameters
        ----------
        full_identifier : str
            Full identifier for the timer of interest.

        Returns
        -------
        float
            Total time spent with identifier active.
        """
        return self._hierarchical_timer.get_total_time(identifier=full_identifier)

    def get_main_elapsed_time(self):
        """
        Get total time elapsed for main timer of
        the HierarchicalTimer contained in self.

        Returns
        -------
        float
            Total elapsed time.

        Note
        ----
        This method is meant for use while the main timer is active.
        Otherwise, use ``self.get_total_time("main")``.
        """
        # clean?
        return self._hierarchical_timer.timers["main"].tic_toc.toc(
            msg=None, delta=False
        )


'''Code borrowed from gdpopt: time_code, get_main_elapsed_time, a_logger.'''


@contextmanager
def time_code(timing_data_obj, code_block_name, is_main_timer=False):
    """
    Starts timer at entry, stores elapsed time at exit.

    Parameters
    ----------
    timing_data_obj : TimingData
        Timing data object.
    code_block_name : str
        Name of code block being timed.

    If `is_main_timer=True`, the start time is stored in the timing_data_obj,
    allowing calculation of total elapsed time 'on the fly' (e.g. to enforce
    a time limit) using `get_main_elapsed_time(timing_data_obj)`.
    """
    # initialize tic toc timer
    timing_data_obj.start_timer(code_block_name)

    start_time = timeit.default_timer()
    if is_main_timer:
        timing_data_obj.main_timer_start_time = start_time
    yield
    timing_data_obj.stop_timer(code_block_name)


def get_main_elapsed_time(timing_data_obj):
    """Returns the time since entering the main `time_code` context"""
    return timing_data_obj.get_main_elapsed_time()


def adjust_solver_time_settings(timing_data_obj, solver, config):
    """
    Adjust solver max time setting based on current PyROS elapsed
    time.

    Parameters
    ----------
    timing_data_obj : Bunch
        PyROS timekeeper.
    solver : solver type
        Solver for which to adjust the max time setting.
    config : ConfigDict
        PyROS solver config.

    Returns
    -------
    original_max_time_setting : float or None
        If IPOPT or BARON is used, a float is returned.
        If GAMS is used, the ``options.add_options`` attribute
        of ``solver`` is returned.
        Otherwise, None is returned.
    custom_setting_present : bool or None
        If IPOPT or BARON is used, True if the max time is
        specified, False otherwise.
        If GAMS is used, True if the attribute ``options.add_options``
        is not None, False otherwise.
        If ``config.time_limit`` is None, then None is returned.

    Note
    ----
    (1) Adjustment only supported for GAMS, BARON, and IPOPT
        interfaces. This routine can be generalized to other solvers
        after a generic interface to the time limit setting
        is introduced.
    (2) For IPOPT, and probably also BARON, the CPU time limit
        rather than the wallclock time limit, is adjusted, as
        no interface to wallclock limit available.
        For this reason, extra 30s is added to time remaining
        for subsolver time limit.
        (The extra 30s is large enough to ensure solver
        elapsed time is not beneath elapsed time - user time limit,
        but not so large as to overshoot the user-specified time limit
        by an inordinate margin.)
    """
    if config.time_limit is not None:
        time_remaining = config.time_limit - get_main_elapsed_time(timing_data_obj)
        if isinstance(solver, type(SolverFactory("gams", solver_io="shell"))):
            original_max_time_setting = solver.options["add_options"]
            custom_setting_present = "add_options" in solver.options

            # adjust GAMS solver time
            reslim_str = f"option reslim={max(30, 30 + time_remaining)};"
            if isinstance(solver.options["add_options"], list):
                solver.options["add_options"].append(reslim_str)
            else:
                solver.options["add_options"] = [reslim_str]
        else:
            # determine name of option to adjust
            if isinstance(solver, SolverFactory.get_class("baron")):
                options_key = "MaxTime"
            elif isinstance(solver, SolverFactory.get_class("ipopt")):
                options_key = "max_cpu_time"
            else:
                options_key = None

            if options_key is not None:
                custom_setting_present = options_key in solver.options
                original_max_time_setting = solver.options[options_key]

                # ensure positive value assigned to avoid application error
                solver.options[options_key] = max(30, 30 + time_remaining)
            else:
                custom_setting_present = False
                original_max_time_setting = None
                config.progress_logger.warning(
                    "Subproblem time limit setting not adjusted for "
                    f"subsolver of type:\n    {type(solver)}.\n"
                    "    PyROS time limit may not be honored "
                )

        return original_max_time_setting, custom_setting_present
    else:
        return None, None


def revert_solver_max_time_adjustment(
    solver, original_max_time_setting, custom_setting_present, config
):
    """
    Revert solver `options` attribute to its state prior to a
    time limit adjustment performed via
    the routine `adjust_solver_time_settings`.

    Parameters
    ----------
    solver : solver type
        Solver of interest.
    original_max_time_setting : float, list, or None
        Original solver settings. Type depends on the
        solver type.
    custom_setting_present : bool or None
        Was the max time, or other custom solver settings,
        specified prior to the adjustment?
        Can be None if ``config.time_limit`` is None.
    config : ConfigDict
        PyROS solver config.
    """
    if config.time_limit is not None:
        assert isinstance(custom_setting_present, bool)

        # determine name of option to adjust
        if isinstance(solver, type(SolverFactory("gams", solver_io="shell"))):
            options_key = "add_options"
        elif isinstance(solver, SolverFactory.get_class("baron")):
            options_key = "MaxTime"
        elif isinstance(solver, SolverFactory.get_class("ipopt")):
            options_key = "max_cpu_time"
        else:
            options_key = None

        if options_key is not None:
            if custom_setting_present:
                # restore original setting
                solver.options[options_key] = original_max_time_setting

                # if GAMS solver used, need to remove the last entry
                # of 'add_options', which contains the max time setting
                # added by PyROS
                if isinstance(solver, type(SolverFactory("gams", solver_io="shell"))):
                    solver.options[options_key].pop()
            else:
                # remove the max time specification introduced.
                # All lines are needed here to completely remove the option
                # from access through getattr and dictionary reference.
                delattr(solver.options, options_key)
                if options_key in solver.options.keys():
                    del solver.options[options_key]


class PreformattedLogger(logging.Logger):
    """
    A specialized logger object designed to cast log messages
    to Pyomo `Preformatted` objects prior to logging the messages.
    Useful for circumventing the formatters of the standard Pyomo
    logger in the event an instance is a descendant of the Pyomo
    logger.
    """

    def critical(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.CRITICAL`.
        """
        return super(PreformattedLogger, self).critical(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def error(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.ERROR`.
        """
        return super(PreformattedLogger, self).error(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def warning(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.WARNING`.
        """
        return super(PreformattedLogger, self).warning(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def info(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.INFO`.
        """
        return super(PreformattedLogger, self).info(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def debug(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.DEBUG`.
        """
        return super(PreformattedLogger, self).debug(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def log(self, level, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with integer
        severity `level`.
        """
        return super(PreformattedLogger, self).log(
            level, Preformatted(msg % args if args else msg), **kwargs
        )


def setup_pyros_logger(name=DEFAULT_LOGGER_NAME):
    """
    Set up pyros logger.
    """
    # default logger: INFO level, with preformatted messages
    current_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(PreformattedLogger)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    logging.setLoggerClass(current_logger_class)

    return logger


class pyrosTerminationCondition(Enum):
    """Enumeration of all possible PyROS termination conditions."""

    robust_feasible = 0
    """Final solution is robust feasible."""

    robust_optimal = 1
    """Final solution is robust optimal."""

    robust_infeasible = 2
    """Problem is robust infeasible."""

    max_iter = 3
    """Maximum number of GRCS iteration reached."""

    subsolver_error = 4
    """Subsolver(s) provided could not solve a subproblem to
    an acceptable termination status."""

    time_out = 5
    """Maximum allowable time exceeded."""

    @property
    def message(self):
        """
        str : Message associated with a given PyROS
        termination condition.
        """
        message_dict = {
            self.robust_optimal: "Robust optimal solution identified.",
            self.robust_feasible: "Robust feasible solution identified.",
            self.robust_infeasible: "Problem is robust infeasible.",
            self.time_out: "Maximum allowable time exceeded.",
            self.max_iter: "Maximum number of iterations reached.",
            self.subsolver_error: (
                "Subordinate optimizer(s) could not solve a subproblem "
                "to an acceptable status."
            ),
        }
        return message_dict[self]


class SeparationStrategy(Enum):
    all_violations = auto()
    max_violation = auto()


class SolveMethod(Enum):
    local_solve = auto()
    global_solve = auto()


class ObjectiveType(Enum):
    worst_case = auto()
    nominal = auto()


def recast_to_min_obj(model, obj):
    """
    Recast model objective to a minimization objective, as necessary.

    Parameters
    ----------
    model : ConcreteModel
        Model of interest.
    obj : ScalarObjective
        Objective of interest.
    """
    if obj.sense is not minimize:
        if isinstance(obj.expr, SumExpression):
            # ensure additive terms in objective
            # are split in accordance with user declaration
            obj.expr = sum(-term for term in obj.expr.args)
        else:
            obj.expr = -obj.expr
        obj.sense = minimize


def get_state_vars(blk, first_stage_variables, second_stage_variables):
    """
    Get state variables of block.
    based on degree-of-freedom specification.

    Parameters
    ----------
    blk : ScalarBlock
        Block with structure of, say, a determinstic model
        passed to PyROS solver, or a master problem scenario
        block.
    first_stage_variables : Iterable of VarData
        First-stage variables.
    second_stage_variables : Iterable of VarData
        Second-stage variables.

    Yields
    ------
    _VarData
        State variable.
    """
    dof_var_set = ComponentSet(first_stage_variables) | ComponentSet(
        second_stage_variables
    )
    seen = set()
    for var in blk.component_data_objects(Var, active=True, descend_into=True):
        is_state_var = not var.fixed and var not in dof_var_set
        if is_state_var and id(var) not in seen:
            seen.add(id(var))
            yield var


def turn_variable_bounds_to_constraints(working_model):
    """
    Cast bounds of all first-stage, second-stage, and state
    variables to inequality constraints.

    Parameters
    ----------
    model_data : ModelData
        Model data object.
    skip_first_stage_vars : bool, optional
        True to exclude first-stage variables from this procedure,
        False otherwise.

    Note
    ----
    This method accounts carefully for:

    - bounds expressions involving uncertain parameters
    - bounds implicitly added through custom Var domain
      specifications.

    All Vars are assumed to have continuous domains.
    """
    all_model_vars = (
        working_model.util.first_stage_variables
        + working_model.util.second_stage_variables
        + working_model.util.state_vars
    )
    uncertain_params_set = ComponentSet(working_model.util.uncertain_params)

    def generate_constraint_expr(var, bound, bound_desc):
        """
        Generate bound constraint expression.
        """
        if bound_desc == "lower":
            return (None, bound - var, 0)
        elif bound_desc == "upper":
            return (None, var - bound, 0)
        elif bound_desc == "eq":
            return (0, var - bound, 0)
        else:
            raise ValueError(f"Bound type {bound_desc!r} not supported.")

    for var in all_model_vars:
        # use upper/lower to get bounds expressions, rather
        # than just their values
        lb = var.lower
        ub = var.upper

        uncertain_params_in_lb = (
            ComponentSet(identify_mutable_parameters(lb))
            & uncertain_params_set
        )
        uncertain_params_in_ub = (
            ComponentSet(identify_mutable_parameters(ub))
            & uncertain_params_set
        )

        lb_args = [lb] if lb is not None else []
        ub_args = [ub] if ub is not None else []

        # lb (ub) could be a max (min) of expressions due to, say,
        # custom Var domain specifications.
        # if the lb (ub) is contingent on the uncertain params,
        # then we will treat each arguemnt of the max (min) separately
        if isinstance(lb, NPV_MaxExpression):
            lb_args = list(lb.args) if uncertain_params_in_lb else lb
        if isinstance(ub, NPV_MinExpression):
            ub_args = list(ub.args) if uncertain_params_in_ub else ub

        # lb args and ub args may have identical entries.
        # for every occurrence of an identical entry,
        # we cast the bound arg requirement to an equality
        # constraint instead of separate lower bound and
        # upper bound inequality constraints.
        # ----
        # args could be numeric type, expression, or
        # mutable param, so instead of using set.in/ComponentSet.in,
        # we use this custom approach to separate
        # identical lb/ub args from the rest
        matched_lb_arg_idxs = []
        matching_lb_ub_args = []
        for lb_idx, lb_arg in enumerate(lb_args):
            # look for matching upper bound args
            matching_ub_arg_idxs = [
                idx for idx, ub_arg in enumerate(ub_args) if lb_arg is ub_arg
            ]
            if matching_ub_arg_idxs:
                matched_lb_arg_idxs.append(lb_idx)
                matching_lb_ub_args.append(lb_arg)

            # remove matched upper bound args to prevent
            # addition of redundant upper bound constraints
            for ub_idx in matching_ub_arg_idxs:
                ub_args.pop(ub_idx)

        # remove matched lower bound args to prevent
        # addition of redundant lower bound constraints
        for lb_idx in matched_lb_arg_idxs:
            lb_args.pop(lb_idx)

        # finally, perform the bound -> constraint reformulations
        bound_zip = zip(
            (lb_args, ub_args, matching_lb_ub_args),
            ("lower", "upper", "eq"),
        )
        for bound_args_list, desc in bound_zip:
            for arg in bound_args_list:
                working_model.add_component(
                    unique_component_name(
                        instance=working_model,
                        name=f"{var.local_name}_{desc}_bound_con",
                    ),
                    Constraint(expr=generate_constraint_expr(var, arg, desc)),
                )

        # adjust domain. remove variable bounds
        var.domain = Reals
        var.setlb(None)
        var.setub(None)


def get_time_from_solver(results):
    """
    Obtain solver time from a Pyomo `SolverResults` object.

    Returns
    -------
    : float
        Solver time. May be CPU time or elapsed time,
        depending on the solver. If no time attribute
        is found, then `float("nan")` is returned.

    NOTE
    ----
    This method attempts to access solver time through the
    attributes of `results.solver` in the following order
    of precedence:

    1) Attribute with name ``pyros.util.TIC_TOC_SOLVE_TIME_ATTR``.
       This attribute is an estimate of the elapsed solve time
       obtained using the Pyomo `TicTocTimer` at the point the
       solver from which the results object is derived was invoked.
       Preferred over other time attributes, as other attributes
       may be in CPUs, and for purposes of evaluating overhead
       time, we require wall s.
    2) `'user_time'` if the results object was returned by a GAMS
       solver, `'time'` otherwise.
    """
    solver_name = getattr(results.solver, "name", None)

    # is this sufficient to confirm GAMS solver used?
    from_gams = solver_name is not None and str(solver_name).startswith("GAMS ")
    time_attr_name = "user_time" if from_gams else "time"
    for attr_name in [TIC_TOC_SOLVE_TIME_ATTR, time_attr_name]:
        solve_time = getattr(results.solver, attr_name, None)
        if solve_time is not None:
            break

    return float("nan") if solve_time is None else solve_time


def standardize_inequality_constraints(model):
    """
    Recast all model inequality constraints of the form `a <= g(v)` (`<= b`)
    to the 'standard' form `a - g(v) <= 0` (and `g(v) - b <= 0`),
    in which `v` denotes all model variables and `a` and `b` are
    contingent on model parameters.

    Parameters
    ----------
    model : ConcreteModel
        The model to search for constraints. This will descend into all
        active Blocks and sub-Blocks as well.

    Note
    ----
    If `a` and `b` are identical and the constraint is not classified as an
    equality (i.e. the `equality` attribute of the constraint object
    is `False`), then the constraint is recast to the equality `g(v) == a`.
    """
    # Note: because we will be adding / modifying the number of
    # constraints, we want to resolve the generator to a list before
    # starting.
    cons = list(
        model.component_data_objects(Constraint, descend_into=True, active=True)
    )
    for con in cons:
        if not con.equality:
            has_lb = con.lower is not None
            has_ub = con.upper is not None

            if has_lb and has_ub:
                if con.lower is con.upper:
                    # recast as equality Constraint
                    con.set_value(con.lower == con.body)
                else:
                    # range inequality; split into two Constraints.
                    uniq_name = unique_component_name(model, con.name + '_lb')
                    model.add_component(
                        uniq_name, Constraint(expr=con.lower - con.body <= 0)
                    )
                    con.set_value(con.body - con.upper <= 0)
            elif has_lb:
                # not in standard form; recast.
                con.set_value(con.lower - con.body <= 0)
            elif has_ub:
                # move upper bound to body.
                con.set_value(con.body - con.upper <= 0)
            else:
                # unbounded constraint: deactivate
                con.deactivate()


def standardize_equality_constraints(model):
    """
    Standardize equality constraints of model.
    That is, constraint of form ``g(v) == a`` is cast to
    ``g(v) - a == 0``.
    """
    for con in model.component_data_objects(Constraint, active=True):
        if con.equality:
            con.set_value((0, con.body - con.lower, 0))


def reformulate_objective(blk, obj):
    """
    Epigraph formulation of minimization objective.

    NOTE
    ----
    Ensure `blk` does not contain attributes
    called 'epigraph_var' and 'epigraph_con'.
    """
    blk.epigraph_var = Var(
        initialize=value(obj, exception=False),
    )
    blk.epigraph_con = Constraint(expr=(obj.expr - blk.epigraph_var <= 0))
    obj.deactivate()


def identify_performance_constraints(working_model, config):
    """
    Identify performance constraints of working model.
    """
    # initialize performance constraints list
    working_model.util.performance_constraints = perf_cons = []
    working_model.util.first_stage_ineq_cons = first_stage_cons = []

    uncertain_params_set = ComponentSet(working_model.util.uncertain_params)
    state_vars_set = ComponentSet(working_model.util.state_vars)

    # efficiency: second-stage variables are effectively first-stage
    # under static DR (decision rule order == 0)
    if config.decision_rule_order == 0:
        effective_second_stage_vars_set = ComponentSet()
    else:
        effective_second_stage_vars_set = ComponentSet(
            working_model.util.second_stage_variables
        )

    # objective focus needed for checking epigraph constraint
    nominal_focus = config.objective_focus == ObjectiveType.nominal

    # now identify the constraints
    ineq_cons = (
        con for con
        in working_model.component_data_objects(Constraint, active=True)
        if not con.equality
    )
    for con in ineq_cons:
        vars_in_con = ComponentSet(identify_variables(con.body))
        uncertain_params_in_con = uncertain_params_set & ComponentSet(
            identify_mutable_parameters(con.body)
        )
        second_stage_vars_in_con = (
            effective_second_stage_vars_set & vars_in_con
        )
        state_vars_in_con = state_vars_set & vars_in_con
        is_epigraph_con = con is working_model.util.epigraph_con

        # under nominal focus, epigraph constraint is never
        # taken to be a performance constraint
        has_second_stage_components = bool(
            uncertain_params_in_con
            | second_stage_vars_in_con
            | state_vars_in_con
        )
        allow_based_on_obj_focus = not is_epigraph_con or not nominal_focus
        is_perf_con = (
            has_second_stage_components & allow_based_on_obj_focus
        )
        if is_perf_con:
            perf_cons.append(con)
        else:
            first_stage_cons.append(con)


def generate_all_decision_rule_vars(working_blk):
    """
    Generate sequence of all decision rule variables.
    """
    for indexed_var in working_blk.util.decision_rule_vars:
        yield from indexed_var.values()


def generate_all_decision_rule_eqns(working_blk):
    """
    Generate sequence of all decision rule equations.
    """
    for indexed_con in working_blk.util.decision_rule_eqns:
        yield from indexed_con.values()


def get_dr_expression(working_blk, second_stage_var):
    """
    Get DR expression corresponding given second-stage variable.
    """
    dr_con = working_blk.util.second_stage_var_to_dr_eq_map[second_stage_var]
    return sum(dr_con.body.args[:-1])


def add_bounds_for_uncertain_parameters(model, config):
    '''
    This function solves a set of optimization problems to determine bounds on the uncertain parameters
    given the uncertainty set description. These bounds will be added as additional constraints to the uncertainty_set_constr
    constraint. Should only be called once set_as_constraint() has been called on the separation_model object.
    :param separation_model: the model on which to add the bounds
    :param config: solver config
    :return:
    '''
    # === Determine bounds on all uncertain params
    uncertain_param_bounds = []
    bounding_model = ConcreteModel()
    bounding_model.util = Block()
    bounding_model.util.uncertain_param_vars = IndexedVar(
        model.util.uncertain_param_vars.index_set()
    )
    for tup in model.util.uncertain_param_vars.items():
        bounding_model.util.uncertain_param_vars[tup[0]].set_value(
            tup[1].value, skip_validation=True
        )

    bounding_model.add_component(
        "uncertainty_set_constraint",
        config.uncertainty_set.set_as_constraint(
            uncertain_params=bounding_model.util.uncertain_param_vars,
            model=bounding_model,
            config=config,
        ),
    )

    for idx, param in enumerate(
        list(bounding_model.util.uncertain_param_vars.values())
    ):
        bounding_model.add_component(
            "lb_obj_" + str(idx), Objective(expr=param, sense=minimize)
        )
        bounding_model.add_component(
            "ub_obj_" + str(idx), Objective(expr=param, sense=maximize)
        )

    for o in bounding_model.component_data_objects(Objective):
        o.deactivate()

    for i in range(len(bounding_model.util.uncertain_param_vars)):
        bounds = []
        for limit in ("lb", "ub"):
            getattr(bounding_model, limit + "_obj_" + str(i)).activate()
            res = config.global_solver.solve(bounding_model, tee=False)
            bounds.append(bounding_model.util.uncertain_param_vars[i].value)
            getattr(bounding_model, limit + "_obj_" + str(i)).deactivate()
        uncertain_param_bounds.append(bounds)

    # === Add bounds as constraints to uncertainty_set_constraint ConstraintList
    for idx, bound in enumerate(uncertain_param_bounds):
        model.util.uncertain_param_vars[idx].setlb(bound[0])
        model.util.uncertain_param_vars[idx].setub(bound[1])

    return


def get_vars_from_component(block, ctype):
    """Determine all variables used in active components within a block.

    Parameters
    ----------
    block: Block
        The block to search for components.  This is a recursive
        generator and will descend into any active sub-Blocks as well.
    ctype:  class
        The component type (typically either :py:class:`Constraint` or
        :py:class:`Objective` to search for).

    """

    return get_vars_from_components(block, ctype, active=True, descend_into=True)


def replace_uncertain_bounds_with_constraints(model, uncertain_params):
    """
    For variables of which the bounds are dependent on the parameters
    in the list `uncertain_params`, remove the bounds and add
    explicit variable bound inequality constraints.

    :param model: Model in which to make the bounds/constraint replacements
    :type model: class:`pyomo.core.base.PyomoModel.ConcreteModel`
    :param uncertain_params: List of uncertain model parameters
    :type uncertain_params: list
    """
    uncertain_param_set = ComponentSet(uncertain_params)

    # component for explicit inequality constraints
    uncertain_var_bound_constrs = ConstraintList()
    model.add_component(
        unique_component_name(model, 'uncertain_var_bound_cons'),
        uncertain_var_bound_constrs,
    )

    # get all variables in active objective and constraint expression(s)
    vars_in_cons = ComponentSet(get_vars_from_component(model, Constraint))
    vars_in_obj = ComponentSet(get_vars_from_component(model, Objective))

    for v in vars_in_cons | vars_in_obj:
        # get mutable parameters in variable bounds expressions
        ub = v.upper
        mutable_params_ub = ComponentSet(identify_mutable_parameters(ub))
        lb = v.lower
        mutable_params_lb = ComponentSet(identify_mutable_parameters(lb))

        # add explicit inequality constraint(s), remove variable bound(s)
        if mutable_params_ub & uncertain_param_set:
            if type(ub) is NPV_MinExpression:
                upper_bounds = ub.args
            else:
                upper_bounds = (ub,)
            for u_bnd in upper_bounds:
                uncertain_var_bound_constrs.add(v - u_bnd <= 0)
            v.setub(None)
        if mutable_params_lb & uncertain_param_set:
            if type(ub) is NPV_MaxExpression:
                lower_bounds = lb.args
            else:
                lower_bounds = (lb,)
            for l_bnd in lower_bounds:
                uncertain_var_bound_constrs.add(l_bnd - v <= 0)
            v.setlb(None)


def substitute_ssv_in_dr_constraints(model, constraint):
    '''
    Generate the standard_repn for the dr constraints. Generate new expression with replace_expression to ignore
    the ssv component.
    Then, replace_expression with substitution_map between ssv and the new expression.
    Deactivate or del_component the original dr equation.
    Then, return modified model and do coefficient matching as normal.
    :param model: the working_model
    :param constraint: an equality constraint from the working model identified to be of the form h(x,z,q) = 0.
    :return:
    '''
    dr_eqns = model.util.decision_rule_eqns
    fsv = ComponentSet(model.util.first_stage_variables)
    if not hasattr(model, "dr_substituted_constraints"):
        model.dr_substituted_constraints = ConstraintList()

    substitution_map = {}
    for eqn in dr_eqns:
        repn = generate_standard_repn(eqn.body, compute_values=False)
        new_expression = 0
        map_linear_coeff_to_var = [
            x
            for x in zip(repn.linear_coefs, repn.linear_vars)
            if x[1] in ComponentSet(fsv)
        ]
        map_quad_coeff_to_var = [
            x
            for x in zip(repn.quadratic_coefs, repn.quadratic_vars)
            if x[1] in ComponentSet(fsv)
        ]
        if repn.linear_coefs:
            for coeff, var in map_linear_coeff_to_var:
                new_expression += coeff * var
        if repn.quadratic_coefs:
            for coeff, var in map_quad_coeff_to_var:
                new_expression += coeff * var[0] * var[1]  # var here is a 2-tuple

        substitution_map[id(repn.linear_vars[-1])] = new_expression

    model.dr_substituted_constraints.add(
        replace_expressions(expr=constraint.lower, substitution_map=substitution_map)
        == replace_expressions(expr=constraint.body, substitution_map=substitution_map)
    )

    # === Delete the original constraint
    model.del_component(constraint.name)

    return model.dr_substituted_constraints[
        max(model.dr_substituted_constraints.keys())
    ]


def is_certain_parameter(uncertain_param_index, config):
    '''
    If an uncertain parameter's inferred LB and UB are within a relative tolerance,
    then the parameter is considered certain.
    :param uncertain_param_index: index of the parameter in the config.uncertain_params list
    :param config: solver config
    :return: True if param is effectively "certain," else return False
    '''
    if config.uncertainty_set.parameter_bounds:
        param_bounds = config.uncertainty_set.parameter_bounds[uncertain_param_index]
        return math.isclose(
            a=param_bounds[0],
            b=param_bounds[1],
            rel_tol=PARAM_IS_CERTAIN_REL_TOL,
            abs_tol=PARAM_IS_CERTAIN_ABS_TOL,
        )
    else:
        return False  # cannot be determined without bounds


def coefficient_matching(working_model, config):
    """
    Perform coefficient matching.

    Parameters
    ----------
    model_data : ModelData
        PyROS model data object.

    Returns
    -------
    robust_infeasible : bool
        True if model found to be robust infeasible,
        False otherwise.
    """
    # initialize coefficient matching constraints
    working_model.util.coefficient_matching_constraints = ConstraintList()

    # cast component lists to sets
    first_stage_var_set = ComponentSet(working_model.util.first_stage_variables)
    second_stage_var_set = ComponentSet(working_model.util.second_stage_variables)
    state_var_set = ComponentSet(working_model.util.state_vars)
    decision_rule_var_set = ComponentSet(
        generate_all_decision_rule_vars(working_model)
    )
    all_vars_set = (
        first_stage_var_set
        | second_stage_var_set
        | state_var_set
        | decision_rule_var_set
    )
    uncertain_params_set = ComponentSet(working_model.util.uncertain_params)

    # map second-stage variables to DR expressions
    ssvar_to_dr_expr_map = ComponentMap(
        (ss_var, get_dr_expression(working_model, ss_var))
        for ss_var in second_stage_var_set
    )
    ssvar_id_to_dr_expr_map = {
        id(ss_var): expr
        for ss_var, expr in ssvar_to_dr_expr_map.items()
    }

    # goal: examine constraint expressions in terms of the
    #       uncertain params.
    # for compatibility with repn.standard_repn,
    # we map all variables to temporarily defined params
    # and all uncertain params to temporarily defined vars
    temp_block = Block()
    working_model.add_component(
        unique_component_name(instance=working_model, name="temp_block"),
        temp_block,
    )

    # map vars to temporarily defined params
    temp_var_params = temp_block.temp_var_params = Param(
        range(len(all_vars_set)), initialize=1, mutable=True,
    )
    model_var_to_temp_param_map = ComponentMap(
        (var, var_param)
        for var, var_param in zip(all_vars_set, temp_var_params.values())
    )
    model_var_id_to_temp_param_map = {
        id(var): param for var, param in model_var_to_temp_param_map.items()
    }
    inverse_model_var_id_to_temp_param_map = {
        id(param): var for var, param in model_var_to_temp_param_map.items()
    }

    # map params to temporarily defined vars
    temp_param_vars = working_model.temp_param_vars = Var(
        range(len(uncertain_params_set)), initialize=1
    )
    uncertain_param_to_temp_var_map = ComponentMap(
        (param, param_var)
        for param, param_var
        in zip(uncertain_params_set, temp_param_vars.values())
    )
    uncertain_param_id_to_temp_var_map = {
        id(param): var for param, var in uncertain_param_to_temp_var_map.items()
    }
    inverse_uncertain_param_id_to_temp_var_map = {
        id(param): var for param, var in uncertain_param_to_temp_var_map.items()
    }

    robust_infeasible = False
    dr_cons_set = ComponentSet(working_model.util.decision_rule_eqns)
    for con in working_model.component_data_objects(Constraint, active=True):
        if con in dr_cons_set or not con.equality:
            # coefficient matching only applies to non-DR equality
            # constraints.
            continue

        vars_in_con = ComponentSet(identify_variables(con.expr))
        second_stage_vars_in_con = vars_in_con & second_stage_var_set
        state_vars_in_con = vars_in_con & state_var_set

        # check con.expr instead of con.body,
        # as con.upper (con.lower) may also contain uncertain params
        uncertain_params_in_con = ComponentSet(
            identify_mutable_parameters(con.expr)
        ) & uncertain_params_set

        coefficient_matching_applicable = (
            not state_vars_in_con
            and (uncertain_params_in_con or second_stage_vars_in_con)
        )
        if coefficient_matching_applicable:
            # this constraint will be reformulated to
            # a (set of) coefficient matching constraint(s)
            con.deactivate()

            # substitute decision rule expressions for second-stage variables
            # now we have expression of form h(x, d, q) == 0
            con_expr_after_dr_substitution = replace_expressions(
                expr=con.body - con.upper,
                substitution_map=ssvar_id_to_dr_expr_map,
            )

            # substitute temporarily defined params for vars
            con_expr_after_vars_out = replace_expressions(
                expr=con_expr_after_dr_substitution,
                substitution_map=model_var_id_to_temp_param_map,
            )

            # substitute temporarily defined vars for uncertain params
            con_expr_after_all_substitutions = replace_expressions(
                expr=con_expr_after_vars_out,
                substitution_map=uncertain_param_id_to_temp_var_map,
            )

            # at this point, the only var objects in the
            # expression should be the temporarily defined variables
            # which have been substituted for the uncertain params.
            # now we can invoke standard_repn to view the expression
            # in terms of the uncertain params
            # (or, rather, temporary vars to which the params are mapped)
            expr_repn = generate_standard_repn(
                expr=con_expr_after_all_substitutions,
                compute_values=False,
            )
            if expr_repn.nonlinear_expr is not None:
                # expression has nonlinear portion which cannot be
                # resolved to polynomial in uncertain params.
                # note: polynomials of degree 3 or higher considered
                # nonlinear, due to limitations of standard_repn.
                # cannot determine whether this expression
                # can be resolved to coefficient matching constraints,
                # or the model is robust infeasible.
                config.progress_logger.error(
                    f"Equality constraint {con.name!r} cannot be guaranteed "
                    "to be robustly feasible, given the current partitioning "
                    "among first-stage, second-stage, and state variables. "
                    "Consider editing this constraint to reference some "
                    "second-stage and/or state variable(s)."
                )
                raise ValueError(
                    "Coefficient matching unsuccessful. See the solver logs."
                )

            polynomial_repn_coeffs = (
                [expr_repn.constant]
                + list(expr_repn.linear_coefs)
                + list(expr_repn.quadratic_coefs)
            )
            for coef_expr in polynomial_repn_coeffs:
                # invert the substitutions of temporarily declared
                # components. now this coefficient is in terms of
                # on first-stage vars and DR vars
                coef_expr_with_orig_vars = replace_expressions(
                    expr=coef_expr,
                    substitution_map=inverse_model_var_id_to_temp_param_map,
                )
                coef_expr_with_orig_vars_and_params = replace_expressions(
                    expr=coef_expr_with_orig_vars,
                    substitution_map=inverse_uncertain_param_id_to_temp_var_map,
                )

                # simplify the expression
                simplified_coef_expr = generate_standard_repn(
                    expr=coef_expr_with_orig_vars_and_params,
                    compute_values=True,
                ).to_expression()

                if isinstance(simplified_coef_expr, tuple(native_types)):
                    # coefficient is a constant, and therefore
                    # must be close (within tolerances) to 0.
                    # otherwise, problem is robust infeasible.
                    robust_infeasible = not math.isclose(
                        a=simplified_coef_expr,
                        b=0,
                        rel_tol=COEFF_MATCH_REL_TOL,
                        abs_tol=COEFF_MATCH_ABS_TOL,
                    )
                else:
                    # coefficient is dependent on model first-stage
                    # and DR variables. add new matching constraint
                    working_model.util.coefficient_matching_constraints.add(
                        simplified_coef_expr == 0
                    )

        if robust_infeasible:
            config.progress_logger.info(
                "PyROS has determined that the model is robust infeasible. "
                "One reason for this is that "
                f"the equality constraint {con.name!r} "
                "cannot be satisfied against all realizations of uncertainty, "
                "given the current partitioning between "
                "first-stage, second-stage, and state variables. "
                "Consider editing this constraint to reference some "
                "(additional) second-stage and/or state variable(s)."
            )
            break

    # done. remove temporarily added components
    working_model.del_component(temp_param_vars)
    working_model.del_component(temp_param_vars.index_set())
    working_model.del_component(temp_var_params)
    working_model.del_component(temp_block)

    return robust_infeasible


def selective_clone(block, first_stage_vars):
    """
    Clone block and all underlying attributes except for
    the first-stage variables.

    In lieu of this method, consider using
    ``block.clone(memo=...)``, which offers a similar
    functionality provided the first-stage variables
    are mapped in the `memo` argument.

    Parameters
    ----------
    block : _BlockData
        Block to be cloned.
    first_stage_variables : Iterable of _VarData
        Variables to be maintained in the clone.

    Returns
    -------
    new_block : _BlockData
        Cloned block.
    """
    memo = {'__block_scope__': {id(block): True, id(None): False}}
    for v in first_stage_vars:
        memo[id(v)] = v
    new_block = copy.deepcopy(block, memo)
    new_block._parent = None

    return new_block


def get_decision_rule_to_second_stage_var_map(working_blk, indexed_dr_vars=False):
    """
    Generate mapping of decision rule variables to second-stage
    variables.

    Parameters
    ----------
    working_blk : _BlockData
        Working model-like block.
    indexed_dr_vars : bool, optional
        True to use indexed DR variables, rather than the individual
        _VarData members of the indexed DR variables, as
        the mapping keys; False otherwise.

    Yields
    ------
    dr_var : _VarData or IndexedVar
        Decision rule variable. Whether this is an individual
        member of an indexed DR var or not depends on
        option `indexed_dr_vars`.
    ss_var : _VarData
        Corresponding second-stage variable.
    """
    for ss_var, indexed_var in working_blk.util.ss_var_to_dr_var_map.items():
        if indexed_dr_vars:
            yield indexed_var, ss_var
        else:
            for var in indexed_var.values():
                yield var, ss_var


def add_decision_rule_variables(model_data, config):
    """
    Add variables for polynomial decision rules to the working
    model.

    Parameters
    ----------
    model_data : ROSolveResults
        Model data.
    config : config_dict
        PyROS solver options.

    Note
    ----
    Decision rule variables are considered first-stage decision
    variables which do not get copied at each iteration.
    PyROS currently supports static (zeroth order),
    affine (first-order), and quadratic DR.
    """
    second_stage_variables = model_data.working_model.util.second_stage_variables
    decision_rule_vars = []

    # map second-stage variables to indexed DR variables.
    ss_var_to_dr_var_map = ComponentMap()

    # since DR expression is a general polynomial in the uncertain
    # parameters, the exact number of DR variables per second-stage
    # variable depends on DR order and uncertainty set dimension
    degree = config.decision_rule_order
    num_uncertain_params = len(model_data.working_model.util.uncertain_params)
    num_dr_vars = sp.special.comb(
        N=num_uncertain_params + degree, k=degree, exact=True, repetition=False
    )

    for idx, ss_var in enumerate(second_stage_variables):
        # declare DR coefficients for current second-stage variable
        indexed_dr_var = Var(
            range(num_dr_vars), initialize=0, bounds=(None, None), domain=Reals
        )
        model_data.working_model.add_component(
            f"decision_rule_var_{idx}", indexed_dr_var
        )

        # index 0 entry of the IndexedVar is the static
        # DR term. initialize to user-provided value of
        # the corresponding second-stage variable.
        # all other entries remain initialized to 0.
        indexed_dr_var[0].set_value(value(ss_var, exception=False))

        # update attributes
        decision_rule_vars.append(indexed_dr_var)
        ss_var_to_dr_var_map[ss_var] = indexed_dr_var

    model_data.working_model.util.decision_rule_vars = decision_rule_vars
    model_data.working_model.util.ss_var_to_dr_var_map = ss_var_to_dr_var_map


def add_decision_rule_constraints(model_data, config):
    """
    Add decision rule equality constraints to the working model.

    Parameters
    ----------
    model_data : ROSolveResults
        Model data.
    config : ConfigDict
        PyROS solver options.
    """

    second_stage_variables = model_data.working_model.util.second_stage_variables
    uncertain_params = model_data.working_model.util.uncertain_params
    decision_rule_eqns = []
    decision_rule_vars_list = model_data.working_model.util.decision_rule_vars
    degree = config.decision_rule_order

    # keeping track of degree of monomial in which each
    # DR coefficient participates will be useful for later
    dr_var_to_exponent_map = ComponentMap()
    second_stage_var_to_dr_eq_map = ComponentMap()

    # set up uncertain parameter combinations for
    # construction of the monomials of the DR expressions
    monomial_param_combos = []
    for power in range(degree + 1):
        power_combos = it.combinations_with_replacement(uncertain_params, power)
        monomial_param_combos.extend(power_combos)

    # now construct DR equations and declare them on the working model
    second_stage_dr_var_zip = zip(second_stage_variables, decision_rule_vars_list)
    for idx, (ss_var, indexed_dr_var) in enumerate(second_stage_dr_var_zip):
        # for each DR equation, the number of coefficients should match
        # the number of monomial terms exactly
        if len(monomial_param_combos) != len(indexed_dr_var.index_set()):
            raise ValueError(
                f"Mismatch between number of DR coefficient variables "
                f"and number of DR monomials for DR equation index {idx}, "
                f"corresponding to second-stage variable {ss_var.name!r}. "
                f"({len(indexed_dr_var.index_set())}!= {len(monomial_param_combos)})"
            )

        # construct the DR polynomial
        dr_expression = 0
        for dr_var, param_combo in zip(indexed_dr_var.values(), monomial_param_combos):
            dr_expression += dr_var * prod(param_combo)

            # map decision rule var to degree (exponent) of the
            # associated monomial with respect to the uncertain params
            dr_var_to_exponent_map[dr_var] = len(param_combo)

        # declare constraint on model
        dr_eqn = Constraint(expr=dr_expression - ss_var == 0)
        model_data.working_model.add_component(f"decision_rule_eqn_{idx}", dr_eqn)

        # append to list of DR equality constraints
        decision_rule_eqns.append(dr_eqn)

        second_stage_var_to_dr_eq_map[ss_var] = dr_eqn

    # finally, add attributes to util block
    model_data.working_model.util.decision_rule_eqns = decision_rule_eqns
    model_data.working_model.util.dr_var_to_exponent_map = dr_var_to_exponent_map
    model_data.working_model.util.second_stage_var_to_dr_eq_map = (
        second_stage_var_to_dr_eq_map
    )


def enforce_dr_degree(blk, config, degree):
    """
    Make decision rule polynomials of a given degree
    by fixing value of the appropriate subset of the decision
    rule coefficients to 0.

    Parameters
    ----------
    blk : ScalarBlock
        Working model, or master problem block.
    config : ConfigDict
        PyROS solver options.
    degree : int
        Degree of the DR polynomials that is to be enforced.
    """
    second_stage_vars = blk.util.second_stage_variables
    indexed_dr_vars = blk.util.decision_rule_vars
    dr_var_to_exponent_map = blk.util.dr_var_to_exponent_map

    for ss_var, indexed_dr_var in zip(second_stage_vars, indexed_dr_vars):
        for dr_var in indexed_dr_var.values():
            dr_var_degree = dr_var_to_exponent_map[dr_var]

            if dr_var_degree > degree:
                dr_var.fix(0)
            else:
                dr_var.unfix()


def identify_objective_functions(model, objective, blk_for_obj_exprs=None):
    """
    Identify the first and second-stage portions of an Objective
    expression, subject to user-provided variable partitioning and
    uncertain parameter choice. In doing so, the first and second-stage
    objective expressions are added to the model as `Expression`
    attributes.

    Parameters
    ----------
    model : ConcreteModel
        Model of interest.
    objective : Objective
        Objective to be resolved into first and second-stage parts.
    blk_for_obj_exprs : _BlockData, optional
        Block descended from model upon which to declare
        the identified objective expressions.
        If `None` is passed, then `model` is used.
    """
    expr_to_split = objective.expr

    has_args = hasattr(expr_to_split, "args")
    is_sum = isinstance(expr_to_split, SumExpression)

    # determine additive terms of the objective expression
    # additive terms are in accordance with user declaration
    if has_args and is_sum:
        obj_args = expr_to_split.args
    else:
        obj_args = [expr_to_split]

    # initialize first and second-stage cost expressions
    first_stage_cost_expr = 0
    second_stage_cost_expr = 0

    first_stage_var_set = ComponentSet(model.util.first_stage_variables)
    uncertain_param_set = ComponentSet(model.util.uncertain_params)

    for term in obj_args:
        non_first_stage_vars_in_term = ComponentSet(
            v for v in identify_variables(term) if v not in first_stage_var_set
        )
        uncertain_params_in_term = ComponentSet(
            param
            for param in identify_mutable_parameters(term)
            if param in uncertain_param_set
        )

        if non_first_stage_vars_in_term or uncertain_params_in_term:
            second_stage_cost_expr += term
        else:
            first_stage_cost_expr += term

    if blk_for_obj_exprs is None:
        blk_for_obj_exprs = model

    blk_for_obj_exprs.full_objective = Expression(expr=expr_to_split)
    blk_for_obj_exprs.first_stage_objective = Expression(expr=first_stage_cost_expr)
    blk_for_obj_exprs.second_stage_objective = Expression(expr=second_stage_cost_expr)


def load_final_solution(model_data, master_soln, config):
    """
    Load final master solution to original deterministic model
    provided by the user.

    Parameters
    ----------
    model_data : ROSolveResults
        Main model data object.
    master_soln : MasterResult
        Master problem solution results.
    config : ConfigDict
        PyROS solver options.
    """
    original_model = model_data.original_model
    master_model = master_soln.master_model

    # determine scenario block from which to load variable values.
    # this depends on the objective focus, and may be changed later.
    if config.objective_focus == ObjectiveType.nominal:
        final_master_block = master_soln.nominal_block
    elif config.objective_focus == ObjectiveType.worst_case:
        final_idx = max(
            master_model.scenarios.keys(),
            key=lambda idx: value(
                master_model.scenarios[idx].util.full_objective
            ),
        )
        final_master_block = master_soln.master_model.scenarios[final_idx]

    # load master variable values to original model
    original_model_vars = (
        original_model.util.first_stage_variables
        + original_model.util.second_stage_variables
        + original_model.util.state_vars
    )
    final_master_vars = (
        final_master_block.util.first_stage_variables
        + final_master_block.util.second_stage_variables
        + final_master_block.util.state_vars
    )
    for orig_var, master_var in zip(original_model_vars, final_master_vars):
        orig_var.set_value(master_var.value, skip_validation=True)


def process_termination_condition_master_problem(config, results):
    '''
    :param config: pyros config
    :param results: solver results object
    :return: tuple (try_backups (True/False)
                  pyros_return_code (default NONE or robust_infeasible or subsolver_error))
    '''
    locally_acceptable = [tc.optimal, tc.locallyOptimal, tc.globallyOptimal]
    globally_acceptable = [tc.optimal, tc.globallyOptimal]
    robust_infeasible = [tc.infeasible]
    try_backups = [
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
    ]

    termination_condition = results.solver.termination_condition
    if config.solve_master_globally == False:
        if termination_condition in locally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, pyrosTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError(
                "This solver return termination condition (%s) "
                "is currently not supported by PyROS." % termination_condition
            )
    else:
        if termination_condition in globally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, pyrosTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError(
                "This solver return termination condition (%s) "
                "is currently not supported by PyROS." % termination_condition
            )


class _ModelStatistics(Bunch):
    """
    Component statistics for a working model passed to PyROS.

    Parameters
    ----------
    model_data : ModelData
        PyROS model data object.
    """
    def __init__(self, working_model):
        """
        Initialize self (see class docstring).
        """
        self._get_variable_statistics(working_model)
        self._get_uncertain_param_statistics(working_model)
        self._get_constraint_statistics(working_model)

    def _get_variable_statistics(self, working_model):
        """
        Get variable component statistics.
        """
        num_epigraph_vars = len([working_model.util.epigraph_var])
        num_first_stage_vars = len(working_model.util.first_stage_variables)
        num_second_stage_vars = len(working_model.util.second_stage_variables)
        num_state_vars = len(working_model.util.state_vars)
        num_dr_vars = len(list(
            generate_all_decision_rule_vars(working_model)
        ))
        num_vars = (
            num_epigraph_vars
            + num_first_stage_vars
            + num_second_stage_vars
            + num_state_vars
            + num_dr_vars
        )

        self.variables = Bunch(
            num_variables=num_vars,
            breakdown=Bunch(
                epigraph_variable=1,
                first_stage_variables=num_first_stage_vars,
                second_stage_variables=num_second_stage_vars,
                state_variables=num_state_vars,
                decision_rule_variables=num_dr_vars,
            )
        )

    def _get_uncertain_param_statistics(self, working_model):
        """
        Get uncertain parameter component statistics.
        """
        self.uncertain_params = Bunch(
            num_uncertain_parameters=len(working_model.util.uncertain_params),
        )

    def _get_constraint_statistics(self, working_model):
        """
        Get constraint statistics.
        """
        active_cons = ComponentSet(
            working_model.component_data_objects(Constraint, active=True)
        )

        # equality constraint stats
        equality_cons = ComponentSet(
            con for con in active_cons if con.equality
        )
        num_equality_constraints = len(equality_cons)
        num_coefficient_matching_cons = len(
            working_model.util.coefficient_matching_constraints
        )
        num_decision_rule_eqns = len(
            working_model.util.decision_rule_eqns
        )
        num_cons_from_original_model = len(
            working_model.util.original_model_equality_cons
        )
        equality_constraint_stats = Bunch(
            num_equality_constraints=num_equality_constraints,
            breakdown=Bunch(
                coefficient_matching_constraints=num_coefficient_matching_cons,
                decision_rule_equations=num_decision_rule_eqns,
                inferred_from_original_model=num_cons_from_original_model,
            )
        )

        # inequality constraint stats
        num_inequality_constraints = len(active_cons - equality_cons)
        num_perf_cons = len(
            working_model.util.performance_constraints
        )
        num_first_stage_ineq_cons = len(
            working_model.util.first_stage_ineq_cons
        )
        inequality_constraint_stats = Bunch(
            num_inequality_constraints=num_inequality_constraints,
            breakdown=Bunch(
                first_stage_inequalities=num_first_stage_ineq_cons,
                performance_constraints=num_perf_cons,
            ),
        )

        # now compile stats into attribute
        self.constraints = Bunch(
            num_constraints=len(active_cons),
            equality_constraints=equality_constraint_stats,
            inequality_constraints=inequality_constraint_stats,
        )

    def log_statistics(self, logger, *logger_args, **logger_kwargs):
        """
        Log component statistics contained in self.
        """
        logger.log(msg="Model statistics:", *logger_args, **logger_kwargs)

        # variable stats
        logger.log(
            msg=f"  Number of variables : {self.variables.num_variables}",
            *logger_args,
            **logger_kwargs
        )
        for desc, num in self.variables.breakdown.items():
            full_desc = desc.replace("_", " ").replace(" stage", "-stage").capitalize()
            logger.log(msg=f"    {full_desc} : {num}", *logger_args, **logger_kwargs)

        # uncertain parameter stats
        logger.log(
            msg=(
                "  Number of uncertain parameters : "
                f"{self.uncertain_params.num_uncertain_parameters}"
            ),
            *logger_args,
            **logger_kwargs,
        )

        # constraint stats
        logger.log(
            msg=(
                "  Number of constraints : "
                f"{self.constraints.num_constraints}"
            ),
            *logger_args,
            **logger_kwargs,
        )
        for con_type in ["equality_constraints", "inequality_constraints"]:
            con_type_bunch = self.constraints[con_type]

            num_cons = con_type_bunch[f"num_{con_type}"]
            con_type_desc = (
                con_type.replace("_", ' ').capitalize()
                + " (incl. var bounds)"
            )
            logger.log(
                msg=f"    {con_type_desc} : {num_cons}",
                *logger_args,
                **logger_kwargs,
            )
            for con_sub_type, val in con_type_bunch.breakdown.items():
                con_sub_type_desc = con_sub_type.capitalize().replace("_", " ")
                logger.log(
                    msg=f"      {con_sub_type_desc} : {val}",
                    *logger_args,
                    **logger_kwargs,
                )


def evaluate_and_log_model_statistics(model_data, config):
    """
    Evaluate and log working model component statistics.

    Returns
    -------
    _ModelStatistics
        Model component statistics.
    """
    model_stats = _ModelStatistics(working_model=model_data.working_model)
    model_stats.log_statistics(
        logger=config.progress_logger,
        level=logging.INFO,
    )
    return model_stats


def preprocess_model_data(model_data, config):
    """
    Preprocess model and PyROS options.

    Parameters
    ----------
    model_data : ModelData
        Model data.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    robust_infeasible : bool
        True if RO found to be robust infeasible through
        coefficient matching, False otherwise.
    """
    config.progress_logger.info("Preprocessing...")
    model_data.timing.start_timer("main.preprocessing")

    model = model_data.original_model

    # 'utility' block added to facilitate retrieval of crucial
    #       components after cloning
    # NOTE: Case in which model already contains an attribute
    #       called 'util' is not accounted for here.
    #       This issue will be addressed later,
    #       after reorganization of other modeling objects.
    model.util = Block(concrete=True)

    model.util.first_stage_variables = config.first_stage_variables
    model.util.second_stage_variables = config.second_stage_variables
    model.util.state_vars = list(get_state_vars(
        blk=model,
        first_stage_variables=model.util.first_stage_variables,
        second_stage_variables=model.util.second_stage_variables,
    ))
    model.util.uncertain_params = config.uncertain_params

    # working model: basis for all other pyros subproblems
    model_data.working_model = working_model = model_data.original_model.clone()

    turn_variable_bounds_to_constraints(working_model)
    standardize_inequality_constraints(working_model)
    standardize_equality_constraints(working_model)

    # epigraph reformulation of active objective
    active_obj = next(
        model_data.working_model.component_data_objects(
            Objective, active=True, descend_into=True
        )
    )
    recast_to_min_obj(working_model, active_obj)
    reformulate_objective(working_model.util, active_obj)

    # get first-stage and second-stage objective expressions.
    # useful for logging and recording results.
    identify_objective_functions(
        working_model,
        active_obj,
        blk_for_obj_exprs=working_model.util,
    )

    working_model.util.original_model_equality_cons = [
        con
        for con in working_model.component_data_objects(
            Constraint, active=True
        )
        if con.equality
    ]

    # assemble list of performance constraints
    identify_performance_constraints(working_model, config)

    add_decision_rule_variables(model_data, config)
    add_decision_rule_constraints(model_data, config)

    robust_infeasible = coefficient_matching(working_model, config)

    model_data.timing.stop_timer("main.preprocessing")
    preprocessing_time = model_data.timing.get_total_time("main.preprocessing")
    config.progress_logger.info(
        f"Done preprocessing; required wall time of "
        f"{preprocessing_time:.3f}s."
    )

    return robust_infeasible


class IterationLogRecord:
    """
    PyROS solver iteration log record.

    Parameters
    ----------
    iteration : int or None, optional
        Iteration number.
    objective : int or None, optional
        Master problem objective value.
        Note: if the sense of the original model is maximization,
        then this is the negative of the objective value
        of the original model.
    first_stage_var_shift : float or None, optional
        Infinity norm of the difference between first-stage
        variable vectors for the current and previous iterations.
    second_stage_var_shift : float or None, optional
        Infinity norm of the difference between decision rule
        variable vectors for the current and previous iterations.
    dr_polishing_success : bool or None, optional
        True if DR polishing solved successfully, False otherwise.
    num_violated_cons : int or None, optional
        Number of performance constraints found to be violated
        during separation step.
    all_sep_problems_solved : int or None, optional
        True if all separation problems were solved successfully,
        False otherwise (such as if there was a time out, subsolver
        error, or only a subset of the problems were solved due to
        custom constraint prioritization).
    global_separation : bool, optional
        True if separation problems were solved with the subordinate
        global optimizer(s), False otherwise.
    max_violation : int or None
        Maximum scaled violation of any performance constraint
        found during separation step.
    elapsed_time : float, optional
        Total time elapsed up to the current iteration, in seconds.

    Attributes
    ----------
    iteration : int or None
        Iteration number.
    objective : int or None
        Master problem objective value.
        Note: if the sense of the original model is maximization,
        then this is the negative of the objective value
        of the original model.
    first_stage_var_shift : float or None
        Infinity norm of the relative difference between first-stage
        variable vectors for the current and previous iterations.
    second_stage_var_shift : float or None
        Infinity norm of the relative difference between second-stage
        variable vectors (evaluated subject to the nominal uncertain
        parameter realization) for the current and previous iterations.
    dr_var_shift : float or None
        Infinity norm of the relative difference between decision rule
        variable vectors for the current and previous iterations.
        NOTE: This value is not reported in log messages.
    dr_polishing_success : bool or None
        True if DR polishing was solved successfully, False otherwise.
    num_violated_cons : int or None
        Number of performance constraints found to be violated
        during separation step.
    all_sep_problems_solved : int or None
        True if all separation problems were solved successfully,
        False otherwise (such as if there was a time out, subsolver
        error, or only a subset of the problems were solved due to
        custom constraint prioritization).
    global_separation : bool
        True if separation problems were solved with the subordinate
        global optimizer(s), False otherwise.
    max_violation : int or None
        Maximum scaled violation of any performance constraint
        found during separation step.
    elapsed_time : float
        Total time elapsed up to the current iteration, in seconds.
    """

    _LINE_LENGTH = 78
    _ATTR_FORMAT_LENGTHS = {
        "iteration": 5,
        "objective": 13,
        "first_stage_var_shift": 13,
        "second_stage_var_shift": 13,
        "dr_var_shift": 13,
        "num_violated_cons": 8,
        "max_violation": 13,
        "elapsed_time": 13,
    }
    _ATTR_HEADER_NAMES = {
        "iteration": "Itn",
        "objective": "Objective",
        "first_stage_var_shift": "1-Stg Shift",
        "second_stage_var_shift": "2-Stg Shift",
        "dr_var_shift": "DR Shift",
        "num_violated_cons": "#CViol",
        "max_violation": "Max Viol",
        "elapsed_time": "Wall Time (s)",
    }

    def __init__(
        self,
        iteration,
        objective,
        first_stage_var_shift,
        second_stage_var_shift,
        dr_var_shift,
        dr_polishing_success,
        num_violated_cons,
        all_sep_problems_solved,
        global_separation,
        max_violation,
        elapsed_time,
    ):
        """Initialize self (see class docstring)."""
        self.iteration = iteration
        self.objective = objective
        self.first_stage_var_shift = first_stage_var_shift
        self.second_stage_var_shift = second_stage_var_shift
        self.dr_var_shift = dr_var_shift
        self.dr_polishing_success = dr_polishing_success
        self.num_violated_cons = num_violated_cons
        self.all_sep_problems_solved = all_sep_problems_solved
        self.global_separation = global_separation
        self.max_violation = max_violation
        self.elapsed_time = elapsed_time

    def get_log_str(self):
        """Get iteration log string."""
        attrs = [
            "iteration",
            "objective",
            "first_stage_var_shift",
            "second_stage_var_shift",
            # "dr_var_shift",
            "num_violated_cons",
            "max_violation",
            "elapsed_time",
        ]
        return "".join(self._format_record_attr(attr) for attr in attrs)

    def _format_record_attr(self, attr_name):
        """Format attribute record for logging."""
        attr_val = getattr(self, attr_name)
        if attr_val is None:
            fmt_str = f"<{self._ATTR_FORMAT_LENGTHS[attr_name]}s"
            return f"{'-':{fmt_str}}"
        else:
            attr_val_fstrs = {
                "iteration": "f'{attr_val:d}'",
                "objective": "f'{attr_val: .4e}'",
                "first_stage_var_shift": "f'{attr_val:.4e}'",
                "second_stage_var_shift": "f'{attr_val:.4e}'",
                "dr_var_shift": "f'{attr_val:.4e}'",
                "num_violated_cons": "f'{attr_val:d}'",
                "max_violation": "f'{attr_val:.4e}'",
                "elapsed_time": "f'{attr_val:.3f}'",
            }

            # qualifier for DR polishing and separation columns
            if attr_name in ["second_stage_var_shift", "dr_var_shift"]:
                qual = "*" if not self.dr_polishing_success else ""
            elif attr_name == "num_violated_cons":
                qual = "+" if not self.all_sep_problems_solved else ""
            elif attr_name == "max_violation":
                qual = "g" if self.global_separation else ""
            else:
                qual = ""

            attr_val_str = f"{eval(attr_val_fstrs[attr_name])}{qual}"

            return f"{attr_val_str:{f'<{self._ATTR_FORMAT_LENGTHS[attr_name]}'}}"

    def log(self, log_func, **log_func_kwargs):
        """Log self."""
        log_str = self.get_log_str()
        log_func(log_str, **log_func_kwargs)

    @staticmethod
    def get_log_header_str():
        """Get string for iteration log header."""
        fmt_lengths_dict = IterationLogRecord._ATTR_FORMAT_LENGTHS
        header_names_dict = IterationLogRecord._ATTR_HEADER_NAMES
        return "".join(
            f"{header_names_dict[attr]:<{fmt_lengths_dict[attr]}s}"
            for attr in fmt_lengths_dict
            if attr != "dr_var_shift"
        )

    @staticmethod
    def log_header(log_func, with_rules=True, **log_func_kwargs):
        """Log header."""
        if with_rules:
            IterationLogRecord.log_header_rule(log_func, **log_func_kwargs)
        log_func(IterationLogRecord.get_log_header_str(), **log_func_kwargs)
        if with_rules:
            IterationLogRecord.log_header_rule(log_func, **log_func_kwargs)

    @staticmethod
    def log_header_rule(log_func, fillchar="-", **log_func_kwargs):
        """Log header rule."""
        log_func(fillchar * IterationLogRecord._LINE_LENGTH, **log_func_kwargs)
