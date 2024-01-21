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
    _VarData,
    _ConstraintData,
    _ObjectiveData,
)
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.suffix import SuffixFinder
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
    Get state variables of a modeling block.

    A state variable is any unfixed Var which:

    - is not a first-stage variable or a second-stage variable
    - participates in an active Objective or Constraint expression
      declared on `blk` or any of its active sub-blocks.

    Parameters
    ----------
    blk : ScalarBlock
        Block of interest.
    first_stage_variables : Iterable of VarData
        First-stage variables.
    second_stage_variables : Iterable of VarData
        Second-stage variables.

    Yields
    ------
    _VarData
        State variable.
    """
    dof_var_set = (
        ComponentSet(first_stage_variables)
        | ComponentSet(second_stage_variables)
    )
    for var in get_vars_from_component(blk, (Objective, Constraint)):
        is_state_var = not var.fixed and var not in dof_var_set
        if is_state_var:
            yield var


class BoundType(Enum):
    """
    Indicator for whether a bound specification on a constraint
    is a lower bound, upper bound, or 'bound by equality'.
    """
    LOWER = "lower"
    UPPER = "upper"
    EQ = "eq"

    def generate_bound_constraint_expr(self, body, bound):
        """
        Generate bound constraint expression depending on
        bound type indicated by `self`.

        Parameters
        ----------
        body : _VarData or NumericExpression
            Body of the constraint.
        bound : NumericExpression or numeric type
            Bound for the body expression.
            If an expression, then should include mutable
            params only.

        Returns
        -------
        InequalityExpression
            If bound type indicated by self is `BoundType.LOWER`
            or `BoundType.UPPER`.
        EqualityExpression
            If bound type indicated by `self` is `BoundType.EQ`.

        Raises
        ------
        ValueError
            If bound type indicated by `self` is currently
            not supported.
        """
        if self == self.LOWER:
            return bound - body <= 0
        elif self == self.UPPER:
            return body - bound <= 0
        elif self == self.EQ:
            return body - bound == 0
        else:
            raise ValueError(f"Bound type {self!r} not supported.")


def _resolve_component_bounds(var_or_con, uncertain_params):
    """
    Determine effective lower, upper, and/or equality bounds
    on a Var or Constraint component data object.

    Parameters
    ----------
    var_or_con : _VarData or _ConstraintData
        Variable or constraint whose bounds are to be
        treated.
    uncertain_params : Iterable of _ParamData
        Mutable Param objects in the bounds expressions
        of `var_or_con` considered uncertain.

    Returns
    -------
    dict
        Maps each member of `BoundType` is to a list of
        bounds of the corresponding type.
    """
    uncertain_params_set = uncertain_params
    if not isinstance(uncertain_params, ComponentSet):
        uncertain_params_set = ComponentSet(uncertain_params)

    lb_expr = var_or_con.lower
    ub_expr = var_or_con.upper

    # lb (ub) can be a max (min) expression due to, say
    # Var domain specifications.
    # capture all the arguments of the max (min)
    # to address them separately
    lb_args = [lb_expr] if lb_expr is not None else []
    if isinstance(lb_expr, NPV_MaxExpression):
        lb_args = list(lb_expr.args)
    ub_args = [ub_expr] if ub_expr is not None else []
    if isinstance(ub_expr, NPV_MinExpression):
        ub_args = list(ub_expr.args)

    # we only care about dependence of bounds on the uncertain params,
    # so we evaluate all the other mutable params
    bound_type_to_args_map = {BoundType.LOWER: lb_args, BoundType.UPPER: ub_args}
    for bound_type, args_list in bound_type_to_args_map.items():
        for idx, arg in enumerate(args_list):
            certain_params_in_arg = (
                ComponentSet(identify_mutable_parameters(arg))
                - uncertain_params_set
            )

            # any potential for precision loss here?
            args_list[idx] = replace_expressions(
                expr=arg,
                substitution_map={
                    id(param): value(param) for param in certain_params_in_arg
                },
            )

    # in case upper bound and lower bound share identical args,
    # we consider that an equality bound instead of separate
    # lower/upper bounds
    final_lb_args = []
    final_eq_args = []
    matched_ub_args_idx_set = set()
    for lb_idx, lb_arg in enumerate(lb_args):
        matching_ub_arg_idxs = set(
            ub_idx for ub_idx, ub_arg in enumerate(ub_args)
            if lb_arg is ub_arg
        )
        if not matching_ub_arg_idxs:
            final_lb_args.append(lb_arg)
        else:
            final_eq_args.append(lb_arg)
        matched_ub_args_idx_set.update(matching_ub_arg_idxs)

    unmatched_ub_args_idx_set = set(range(len(ub_args))) - matched_ub_args_idx_set
    final_ub_args = [ub_args[idx] for idx in unmatched_ub_args_idx_set]

    return {
        BoundType.LOWER: final_lb_args,
        BoundType.UPPER: final_ub_args,
        BoundType.EQ: final_eq_args,
    }


def turn_bounds_to_constraints(model, uncertain_params, variables=None):
    """
    Cast model variable bound/domain specifications to inequality
    constraints.

    Parameters
    ----------
    model : ConcreteModel
        Model on which to act.
    uncertain_params : Iterable of _ParamData
        Mutable parameters considered uncertain.
    variables : None or list of _VarData, optional
        Variables for which bound specifications are to be
        turned to constraints. All variables are assumed
        to have continuous domains. If `None` is passed, then
        all unfixed variables participating in expressions of
        all the active Objective and Constraint component expressions
        (including components found in active sub-Blocks) are
        acted upon.

    Returns
    -------
    var_to_bound_con_map : ComponentMap
        Maps the variables to ComponentMap objects containing
        the bound constraints. Each inner component map
        matches a bound constraint to a BoundType instance
        indicating whether the constraint specifies a lower bound,
        upper bound, or bound by equality.

    Note
    ----
    This method accounts carefully for:

    - bounds expressions involving uncertain parameters
    - bounds implicitly added through custom Var domain
      specifications.

    All Vars are assumed to have continuous domains.
    """
    if variables is None:
        variables = (
            var for var in get_vars_from_component(
                block=model,
                ctype=(Objective, Constraint),
            )
            if not var.fixed
        )
    uncertain_params_set = ComponentSet(uncertain_params)
    var_to_bound_con_map = ComponentMap()
    for var in variables:
        var_to_bound_con_map[var] = std_con_to_bound_map = ComponentMap()
        resolved_bounds_map = _resolve_component_bounds(
            var,
            uncertain_params=uncertain_params_set,
        )
        for bound_type, bounds_list in resolved_bounds_map.items():
            for arg in bounds_list:
                bound_con = Constraint(
                    expr=bound_type.generate_bound_constraint_expr(
                        body=var,
                        bound=arg,
                    )
                )
                model.add_component(
                    name=unique_component_name(
                        instance=model,
                        name=f"{var.local_name}_{bound_type.value}_bound_con",
                    ),
                    val=bound_con,
                )
                std_con_to_bound_map[bound_con] = bound_type

        # finally, remove variable domain and bound specifications
        var.domain = Reals
        var.setlb(None)
        var.setub(None)

    return var_to_bound_con_map


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


def standardize_active_constraints(model, uncertain_params):
    """
    Recast all active model inequality constraints of the form
    `a <= g(v)` (`<= b`) to the 'standard' form `a - g(v) <= 0`
    (and `g(v) - b <= 0`), in which `v` denotes all model variables
    and `a` and `b` are contingent on model parameters.
    If `a` and `b` are identical objects, then the constraint
    is recast to `g(v) - a == 0`.

    Parameters
    ----------
    model : ConcreteModel
        The model to search for constraints. This will descend into all
        active Blocks and sub-Blocks as well.
    uncertain_params : Iterable of _ParamData
        Mutable parameters considered uncertain.

    Returns
    -------
    std_ineq_con_map : ComponentMap
        Each entry matches an inequality constraint of `model`
        to a ComponentMap. The inner map matches a standardized
        inequality constraint of `model` to a BoundType depending
        on whether the standardized constraint was derived from
        the lower bound, upper bound, or deduced to be an equality.
    """
    uncertain_params_set = ComponentSet(uncertain_params)

    # keeping track of separation priorities
    std_ineq_con_map = ComponentMap()

    # Note: because we will be adding / modifying the number of
    # constraints, we want to resolve the generator to a list before
    # starting.
    cons = list(
        model.component_data_objects(Constraint, descend_into=True, active=True)
    )
    for con in cons:
        std_ineq_con_map[con] = std_con_to_bound_map = ComponentMap()
        if not con.equality:
            resolved_bounds_map = _resolve_component_bounds(
                var_or_con=con,
                uncertain_params=uncertain_params_set,
            )
            btype_bound_pairs = [
                (btype, bound)
                for btype, bound_list in resolved_bounds_map.items()
                for bound in bound_list
            ]
            num_bounds = len(btype_bound_pairs)
            if num_bounds == 1:
                btype, bound = btype_bound_pairs[0]
                con.set_value(btype.generate_bound_constraint_expr(
                    body=con.body,
                    bound=bound,
                ))
                std_con_to_bound_map[con] = btype
            else:
                con.deactivate()
                for btype, bound in btype_bound_pairs:
                    bound_con = Constraint(
                        expr=btype.generate_bound_constraint_expr(
                            body=con.body,
                            bound=bound,
                        )
                    )
                    model.add_component(
                        name=unique_component_name(
                            instance=model,
                            name=f"{con.name}_{btype.value}_bound",
                        ),
                        val=bound_con,
                    )
                    std_con_to_bound_map[bound_con] = btype
        else:
            std_con_to_bound_map[con] = BoundType.EQ
            con.set_value(con.body - con.upper == 0)

    return std_ineq_con_map


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


def coefficient_matching(model_data, config):
    """
    Perform coefficient matching.

    Parameters
    ----------
    model_data : ROSolveResults
        PyROS model data object.

    Returns
    -------
    robust_infeasible : bool
        True if model found to be robust infeasible,
        False otherwise.
    """
    # initialize coefficient matching constraints
    working_model = model_data.working_model
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


def standardize_objective(model_data, config):
    """
    Standardize active objective of the working model and
    related components.

    This method involves:

    - recasting the active objective to a minimization sense,
      if necessary
    - performing an epigraph reformulation and setting the
      priority of the epigraph constraint to that of the
      active objective
    - identifying first-stage and second-stage additive terms
      of the objective expression.

    Parameters
    ----------
    model_data : ROSolveResults
        Main model data object.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    active_obj : _ObjectiveData
        Active objective of the working model.
    """
    working_model = model_data.working_model
    active_obj = next(
        model_data.working_model.component_data_objects(
            Objective, active=True, descend_into=True
        )
    )

    # epigraph reformulation
    recast_to_min_obj(working_model, active_obj)
    reformulate_objective(working_model.util, active_obj)
    identify_objective_functions(
        working_model,
        active_obj,
        blk_for_obj_exprs=working_model.util,
    )

    return active_obj


class _SeparationPriorityStandardizer:
    """
    Standardizer callable for separation problem priority
    specification.
    """
    DEFAULT_SEPARATION_PRIORITY = 0

    def _standardize_constraint_priority(self, component, priority, logger):
        """
        Standardize separation priority for constraint data object.

        Parameters
        ----------
        component : _ConstraintData
            Constraint for which priority is to be standardized.
        priority : None, int, or 2-tuple of None/int
            Priority specification.
        logger : logging.Logger
            Logger for warning/error messages.

        Returns
        -------
        2-tuple of int/None
            Priority specification.
            The first entry specifies the priority for the
            lower bound requirement.
            The second entry specifies the priority for the
            upper bound requirement.
        """
        if priority is None:
            priority = (None, None)

        if isinstance(priority, tuple):
            if len(priority) != 2:
                raise ValueError(
                    f"Separation priority for component {component.name!r} "
                    f"is not of length 2 (got tuple of length {len(priority)})"
                )
        else:
            lb_priority = priority if component.lower is not None else None
            ub_priority = priority if component.upper is not None else None
            priority = (lb_priority, ub_priority)

        final_priority_list = []
        priority_zip = zip(
            priority,
            (component.lower, component.upper),
            ("lower", "upper"),
        )
        for val, bnd, desc in priority_zip:
            if bnd is None:
                if val is not None:
                    logger.warning(
                        f"Component {component.name!r} has separation priority "
                        f"specified for its {desc} bound attribute, "
                        f"but the bound attribute is of value None. "
                        "Separation priority for this bound will be disregarded."
                    )
                final_priority_list.append(None)
            else:
                std_val = (
                    int(val) if val is not None
                    else self.DEFAULT_SEPARATION_PRIORITY
                )
                if val is not None and val != std_val:
                    raise ValueError(
                        "Could not cast separation priority for component "
                        f"{component.name!r} to 2-tuple of ints. "
                        f"(Got value {val}.)"
                    )
                final_priority_list.append(std_val)

        return tuple(final_priority_list)

    def _standardize_var_priority(self, component, priority, logger):
        """
        Standardize separation priority for variable data object.

        Parameters
        ----------
        component : _VarData
            Variable for which priority is to be standardized.
        priority : None, int, or 2-tuple of None/int
            Priority specification.
        logger : logging.Logger
            Logger for warning/error messages.

        Returns
        -------
        2-tuple of int/None
            Priority specification.
            The first entry specifies the priority for the
            lower bound requirement.
            The second entry specifies the priority for the
            upper bound requirement.
        """
        return self._standardize_constraint_priority(component, priority, logger)

    def _standardize_objective_priority(self, component, priority, logger):
        """
        Standardize separation priority for objective data object.

        Parameters
        ----------
        component : _ObjectiveData
            Objective for which priority is to be standardized.
        priority : int or None
             Priority specification.
        logger : logging.Logger
            Logger for warning/error messages.

        Returns
        -------
        int or None
            Separation priority.
        """
        if priority is None:
            return self.DEFAULT_SEPARATION_PRIORITY
        else:
            std_priority = int(priority)
            if std_priority != priority:
                raise ValueError(
                    "Separation priority of objective with name "
                    f"{component.name!r} is not castable to int "
                    f"(got priority value {priority})"
                )
            return int(priority)

    def __call__(self, component, priority, logger):
        """
        Standardize separation priority specification for
        a component data attribute.

        Parameters
        ----------
        component : _ConstraintData, _ObjectiveData, or _VarData
            Component for which priority is to be standardized.
        priority : int, None, or 2-tuple of int/None
             Priority specification.
        logger : logging.Logger
            Logger for warning/error messages.

        Returns
        -------
        int or None
            If `component` is of type `_ObjectiveData`.
        2-tuple of int/None
            If `component` is of type `_ConstraintData`
            or `_VarData`.

        Raises
        ------
        TypeError
            If component is not of appropriate type.
        """
        if isinstance(component, _VarData):
            return self._standardize_var_priority(
                component=component,
                priority=priority,
                logger=logger,
            )
        elif isinstance(component, _ConstraintData):
            return self._standardize_constraint_priority(
                component=component,
                priority=priority,
                logger=logger,
            )
        elif isinstance(component, _ObjectiveData):
            return self._standardize_objective_priority(
                component=component,
                priority=priority,
                logger=logger,
            )
        else:
            raise TypeError(
                f"Argument `{component=!r}` should be of type "
                "_VarData, _ConstraintData, or _ObjectiveData, but got type "
                f"{type(component).__name__!r}."
            )


def standardize_separation_priorities(model_data, config):
    """
    Standardize separation priority specifications for
    model components from which performance constraints
    may be inferred.

    This method involves parsing custom priority specifications
    from both the `separation_priority_order` attribute of
    the PyROS solver options contained in `config` and
    (if present) the `pyros_separation_priority` Suffix
    declared on the working model or any sub-blocks
    thereof.

    Notes on the order of precedence in the event there
    are overlaps:

    - Priorities specified through the Suffix take precedence over
      priorities specified through `separation_priority_order`.
    - Priorities specified through the Suffix(es) are resolved
      through the `SuffixFinder` interface. See documentation of
      `SuffixFinder.find()` for more information.
    - In the context of `separation_priority_order`, priorities
      specified for component data objects take precedence over
      priorities specified for their containers.

    Parameters
    ----------
    model_data : ROSolveResults
        Main model data object.
    config : ConfigDict
        PyROS solver options.
    """
    working_model = model_data.working_model
    std_priority_map = ComponentMap()
    priority_std_func = _SeparationPriorityStandardizer()

    valid_component_types = (
        _ConstraintData,
        _VarData,
        _ObjectiveData,
        Constraint,
        Var,
        Objective,
    )

    config_priority_order_list = [
        (key, working_model.find_component(key), priority)
        for key, priority in config.separation_priority_order.items()
    ]
    config_priority_comp_map = ComponentMap(
        (comp, priority) for _, comp, priority in config_priority_order_list
    )
    for key, component, priority in config_priority_order_list:
        if not isinstance(component, valid_component_types):
            raise ValueError(
                "Could not a retrieve model component/component data "
                f"attribute through key {key!r} of dict "
                "`separation_priority_order` "
                f"(`.find_component` method returned {component!r})."
            )

        if component.is_indexed():
            for member in component.values():
                # priorities for members of indexed components take
                # precedence over priorities of their containers
                if member in config_priority_comp_map:
                    continue
                else:
                    std_priority_map[member] = priority_std_func(
                        component=member,
                        priority=priority,
                        logger=config.progress_logger,
                    )
        else:
            std_priority_map[component] = priority_std_func(
                component=component,
                priority=priority,
                logger=config.progress_logger,
            )

    # priorities specified through Suffix take precedence
    # over those specified through config.separation_priority_order
    suffix_finder = SuffixFinder(
        name="pyros_separation_priority",
        default=priority_std_func.DEFAULT_SEPARATION_PRIORITY,
    )
    for comp_data in working_model.component_data_objects(
        ctype=(Var, Constraint, Objective),
        active=True,
    ):
        std_priority_map[comp_data] = priority_std_func(
            component=comp_data,
            priority=suffix_finder.find(comp_data),
            logger=config.progress_logger,
        )

    for suffix in suffix_finder.all_suffixes:
        suffix.deactivate()

    working_model.util.separation_priority_map = std_priority_map


def finalize_separation_priorities(model_data, config):
    """
    Finalize separation priority mapping.

    Once this routine is complete, the separation priority map
    at `model_data.working_model.util.working_model` should
    map all performance constraints to ints specifying their
    (upper bound) separation priorities.

    Parameters
    ----------
    model_data : ROSolveResults
        Main model data object.
    config : ConfigDict
        PyROS solver options.
    """
    working_model = model_data.working_model
    var_to_bound_con_map = working_model.util.var_to_bound_con_map
    sep_priority_map = working_model.util.separation_priority_map
    sep_priority_std = _SeparationPriorityStandardizer()

    # map bound inequality constraints to priorities of
    # corresponding variable bounds
    for var, con_to_type_map in var_to_bound_con_map.items():
        var_sep_priority = sep_priority_map.get(var, None)
        var_sep_priority_dict = {
            BoundType.LOWER: var_sep_priority[0],
            BoundType.UPPER: var_sep_priority[1],
            BoundType.EQ: None,
        }
        for con, bound_type in con_to_type_map.items():
            sep_priority_map[con] = sep_priority_std(
                component=con,
                priority=var_sep_priority_dict[bound_type],
                logger=config.progress_logger,
            )
        del sep_priority_map[var]

    # map standardized inequality constraints to priorities of
    # corresponding original inequality constraints
    con_to_std_con_map = working_model.util.con_to_std_con_map
    for con, std_con_bound_type_map in con_to_std_con_map.items():
        sep_priority = sep_priority_map.get(con, None)
        con_sep_priority_dict = {
            BoundType.LOWER: sep_priority[0],
            BoundType.UPPER: sep_priority[1],
            BoundType.EQ: None,
        }
        for std_con, bound_type in std_con_bound_type_map.items():
            sep_priority_map[con] = sep_priority_std(
                component=con,
                priority=con_sep_priority_dict[bound_type],
                logger=config.progress_logger,
            )

    # map epigraph constraint to priority of active objective
    epigraph_con = working_model.util.epigraph_con
    sep_priority_map[epigraph_con] = sep_priority_std(
        component=working_model.util.epigraph_con,
        priority=sep_priority_map.get(model_data.active_obj, None),
        logger=config.progress_logger,
    )
    del sep_priority_map[model_data.active_obj]

    # simplify separation priority mapping: we only care
    # about performance constraints
    priority_map_items = list(sep_priority_map.items())
    perf_cons_set = ComponentSet(working_model.util.performance_constraints)
    for comp, priority in priority_map_items:
        if comp not in perf_cons_set:
            del sep_priority_map[comp]
        else:
            sep_priority_map[comp] = int(sep_priority_map[comp][1])


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

    standardize_separation_priorities(model_data, config)

    # standardization routines also carefully manage
    # separation priority specifications.
    # once done, priority suffix should be a mapping from
    # inequality constraints to ints
    working_model.util.var_to_bound_con_map = turn_bounds_to_constraints(
        model=working_model,
        uncertain_params=working_model.util.uncertain_params,
        variables=(
            working_model.util.first_stage_variables
            + working_model.util.second_stage_variables
            + working_model.util.state_vars
        ),
    )
    working_model.util.con_to_std_con_map = standardize_active_constraints(
        model=working_model,
        uncertain_params=working_model.util.uncertain_params,
    )
    working_model.util.original_model_equality_cons = list(
        con for con in
        working_model.component_data_objects(Constraint, active=True)
        if con.equality
    )
    model_data.active_obj = standardize_objective(model_data, config)
    identify_performance_constraints(working_model, config)

    add_decision_rule_variables(model_data, config)
    add_decision_rule_constraints(model_data, config)

    finalize_separation_priorities(model_data, config)

    robust_infeasible = coefficient_matching(model_data, config)

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
