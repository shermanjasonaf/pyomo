"""
Interfaces for resolving and validating arguments to the
PyROS solver.
"""


from collections.abc import Iterable
import functools
import logging
import os
from textwrap import indent, dedent, wrap

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    In,
    NonNegativeFloat,
    Path,
    InEnum,
)
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import ApplicationError
from pyomo.core.base import value, ConcreteModel, Constraint, Objective
from pyomo.core.base.param import Param, _ParamData
from pyomo.core.base.var import Var, _VarData
from pyomo.core.expr.visitor import identify_variables
from pyomo.opt import SolverFactory

from pyomo.contrib.pyros.uncertainty_sets import uncertainty_sets
from pyomo.contrib.pyros.util import (
    ObjectiveType,
    setup_pyros_logger,
    get_vars_from_component,
    get_state_vars,
)


default_pyros_solver_logger = setup_pyros_logger()


class LoggerType:
    """
    Domain validator for objects castable to logging.Logger.

    Parameters
    ----------
    str_or_logger : str or logging.Logger
        String or logger object to normalize.

    Returns
    -------
    logging.Logger
        If `str_or_logger` is of type `logging.Logger`,then
        `str_or_logger` is returned.
        Otherwise, ``logging.getLogger(str_or_logger)``
        is returned. In the event `str_or_logger` is
        the name of the default PyROS logger, the logger level
        is set to `logging.INFO`, and a `PreformattedLogger`
        instance is returned in lieu of a standard `Logger`
        instance.
    """
    def __call__(self, str_or_logger):
        if isinstance(str_or_logger, logging.Logger):
            return logging.getLogger(str_or_logger.name)
        else:
            return logging.getLogger(str_or_logger)

    def domain_name(self):
        return "str or logging.Logger"


class PositiveIntOrMinusOne:
    """
    Domain validator for objects castable to a
    strictly positive int or -1.
    """
    def __call__(self, obj):
        ans = int(obj)
        if ans != float(obj) or (ans <= 0 and ans != -1):
            raise ValueError("Expected positive int, but received %s" % (obj,))
        return ans

    def domain_name(self):
        return "positive int or -1"


class NotSolverResolvable(Exception):
    """
    Exception type for failure to cast an object to a Pyomo solver.
    """


class SolverResolvable(object):
    """
    Callable for casting an object (such as a str)
    to a Pyomo solver.

    Parameters
    ----------
    require_available : bool, optional
        True if `available()` method of a standardized solver
        object obtained through `self` must return `True`,
        False otherwise.
    solver_desc : str, optional
        Descriptor for the solver obtained through `self`,
        such as 'local solver'
        or 'global solver'. This argument is used
        for constructing error/exception messages.

    Attributes
    ----------
    require_available
    solver_desc
    """

    def __init__(self, require_available=True, solver_desc="solver"):
        """Initialize self (see class docstring).

        """
        self.require_available = require_available
        self.solver_desc = solver_desc

    @staticmethod
    def is_solver_type(obj):
        """
        Return True if object is considered a Pyomo solver,
        False otherwise.

        An object is considered a Pyomo solver provided that
        it has callable attributes named 'solve' and
        'available'.
        """
        return (
            callable(getattr(obj, "solve", None))
            and callable(getattr(obj, "available", None))
        )

    def __call__(self, obj, require_available=None, solver_desc=None):
        """
        Cast object to a Pyomo solver.

        If `obj` is a string, then ``SolverFactory(obj.lower())``
        is returned. If `obj` is a Pyomo solver type, then
        `obj` is returned.

        Parameters
        ----------
        obj : object
            Object to be cast to Pyomo solver type.
        require_available : bool or None, optional
            True if `available()` method of the resolved solver
            object must return True, False otherwise.
            If `None` is passed, then ``self.require_available``
            is used.
        solver_desc : str or None, optional
            Brief description of the solver, such as 'local solver'
            or 'backup global solver'. This argument is used
            for constructing error/exception messages.
            If `None` is passed, then ``self.solver_desc``
            is used.

        Returns
        -------
        Solver
            Pyomo solver.

        Raises
        ------
        NotSolverResolvable
            If `obj` cannot be cast to a Pyomo solver because
            it is neither a str nor a Pyomo solver type.
        ApplicationError
            In event that solver is not available, the
            method `available(exception_flag=True)` of the
            solver to which `obj` is cast should raise an
            exception of this type. The present method
            will also emit a more detailed error message
            through the default PyROS logger.
        """
        # resort to defaults if necessary
        if require_available is None:
            require_available = self.require_available
        if solver_desc is None:
            solver_desc = self.solver_desc

        # perform casting
        if isinstance(obj, str):
            solver = SolverFactory(obj.lower())
        elif self.is_solver_type(obj):
            solver = obj
        else:
            raise NotSolverResolvable(
                f"Cannot cast object `{obj!r}` to a Pyomo optimizer for use as a "
                f"{solver_desc}, as the object is neither a str nor a "
                f"Pyomo Solver type (got type {type(obj).__name__})."
            )

        # availability check, if so desired
        if require_available:
            try:
                solver.available(exception_flag=True)
            except ApplicationError:
                default_pyros_solver_logger.exception(
                    f"Output of `available()` method for {solver_desc} "
                    f"with repr {solver!r} resolved from object {obj} "
                    "is not `True`. "
                    "Check solver and any required dependencies "
                    "have been set up properly."
                )
                raise

        return solver

    def domain_name(self):
        """Description of domain encompassed by self."""
        return "str or Solver"


class SolverIterable(object):
    """
    Callable for casting an iterable (such as a list of strs)
    to a list of Pyomo solvers.

    Parameters
    ----------
    require_available : bool, optional
        True if `available()` method of a standardized solver
        object obtained through `self` must return `True`,
        False otherwise.
    solver_desc : str, optional
        Descriptor for the solver obtained through `self`,
        such as 'backup local solver'
        or 'backup global solver'.
    """

    def __init__(self, require_available=True, solver_desc="solver"):
        self.require_available = require_available
        self.solver_desc = solver_desc

    def __call__(self, obj, require_available=None, solver_desc=None):
        """
        Cast iterable object to a list of Pyomo solver objects.

        Parameters
        ----------
        obj : Iterable
            Object of interest. Should not be of type `str`.
        require_available : bool or None, optional
            True if `available()` method of each solver
            object must return True, False otherwise.
            If `None` is passed, then ``self.require_available``
            is used.
        solver_desc : str or None, optional
            Descriptor for the solver, such as 'backup local solver'
            or 'backup global solver'. This argument is used
            for constructing error/exception messages.
            If `None` is passed, then ``self.solver_desc``
            is used.

        Returns
        -------
        solvers : list of Solver
            List of solver objects to which obj is cast.

        Raises
        ------
        TypeError
            If `obj` is a str.
        """
        # resort to defaults if necessary
        if require_available is None:
            require_available = self.require_available
        if solver_desc is None:
            solver_desc = self.solver_desc

        # set up single standardization callable
        solver_resolve_func = SolverResolvable()

        if isinstance(obj, str):
            # as str is iterable, check explicitly that str not passed,
            # otherwise this method would attempt to resolve each
            # character
            raise TypeError("Object should be an iterable not of type str.")

        # now resolve to list of solver objects
        solvers = []
        obj_as_list = list(obj)
        for idx, val in enumerate(obj_as_list):
            solver_desc_str = (
                f"{solver_desc} "
                f"(index {idx})"
            )
            solvers.append(solver_resolve_func(
                obj=val,
                require_available=require_available,
                solver_desc=solver_desc_str,
            ))

        return solvers

    def domain_name(self):
        return "Iterable of str or Solver"


class PathLikeOrNone:
    """
    Validator for path-like objects.

    This interface is a wrapper around the domain validator
    ``common.config.Path``, and extends the domain of interest to
    to include:
        - None
        - objects following the Python ``os.PathLike`` protocol.

    Parameters
    ----------
    **config_path_kwargs : dict
        Keyword arguments to ``common.config.Path``.
    """
    def __init__(self, **config_path_kwargs):
        """Initialize self (see class docstring)."""
        self.config_path = Path(**config_path_kwargs)

    def __call__(self, path):
        """
        Cast path to expanded string representation.

        Parameters
        ----------
        path : None str, bytes, or path-like
            Object to be cast.

        Returns
        -------
        None
            If obj is None.
        str
            String representation of path-like object.
        """
        if path is None:
            return path

        # cast to str. if not str, bytes, or path-like,
        # expect TypeError to be raised here
        path_str = os.fsdecode(path)

        # expand path using ``common.config.Path`` interface
        return self.config_path(path_str)

    def domain_name(self):
        """str : Brief description of the domain encompassed by self."""
        return "path-like or None"


class ImmutableParamError(Exception):
    """
    Exception raised whenever a Param or ParamData
    object for which we expect ``mutable=True``
    is immutable.
    """


def mutable_param_validator(param_obj):
    """
    Check that Param-like object has attribute `mutable=True`.

    Parameters
    ----------
    param_obj : Param or _ParamData
        Param-like object of interest.

    Raises
    ------
    ValueError
        If lengths of the param object and the accompanying
        index set do not match. This may occur if some entry
        of the Param is not initialized.
    ImmutableParamError
        If attribute `mutable` is of value False.
    """
    if len(param_obj) != len(param_obj.index_set()):
        raise ValueError(
            f"Length of Param component object with "
            f"name {param_obj.name!r} is {len(param_obj)}, "
            "and does not match that of its index set, "
            f"which is of length {len(param_obj.index_set())}. "
            "Check that all entries of the component object "
            "have been initialized."
        )
    if not param_obj.mutable:
        raise ImmutableParamError(
            f"Param object with name {param_obj.name!r} is immutable."
        )


class InputDataStandardizer(object):
    """
    Standardizer for objects castable to a list of Pyomo
    component types.

    Parameters
    ----------
    ctype : type
        Pyomo component type, such as Component, Var or Param.
    cdatatype : type
        Corresponding Pyomo component data type, such as
        _ComponentData, _VarData, or _ParamData.

    Attributes
    ----------
    ctype
    cdatatype
    """
    def __init__(
            self,
            ctype,
            cdatatype,
            ctype_validator=None,
            cdatatype_validator=None,
            allow_repeats=False,
            ):
        """Initialize self (see class docstring).

        """
        self.ctype = ctype
        self.cdatatype = cdatatype
        self.ctype_validator = ctype_validator
        self.cdatatype_validator = cdatatype_validator
        self.allow_repeats = allow_repeats

    def standardize_ctype_obj(self, obj):
        """
        Standardize object of type ``self.ctype`` to list
        of objects of type ``self.cdatatype``.
        """
        if self.ctype_validator is not None:
            self.ctype_validator(obj)
        return list(obj.values())

    def standardize_cdatatype_obj(self, obj):
        """
        Standarize object of type ``self.cdatatype`` to
        ``[obj]``.
        """
        if self.cdatatype_validator is not None:
            self.cdatatype_validator(obj)
        return [obj]

    def __call__(self, obj, from_iterable=None, allow_repeats=None):
        """
        Cast object to a flat list of Pyomo component data type
        entries.

        Parameters
        ----------
        obj : object
            Object to be cast.
        from_iterable : Iterable or None, optional
            Iterable from which `obj` obtained, if any.
        allow_repeats : bool or None, optional
            True if list can contain repeated entries,
            False otherwise.

        Raises
        ------
        TypeError
            If all entries in the resulting list
            are not of type ``self.cdatatype``.
        ValueError
            If the resulting list contains duplicate entries.
        """
        if allow_repeats is None:
            allow_repeats = self.allow_repeats

        if isinstance(obj, self.ctype):
            ans = self.standardize_ctype_obj(obj)
        elif isinstance(obj, self.cdatatype):
            ans = self.standardize_cdatatype_obj(obj)
        elif isinstance(obj, Iterable) and not isinstance(obj, str):
            ans = []
            for item in obj:
                ans.extend(self.__call__(item, from_iterable=obj))
        else:
            from_iterable_qual = (
                f" (entry of iterable {from_iterable})"
                if from_iterable is not None else ""
            )
            raise TypeError(
                f"Input object {obj!r}{from_iterable_qual} "
                "is not of valid component type "
                f"{self.ctype.__name__} or component data type "
                f"{self.cdatatype.__name__}."
            )

        # check for duplicates if desired
        if not allow_repeats and len(ans) != len(ComponentSet(ans)):
            comp_name_list = [comp.name for comp in ans]
            raise ValueError(
                f"Standardized component list {comp_name_list} "
                f"derived from input {obj} "
                "contains duplicate entries."
            )

        return ans

    def domain_name(self):
        """str : Brief description of the domain encompassed by self."""
        return (
            f"{self.cdatatype.__name__}, {self.ctype.__name__}, "
            f"or Iterable of {self.cdatatype.__name__}/{self.ctype.__name__}"
        )


class PyROSConfigValue(ConfigValue):
    """
    Subclass of ``common.collections.ConfigValue``,
    with a few attributes added to facilitate documentation
    of the PyROS solver.
    An instance of this class is used for storing and
    documenting an argument to the PyROS solver.

    Attributes
    ----------
    is_optional : bool
        Argument is optional.
    document_default : bool, optional
        Document the default value of the argument
        in any docstring generated from this instance,
        or a `ConfigDict` object containing this instance.
    dtype_spec_str : None or str, optional
        String documenting valid types for this argument.
        If `None` is provided, then this string is automatically
        determined based on the `domain` argument to the
        constructor.

    NOTES
    -----
    Cleaner way to access protected attributes
    (particularly _doc, _description) inherited from ConfigValue?

    """

    def __init__(
        self,
        default=None,
        domain=None,
        description=None,
        doc=None,
        visibility=0,
        is_optional=True,
        document_default=True,
        dtype_spec_str=None,
    ):
        """Initialize self (see class docstring)."""

        # initialize base class attributes
        super(self.__class__, self).__init__(
            default=default,
            domain=domain,
            description=description,
            doc=doc,
            visibility=visibility,
        )

        self.is_optional = is_optional
        self.document_default = document_default

        if dtype_spec_str is None:
            self.dtype_spec_str = self.domain_name()
            # except AttributeError:
            #     self.dtype_spec_str = repr(self._domain)
        else:
            self.dtype_spec_str = dtype_spec_str


def pyros_config():
    """
    Set up pre-structured ConfigDict for arguments to the
    PyROS solver.

    Returns
    -------
    ConfigDict
        Container for arguments to the PyROS solver.
        Includes argument-wise validators.
    """
    CONFIG = ConfigDict('PyROS')

    # ================================================
    # === Options common to all solvers
    # ================================================
    CONFIG.declare(
        'time_limit',
        PyROSConfigValue(
            default=None,
            domain=NonNegativeFloat,
            doc=(
                """
                Wall time limit for the execution of the PyROS solver
                in seconds (including time spent by subsolvers).
                If `None` is provided, then no time limit is enforced.
                """
            ),
            is_optional=True,
            document_default=False,
            dtype_spec_str="None or NonNegativeFloat",
            visibility=0,
        ),
    )
    CONFIG.declare(
        'keepfiles',
        PyROSConfigValue(
            default=False,
            domain=bool,
            description=(
                """
                Export subproblems with a non-acceptable termination status
                for debugging purposes.
                If True is provided, then the argument `subproblem_file_directory`
                must also be specified.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        'tee',
        PyROSConfigValue(
            default=False,
            domain=bool,
            description="Output subordinate solver logs for all subproblems.",
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        'load_solution',
        PyROSConfigValue(
            default=True,
            domain=bool,
            description=(
                """
                Load final solution(s) found by PyROS to the deterministic model
                provided.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )

    # ================================================
    # === Required User Inputs
    # ================================================
    CONFIG.declare(
        "first_stage_variables",
        PyROSConfigValue(
            default=[],
            domain=InputDataStandardizer(Var, _VarData, allow_repeats=False),
            description="First-stage (or design) variables.",
            is_optional=False,
            dtype_spec_str="VarData, Var, or list of VarData/Var",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "second_stage_variables",
        PyROSConfigValue(
            default=[],
            domain=InputDataStandardizer(Var, _VarData, allow_repeats=False),
            description="Second-stage (or control) variables.",
            is_optional=False,
            dtype_spec_str="VarData, Var, or list of VarData/Var",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "uncertain_params",
        PyROSConfigValue(
            default=[],
            domain=InputDataStandardizer(
                ctype=Param,
                cdatatype=_ParamData,
                ctype_validator=mutable_param_validator,
                allow_repeats=False,
            ),
            description=(
                """
                Uncertain model parameters.
                The `mutable` attribute for all
                Param objects should be set to True.
                """
            ),
            is_optional=False,
            dtype_spec_str="ParamData, Param, or list of ParamData/Param",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "uncertainty_set",
        PyROSConfigValue(
            default=None,
            domain=uncertainty_sets,
            description=(
                """
                Uncertainty set against which the
                final solution(s) returned by PyROS should be certified
                to be robust.
                """
            ),
            is_optional=False,
            dtype_spec_str="UncertaintySet",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "local_solver",
        PyROSConfigValue(
            default=None,
            domain=SolverResolvable(
                solver_desc="local solver",
                require_available=True,
            ),
            description="Subordinate local NLP solver.",
            is_optional=False,
            dtype_spec_str="str or Solver",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "global_solver",
        PyROSConfigValue(
            default=None,
            domain=SolverResolvable(
                solver_desc="global solver",
                require_available=True,
            ),
            description="Subordinate global NLP solver.",
            is_optional=False,
            dtype_spec_str="str or Solver",
            visibility=1,
        ),
    )
    # ================================================
    # === Optional User Inputs
    # ================================================
    CONFIG.declare(
        "objective_focus",
        PyROSConfigValue(
            default=ObjectiveType.nominal,
            domain=InEnum(ObjectiveType),
            description=(
                """
                Choice of objective focus to optimize in the master problems.
                Choices are: `ObjectiveType.worst_case`,
                `ObjectiveType.nominal`.
                """
            ),
            doc=(
                """
                Objective focus for the master problems:
    
                - `ObjectiveType.nominal`:
                  Optimize the objective function subject to the nominal
                  uncertain parameter realization.
                - `ObjectiveType.worst_case`:
                  Optimize the objective function subject to the worst-case
                  uncertain parameter realization.
    
                By default, `ObjectiveType.nominal` is chosen.
    
                A worst-case objective focus is required for certification
                of robust optimality of the final solution(s) returned
                by PyROS.
                If a nominal objective focus is chosen, then only robust
                feasibility is guaranteed.
                """
            ),
            is_optional=True,
            document_default=False,
            dtype_spec_str="ObjectiveType",
            visibility=0,
        ),
    )
    CONFIG.declare(
        "nominal_uncertain_param_vals",
        PyROSConfigValue(
            default=[],
            domain=list,
            doc=(
                """
                Nominal uncertain parameter realization.
                Entries should be provided in an order consistent with the
                entries of the argument `uncertain_params`.
                If an empty list is provided, then the values of the `Param`
                objects specified through `uncertain_params` are chosen.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="iterable of float",
            visibility=0,
        ),
    )
    CONFIG.declare(
        "decision_rule_order",
        PyROSConfigValue(
            default=0,
            domain=In([0, 1, 2]),
            description=(
                """
                Order (or degree) of the polynomial decision rule functions used
                for approximating the adjustability of the second stage
                variables with respect to the uncertain parameters.
                """
            ),
            doc=(
                """
                Order (or degree) of the polynomial decision rule functions used
                for approximating the adjustability of the second stage
                variables with respect to the uncertain parameters.
    
                Choices are:
    
                - 0: static recourse
                - 1: affine recourse
                - 2: quadratic recourse
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        "solve_master_globally",
        PyROSConfigValue(
            default=False,
            domain=bool,
            doc=(
                """
                True to solve all master problems with the subordinate
                global solver, False to solve all master problems with
                the subordinate local solver.
                Along with a worst-case objective focus
                (see argument `objective_focus`),
                solving the master problems to global optimality is required
                for certification
                of robust optimality of the final solution(s) returned
                by PyROS. Otherwise, only robust feasibility is guaranteed.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        "max_iter",
        PyROSConfigValue(
            default=-1,
            domain=PositiveIntOrMinusOne(),
            description=(
                """
                Iteration limit. If -1 is provided, then no iteration
                limit is enforced.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="int",
            visibility=0,
        ),
    )
    CONFIG.declare(
        "robust_feasibility_tolerance",
        PyROSConfigValue(
            default=1e-4,
            domain=NonNegativeFloat,
            description=(
                """
                Relative tolerance for assessing maximal inequality
                constraint violations during the GRCS separation step.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        "separation_priority_order",
        PyROSConfigValue(
            default={},
            domain=dict,
            doc=(
                """
                Mapping from model inequality constraint names
                to positive integers specifying the priorities
                of their corresponding separation subproblems.
                A higher integer value indicates a higher priority.
                Constraints not referenced in the `dict` assume
                a priority of 0.
                Separation subproblems are solved in order of decreasing
                priority.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        "progress_logger",
        PyROSConfigValue(
            default=setup_pyros_logger(),
            domain=LoggerType(),
            doc=(
                """
                Logger (or name thereof) used for reporting PyROS solver
                progress. If a `str` is specified, then ``progress_logger``
                is cast to ``logging.getLogger(progress_logger)``.
                In the default case, `progress_logger` is set to
                a :class:`pyomo.contrib.pyros.util.PreformattedLogger`
                object of level ``logging.INFO``.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="str or logging.Logger",
            visibility=0,
        ),
    )
    CONFIG.declare(
        "backup_local_solvers",
        PyROSConfigValue(
            default=[],
            domain=SolverIterable(
                solver_desc="backup local solver",
                require_available=True,
            ),
            doc=(
                """
                Additional subordinate local NLP optimizers to invoke
                in the event the primary local NLP optimizer fails
                to solve a subproblem to an acceptable termination condition.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="Iterable of str or Solver",
            visibility=0,
        ),
    )
    CONFIG.declare(
        "backup_global_solvers",
        PyROSConfigValue(
            default=[],
            domain=SolverIterable(
                solver_desc="backup global solver",
                require_available=True,
            ),
            doc=(
                """
                Additional subordinate global NLP optimizers to invoke
                in the event the primary global NLP optimizer fails
                to solve a subproblem to an acceptable termination condition.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="Iterable of str or Solver",
            visibility=0,
        ),
    )
    CONFIG.declare(
        "subproblem_file_directory",
        PyROSConfigValue(
            default=None,
            domain=PathLikeOrNone(),
            description=(
                """
                Path of directory to which
                to export subproblems not successfully
                solved to an acceptable termination condition.
                If a path is specified, i.e. str, bytes,
                or path-like value is passed, then the directory
                to which it refers must exist.
                Subproblems are exported only if a path is specified
                and user passes argument ``keepfiles=True``.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="None, str, bytes, or path-like",
            visibility=0,
        ),
    )

    # ================================================
    # === Advanced Options
    # ================================================
    CONFIG.declare(
        "bypass_local_separation",
        PyROSConfigValue(
            default=False,
            domain=bool,
            description=(
                """
                This is an advanced option.
                Solve all separation subproblems with the subordinate global
                solver(s) only.
                This option is useful for expediting PyROS
                in the event that the subordinate global optimizer(s) provided
                can quickly solve separation subproblems to global optimality.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        "bypass_global_separation",
        PyROSConfigValue(
            default=False,
            domain=bool,
            doc=(
                """
                This is an advanced option.
                Solve all separation subproblems with the subordinate local
                solver(s) only.
                If `True` is chosen, then robustness of the final solution(s)
                returned by PyROS is not guaranteed, and a warning will
                be issued at termination.
                This option is useful for expediting PyROS
                in the event that the subordinate global optimizer provided
                cannot tractably solve separation subproblems to global
                optimality.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )
    CONFIG.declare(
        "p_robustness",
        PyROSConfigValue(
            default={},
            domain=dict,
            doc=(
                """
                This is an advanced option.
                Add p-robustness constraints to all master subproblems.
                If an empty dict is provided, then p-robustness constraints
                are not added.
                Otherwise, the dict must map a `str` of value ``'rho'``
                to a non-negative `float`. PyROS automatically
                specifies ``1 + p_robustness['rho']``
                as an upper bound for the ratio of the
                objective function value under any PyROS-sampled uncertain
                parameter realization to the objective function under
                the nominal parameter realization.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
            visibility=0,
        ),
    )

    return CONFIG


def resolve_keyword_arguments(prioritized_kwargs_dicts, func=None):
    """
    Resolve keyword arguments to a callable, each of which may
    be passed through one of multiple possible dicts.

    Parameters
    ----------
    prioritized_kwargs_dicts : dict
        Each entry maps a str to a dict of keyword arguments
        described by the str. Entries are taken to be provided
        in descending order of priority.
    func : callable or None, optional
        Callable to which the keyword arguments are/were passed;
        only the `__name__` attribute is used for logging
        warnings. If `None` is passed, then warning messages
        logged are slightly less informative.

    Returns
    -------
    resolved_kwargs : dict
        Resolved keyword arguments.
    """
    # warnings are issued through logger object
    default_logger = default_pyros_solver_logger

    # used for warning messages
    func_desc = f"passed to {func.__name__}()" if func is not None else "passed"

    # we will loop through the priority dict. initialize:
    # - resolved keyword arguments, taking into account the
    #   priority order and overlap
    # - kwarg dicts already processed
    # - sequence of kwarg dicts yet to be processed
    resolved_kwargs = dict()
    prev_prioritized_kwargs_dicts = dict()
    remaining_kwargs_dicts = prioritized_kwargs_dicts.copy()
    for curr_desc, curr_kwargs in remaining_kwargs_dicts.items():
        overlapping_args = dict()
        overlapping_args_set = set()

        for prev_desc, prev_kwargs in prev_prioritized_kwargs_dicts.items():
            # determine overlap between currrent and previous
            # set of kwargs, and remove overlap of current
            # and higher priority sets from the result
            curr_prev_overlapping_args = (
                (set(curr_kwargs.keys()) & set(prev_kwargs.keys()))
                - overlapping_args_set
            )
            if curr_prev_overlapping_args:
                # if there is overlap, prepare overlapping args
                # for when warning is to be issued
                overlapping_args[prev_desc] = curr_prev_overlapping_args

            # update set of args overlapping with higher priority dicts
            overlapping_args_set |= curr_prev_overlapping_args

        # ensure kwargs specified in higher priority
        # dicts are not overwritten in resolved kwargs
        resolved_kwargs.update({
            kw: val
            for kw, val in curr_kwargs.items()
            if kw not in overlapping_args_set
        })

        # if there are overlaps, log warnings accordingly
        # per priority level
        for overlap_desc, args_set in overlapping_args.items():
            new_overlapping_args_str = ", ".join(
                f"{arg!r}" for arg in args_set
            )
            default_logger.warning(
                f"Arguments [{new_overlapping_args_str}] passed {curr_desc} "
                f"already {func_desc} {overlap_desc}, "
                "and will not be overwritten. "
                "Consider modifying your arguments to remove the overlap."
            )

        # increment sequence of kwarg dicts already processed
        prev_prioritized_kwargs_dicts[curr_desc] = curr_kwargs

    return resolved_kwargs


def check_components_descended_from_model(model, components, components_name, config):
    """
    Check all members in a provided sequence of Pyomo component
    objects are descended from a given ConcreteModel object.

    Parameters
    ----------
    model : ConcreteModel
        Model from which components should all be descended.
    components : Iterable of Component
        Components of interest.
    components_name : str
        Brief description or name for the sequence of components.
        Used for constructing error messages.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one entry of `components` is not descended
        from `model`.
    """
    components_not_in_model = [
        comp for comp in components
        if comp.model() is not model
    ]
    if components_not_in_model:
        comp_names_str = "\n ".join(
            f"{comp.name!r}, from model with name {comp.model().name!r}"
            for comp in components_not_in_model
        )
        config.progress_logger.error(
            f"The following entries of argument `{components_name}` "
            "are not descended from the "
            f"input deterministic model with name {model.name!r}:\n "
            f"{comp_names_str}"
        )
        raise ValueError(
            f"Found entries of argument `{components_name}` "
            "not descended from input model. "
            "Check logger output messages."
        )


def check_variables_continuous(model, config):
    """
    Check that all DOF and state variables of the model
    are continuous.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one variable is found to not be continuous.

    Note
    ----
    A variable is considered continuous if the `is_continuous()`
    method returns True.
    """
    state_vars = list(get_state_vars(
        blk=model,
        first_stage_variables=config.first_stage_variables,
        second_stage_variables=config.second_stage_variables,
    ))
    all_vars = config.first_stage_variables + config.second_stage_variables + state_vars

    # ensure all variables continuous
    non_continuous_vars = [
        var for var in all_vars
        if not var.is_continuous()
    ]
    if non_continuous_vars:
        non_continuous_vars_str = "\n ".join(
            f"{var.name!r}" for var in non_continuous_vars
        )
        config.progress_logger.error(
            f"The following Vars of model with name {model.name!r} "
            f"are non-continuous:\n {non_continuous_vars_str}\n"
            "Ensure all model variables passed to PyROS solver are continuous."
        )
        raise ValueError(
            f"Model with name {model.name!r} contains non-continuous Vars."
        )


def validate_model(model, config):
    """
    Validate deterministic model passed to PyROS solver.

    Parameters
    ----------
    model : ConcreteModel
        Determinstic model. Should have only one active Objective.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    ComponentSet
        The variables participating in the active Objective
        and Constraint expressions of `model`.

    Raises
    ------
    TypeError
        If model is not of type ConcreteModel.
    ValueError
        If model does not have exactly one active Objective
        component, or at least one variable participating in the
        active objective/constraints of the model is not
        descended from the model.
    """
    # note: only support ConcreteModel. no support for Blocks
    if not isinstance(model, ConcreteModel):
        raise TypeError(
            f"Model should be of type {ConcreteModel.__name__}, "
            f"but is of type {type(model).__name__}."
        )

    # active objectives check
    active_objs_list = list(
        model.component_data_objects(Objective, active=True, descend_into=True)
    )
    if len(active_objs_list) != 1:
        raise ValueError(
            "Expected model with exactly 1 active objective, but "
            f"model provided has {len(active_objs_list)}."
        )
    active_obj = active_objs_list[0]

    # variables check
    vars_in_active_cons = ComponentSet(get_vars_from_component(model, Constraint))
    vars_in_active_obj = ComponentSet(identify_variables(active_obj))
    vars_not_in_model = [
        var for var in vars_in_active_cons | vars_in_active_obj if
        var.model() is not model
    ]
    if vars_not_in_model:
        vars_str = "\n ".join(
            f"{var.name!r}, from model with name {var.model().name!r}"
            for var in vars_not_in_model
        )
        config.progress_logger.error(
            f"The following Vars participating in the active Objective or "
            f"Constraints of the model are not components of "
            f"the model with name {model.name!r}:\n {vars_str}\n"
            "Ensure all Vars participating in active objective or constraints "
            f"are components of the model with name {model.name!r}."
        )
        raise ValueError(
            "Found Vars in active objective/constraints of model "
            "not descended from model."
        )

    return vars_in_active_obj | vars_in_active_cons


def validate_variable_partitioning(model, config):
    """
    Check that partitioning of the first-stage variables,
    second-stage variables, and uncertain parameters
    is valid.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If first-stage variables and second-stage variables
        overlap, or there are no first-stage variables
        and no second-stage variables.
    """
    # at least one DOF required
    if not config.first_stage_variables and not config.second_stage_variables:
        raise ValueError(
            "Arguments `first_stage_variables` and "
            "`second_stage_variables` are both empty lists."
        )

    # ensure no overlap between DOF var sets
    overlapping_vars = (
        ComponentSet(config.first_stage_variables)
        & ComponentSet(config.second_stage_variables)
    )
    if overlapping_vars:
        overlapping_var_list = "\n ".join(f"{var.name!r}" for var in overlapping_vars)
        config.progress_logger.error(
            "The following Vars were found in both `first_stage_variables`"
            f"and `second_stage_variables`:\n {overlapping_var_list}"
            "\nEnsure no Vars are included in both arguments."
        )
        raise ValueError(
            "Arguments `first_stage_variables` and `second_stage_variables` "
            "contain at least one common Var object."
        )

    # DOF variables should all be descended from model.
    check_components_descended_from_model(
        model=model,
        components=config.first_stage_variables,
        components_name="first_stage_variables",
        config=config,
    )
    check_components_descended_from_model(
        model=model,
        components=config.second_stage_variables,
        components_name="second_stage_variables",
        config=config,
    )

    # all variables should be continuous
    check_variables_continuous(model, config)


def validate_uncertainty_specification(model, config):
    """
    Validate specification of uncertain parameters and uncertainty
    set.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one of the following holds:

        - dimension of uncertainty set does not equal number of
          uncertain parameters
        - uncertainty set `is_valid()` method does not return
          true.
        - nominal parameter realization is not in the uncertainty set.
    """
    check_components_descended_from_model(
        model=model,
        components=config.uncertain_params,
        components_name="uncertain_params",
        config=config,
    )

    if len(config.uncertain_params) != config.uncertainty_set.dim:
        raise ValueError(
            "Length of argument `uncertain_params` does not match dimension "
            "of argument `uncertainty_set` "
            f"({len(config.uncertain_params)} != {config.uncertainty_set.dim})."
        )

    # validate uncertainty set
    if not config.uncertainty_set.is_valid(config=config):
        raise ValueError(
            f"Uncertainty set {config.uncertainty_set} is invalid, "
            "as it is either empty or unbounded."
        )

    # fill-in nominal point as necessary, if not provided.
    # otherwise, check length matches uncertainty dimension
    if not config.nominal_uncertain_param_vals:
        config.nominal_uncertain_param_vals = [
            value(param, exception=True)
            for param in config.uncertain_params
        ]
    elif len(config.nominal_uncertain_param_vals) != len(config.uncertain_params):
        raise ValueError(
            "Lengths of arguments `uncertain_params` and "
            "`nominal_uncertain_param_vals` "
            "do not match "
            f"({len(config.uncertain_params)} != "
            f"{len(config.nominal_uncertain_param_vals)})."
        )

    # uncertainty set should contain nominal point
    nominal_point_in_set = config.uncertainty_set.point_in_set(
        point=config.nominal_uncertain_param_vals
    )
    if not nominal_point_in_set:
        raise ValueError(
            "Nominal uncertain parameter realization "
            f"{config.nominal_uncertain_param_vals} "
            "is not a point in the uncertainty set "
            f"{config.uncertainty_set!r}."
        )


def validate_separation_problem_options(model, config):
    """
    Validate separation problem arguments to the PyROS solver.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If options `bypass_local_separation` and
        `bypass_global_separation` are set to False.
    """
    if config.bypass_local_separation and config.bypass_global_separation:
        raise ValueError(
            "Arguments `bypass_local_separation` "
            "and `bypass_global_separation` "
            "cannot both be True."
        )


def validate_pyros_inputs(model, config):
    """
    Perform advanced validation of PyROS solver arguments.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.
    """
    validate_model(model, config)
    validate_variable_partitioning(model, config)
    validate_uncertainty_specification(model, config)
    validate_separation_problem_options(model, config)


def wrap_doc(doc, indent_by, width):
    """
    Wrap a string, accounting for paragraph
    breaks ('\n\n') and bullet points (paragraphs
    which, when dedented, are such that each line
    starts with '- ' or '  ').
    """
    paragraphs = doc.split("\n\n")
    wrapped_pars = []
    for par in paragraphs:
        lines = dedent(par).split("\n")
        has_bullets = all(
            line.startswith("- ") or line.startswith("  ")
            for line in lines
            if line != ""
        )
        if has_bullets:
            # obtain strings of each bullet point
            # (dedented, bullet dash and bullet indent removed)
            bullet_groups = []
            new_group = False
            group = ""
            for line in lines:
                new_group = line.startswith("- ")
                if new_group:
                    bullet_groups.append(group)
                    group = ""
                new_line = line[2:]
                group += f"{new_line}\n"
            if group != "":
                # ensure last bullet not skipped
                bullet_groups.append(group)

            # first entry is just ''; remove
            bullet_groups = bullet_groups[1:]

            # wrap each bullet point, then add bullet
            # and indents as necessary
            wrapped_groups = []
            for group in bullet_groups:
                wrapped_groups.append(
                    "\n".join(
                        f"{'- ' if idx == 0 else '  '}{line}"
                        for idx, line in enumerate(
                            wrap(group, width - 2 - indent_by)
                        )
                    )
                )

            # now combine bullets into single 'paragraph'
            wrapped_pars.append(
                indent("\n".join(wrapped_groups), prefix=' ' * indent_by)
            )
        else:
            wrapped_pars.append(
                indent(
                    "\n".join(wrap(dedent(par), width=width - indent_by)),
                    prefix=' ' * indent_by,
                )
            )

    return "\n\n".join(wrapped_pars)


def add_filtered_config_section_to_docstring(
        func,
        config,
        visibility=0,
        section="Keyword Arguments",
        indent_by=8,
        width=72,
        ):
    """
    Add section enumerating entries of a `ConfigDict` to
    the docstring of a callable.

    func : callable
        Function of which docstring is to be modified.
    config : ConfigDict
        Specification of arguments to the function.
    visibility : int, optional
        Visibility filter.
        Maximum visibility for which a member of `config`
        will be listed in the added section.
    section : str, optional
        Title of the section to be added.
    indent_by : int, optional
        Number of spaces by which to indent each line of the
        docstring.
    width : 72
        Maximum line width of the docstring (including indents).

    Returns
    -------
    str
        Modified docstring.
    """
    before = func.__doc__

    indent_str = ' ' * indent_by
    wrap_width = width - indent_by

    arg_docs = []

    section_header = indent(f"{section}\n" + "-" * len(section), indent_str)
    for key, itm in config._data.items():
        if itm._visibility > visibility:
            continue
        arg_name = key
        arg_dtype = itm.dtype_spec_str

        if itm.is_optional:
            if itm.document_default:
                optional_str = f", default={repr(itm._default)}"
            else:
                optional_str = ", optional"
        else:
            optional_str = ""

        arg_header = f"{indent_str}{arg_name} : {arg_dtype}{optional_str}"

        # dedented_doc_str = dedent(itm.doc).replace("\n", ' ').strip()
        if itm._doc is not None:
            raw_arg_desc = itm._doc
        else:
            raw_arg_desc = itm._description

        arg_description = wrap_doc(
            raw_arg_desc, width=wrap_width, indent_by=indent_by + 4
        )

        arg_docs.append(f"{arg_header}\n{arg_description}")

    kwargs_section_doc = "\n".join([section_header] + arg_docs)

    return f"{before}\n{kwargs_section_doc}\n"


def add_config_kwargs_to_doc(**doc_kwargs):
    """
    Create function decorator for adding keyword arguments from
    members of a ConfigDict to the function docstring.

    Parameters
    ----------
    **doc_kwargs : dict, optional
        Keyword arguments to ``generate_filtered_docstring``.

    Returns
    -------
    callable
        Function decorator.
    """
    def decorator_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = add_filtered_config_section_to_docstring(
            func=func,
            **doc_kwargs,
        )
        return wrapper

    return decorator_func
