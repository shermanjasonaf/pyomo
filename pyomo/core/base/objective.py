#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload

from pyomo.common.deprecation import RenamedClass, deprecated
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.enums import ObjectiveSense, minimize, maximize
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer

from pyomo.core.expr.expr_common import _type_check_exception_arg
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.template_expr import templatize_rule
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent,
    UnindexedComponent_set,
    rule_wrapper,
)
from pyomo.core.base.expression import NamedExpressionData
from pyomo.core.base.set import Set
from pyomo.core.base.initializer import (
    Initializer,
    IndexedCallInitializer,
    CountedCallInitializer,
)

logger = logging.getLogger('pyomo.core')

TEMPLATIZE_OBJECTIVES = False

_rule_returned_none_error = """Objective '%s': rule returned None.

Objective rules must return either a valid expression, numeric value, or
Objective.Skip.  The most common cause of this error is forgetting to
include the "return" statement at the end of your rule.
"""


def simple_objective_rule(rule):
    """
    This is a decorator that translates None into Objective.Skip.
    This supports a simpler syntax in objective rules, though these
    can be more difficult to debug when errors occur.

    Example use:

    .. code::

        @simple_objective_rule
        def O_rule(model, i, j):
            # ...

        model.o = Objective(rule=simple_objective_rule(...))

    """
    return rule_wrapper(rule, {None: Objective.Skip})


def simple_objectivelist_rule(rule):
    """
    This is a decorator that translates None into ObjectiveList.End.
    This supports a simpler syntax in objective rules, though these
    can be more difficult to debug when errors occur.

    Example use:

    .. code::

        @simple_objectivelist_rule
        def O_rule(model, i, j):
            # ...

        model.o = ObjectiveList(expr=simple_objectivelist_rule(...))

    """
    return rule_wrapper(rule, {None: ObjectiveList.End})


class ObjectiveData(NamedExpressionData, ActiveComponentData):
    """This class defines the data for a single objective.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Parameters
    ----------
    expr:
        The Pyomo expression stored in this objective.

    sense:
        The direction for this objective.

    component: Objective
        The Objective object that owns this data.

    Attributes
    ----------
    expr:
        The Pyomo expression for this objective

    """

    __slots__ = ("_args_", "_sense")

    def __init__(self, expr=None, sense=minimize, component=None):
        # Inlining NamedExpressionData.__init__
        self._args_ = (expr,)
        # Inlining ActiveComponentData.__init__
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET
        self._active = True
        self._sense = ObjectiveSense(sense)

    def is_minimizing(self):
        """Return True if this is a minimization objective."""
        return self.sense == minimize

    def set_value(self, expr):
        if expr is None:
            raise ValueError(_rule_returned_none_error % (self.name,))
        return super().set_value(expr)

    #
    # Abstract Interface
    #

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        return self._sense

    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        self._sense = ObjectiveSense(sense)


class _ObjectiveData(metaclass=RenamedClass):
    __renamed__new_class__ = ObjectiveData
    __renamed__version__ = '6.7.2'


class _GeneralObjectiveData(metaclass=RenamedClass):
    __renamed__new_class__ = ObjectiveData
    __renamed__version__ = '6.7.2'


class TemplateObjectiveData(ObjectiveData):
    __slots__ = ()

    def __init__(self, template_info, component, index, sense):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ObjectiveData
        #   - ActiveComponentData
        #   - ComponentData
        self._component = component
        self._active = True
        self._index = index
        self._args_ = template_info
        self._sense = sense

    @property
    def args(self):
        # Note that it is faster to just generate the expression from
        # scratch than it is to clone it and replace the IndexTemplate objects
        self.set_value(self.parent_component()._rule(self.parent_block(), self.index()))
        return self._args_

    def template_expr(self):
        return self._args_

    def set_value(self, expr):
        self.__class__ = ObjectiveData
        return self.set_value(expr)


@ModelComponentFactory.register("Expressions that are minimized or maximized.")
class Objective(ActiveIndexedComponent):
    """
    This modeling component defines an objective expression.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        expr
            A Pyomo expression for this objective
        rule
            A function that is used to construct objective expressions
        sense
            Indicate whether minimizing (the default) or maximizing
        name
            A name for this component
        doc
            A text string describing this component

    Public class attributes:
        doc
            A text string describing this component
        name
            A name for this component
        active
            A boolean that is true if this component will be used to construct
            a model instance
        rule
            The rule used to initialize the objective(s)
        sense
            The objective sense

    Private class attributes:
        _constructed
            A boolean that is true if this component has been constructed
        _data
            A dictionary from the index set to component data objects
        _index
            The set of valid indices
        _model
            A weakref to the model that owns this component
        _parent
            A weakref to the parent block that owns this component
        _type
            The class type for the derived subclass
    """

    _ComponentDataClass = ObjectiveData
    NoObjective = ActiveIndexedComponent.Skip

    def __new__(cls, *args, **kwds):
        if cls != Objective:
            return super(Objective, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return ScalarObjective.__new__(ScalarObjective)
        else:
            return IndexedObjective.__new__(IndexedObjective)

    @overload
    def __init__(
        self, *indexes, expr=None, rule=None, sense=minimize, name=None, doc=None
    ): ...

    def __init__(self, *args, **kwargs):
        _sense = kwargs.pop('sense', minimize)
        _init = self._pop_from_kwargs('Objective', kwargs, ('rule', 'expr'), None)

        kwargs.setdefault('ctype', Objective)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

        self._rule = Initializer(_init)
        self._init_sense = Initializer(_sense)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug("Constructing objective %s" % (self.name))

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        rule = self._rule
        try:
            # We do not (currently) accept data for constructing Objectives
            index = None
            assert data is None

            if rule is None:
                # If there is no rule, then we are immediately done.
                return

            if rule.constant() and self.is_indexed():
                raise IndexError(
                    "Objective '%s': Cannot initialize multiple indices "
                    "of an objective with a single expression" % (self.name,)
                )

            block = self.parent_block()
            if rule.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in rule.indices():
                    ans = self.__setitem__(index, rule(block, index))
                    if ans is not None:
                        self[index].set_sense(self._init_sense(block, index))
            elif not self.index_set().isfinite():
                # If the index is not finite, then we cannot iterate
                # over it.  Since the rule doesn't provide explicit
                # indices, then there is nothing we can do (the
                # assumption is that the user will trigger specific
                # indices to be created at a later time).
                pass
            else:
                if TEMPLATIZE_OBJECTIVES:
                    try:
                        template_info = templatize_rule(block, rule, self.index_set())
                        comp = weakref_ref(self)
                        self._data = {
                            idx: TemplateObjectiveData(
                                template_info, comp, idx, self._init_sense(block, index)
                            )
                            for idx in self.index_set()
                        }
                        return
                    except TemplateExpressionError:
                        pass

                # Bypass the index validation and create the member directly
                for index in self.index_set():
                    ans = self._setitem_when_not_present(index, rule(block, index))
                    if ans is not None:
                        ans.set_sense(self._init_sense(block, index))
        except Exception:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed when generating expression for "
                "Objective %s with index %s:\n%s: %s"
                % (self.name, str(index), type(err).__name__, err)
            )
            raise
        finally:
            timer.report()

    def _getitem_when_not_present(self, index):
        if self._rule is None:
            raise KeyError(index)

        block = self.parent_block()
        obj = self._setitem_when_not_present(index, self._rule(block, index))
        if obj is None:
            raise KeyError(index)
        obj.set_sense(self._init_sense(block, index))

        return obj

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [
                ("Size", len(self)),
                ("Index", self._index_set if self.is_indexed() else None),
                ("Active", self.active),
            ],
            self._data.items(),
            ("Active", "Sense", "Expression"),
            lambda k, v: [v.active, v.sense, v.expr],
        )

    @property
    def rule(self):
        return self._rule

    @rule.setter
    @deprecated(
        f"The 'Objective.rule' attribute will be made "
        "read-only in a future Pyomo release.",
        version='6.9.3.dev0',
        remove_in='6.11',
    )
    def rule(self, rule):
        self._rule = rule

    def display(self, prefix="", ostream=None):
        """Provide a verbose display of this object"""
        if not self.active:
            return
        tab = "    "
        if ostream is None:
            ostream = sys.stdout
        ostream.write(prefix + self.local_name + " : ")
        ostream.write(
            ", ".join(
                "%s=%s" % (k, v)
                for k, v in [
                    ("Size", len(self)),
                    ("Index", self._index_set if self.is_indexed() else None),
                    ("Active", self.active),
                ]
            )
        )

        ostream.write("\n")
        tabular_writer(
            ostream,
            prefix + tab,
            ((k, v) for k, v in self._data.items() if v.active),
            ("Active", "Value"),
            lambda k, v: [v.active, value(v)],
        )


class ScalarObjective(ObjectiveData, Objective):
    """
    ScalarObjective is the implementation representing a single,
    non-indexed objective.
    """

    def __init__(self, *args, **kwd):
        ObjectiveData.__init__(self, expr=None, component=self)
        Objective.__init__(self, *args, **kwd)
        self._index = UnindexedComponent_index

    #
    # Override abstract interface methods to first check for
    # construction
    #

    def __call__(self, exception=NOTSET):
        exception = _type_check_exception_arg(self, exception)
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Evaluating the expression of ScalarObjective "
                    "'%s' before the Objective has been assigned "
                    "a sense or expression (there is currently "
                    "no value to return)." % (self.name)
                )
            return super().__call__(exception)
        raise ValueError(
            "Evaluating the expression of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    @property
    def expr(self):
        """Access the expression of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the expression of ScalarObjective "
                    "'%s' before the Objective has been assigned "
                    "a sense or expression (there is currently "
                    "no value to return)." % (self.name)
                )
            return ObjectiveData.expr.fget(self)
        raise ValueError(
            "Accessing the expression of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    @expr.setter
    def expr(self, expr):
        """Set the expression of this objective."""
        self.set_value(expr)

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the sense of ScalarObjective "
                    "'%s' before the Objective has been assigned "
                    "a sense or expression (there is currently "
                    "no value to return)." % (self.name)
                )
            return ObjectiveData.sense.fget(self)
        raise ValueError(
            "Accessing the sense of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    #
    # Singleton objectives are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Objective.Skip are managed. But after that they will behave
    # like ObjectiveData objects where set_value does not handle
    # Objective.Skip but expects a valid expression or None
    #

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression of this objective."""
        if not self._constructed:
            raise ValueError(
                "Setting the value of objective '%s' "
                "before the Objective has been constructed (there "
                "is currently no object to set)." % (self.name)
            )
        if not self._data:
            self._data[None] = self
        return super().set_value(expr)

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return ObjectiveData.set_sense(self, sense)
        raise ValueError(
            "Setting the sense of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no object to set)." % (self.name)
        )

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise ValueError(
                "ScalarObjective object '%s' does not accept "
                "index values other than None. Invalid value: %s" % (self.name, index)
            )
        self.set_value(expr)
        return self


class SimpleObjective(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarObjective
    __renamed__version__ = '6.0'


class IndexedObjective(Objective):
    #
    # Leaving this method for backward compatibility reasons
    #
    # Note: Beginning after Pyomo 5.2 this method will now validate that
    # the index is in the underlying index set (through 5.2 the index
    # was not checked).
    #
    def add(self, index, expr):
        """Add an objective with a given index."""
        return self.__setitem__(index, expr)


@ModelComponentFactory.register("A list of objective expressions.")
class ObjectiveList(IndexedObjective):
    """
    An objective component that represents a list of objectives.
    Objectives can be indexed by their index, but when they are added
    an index value is not specified.
    """

    class End(object):
        pass

    def __init__(self, **kwargs):
        """Constructor"""
        if 'expr' in kwargs:
            raise ValueError("ObjectiveList does not accept the 'expr' keyword")
        _rule = kwargs.pop('rule', None)
        self._starting_index = kwargs.pop('starting_index', 1)

        super().__init__(Set(dimen=1), **kwargs)

        self._rule = Initializer(_rule, allow_generators=True)
        # HACK to make the "counted call" syntax work.  We wait until
        # after the base class is set up so that is_indexed() is
        # reliable.
        if self._rule is not None and type(self._rule) is IndexedCallInitializer:
            self._rule = CountedCallInitializer(self, self._rule, self._starting_index)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        if self._constructed:
            return
        self._constructed = True

        if is_debug_set(logger):
            logger.debug("Constructing objective list %s" % (self.name))

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        if self._rule is not None:
            _rule = self._rule(self.parent_block(), ())
            for cc in iter(_rule):
                if cc is ObjectiveList.End:
                    break
                if cc is Objective.Skip:
                    continue
                self.add(cc, sense=self._init_sense)

    def add(self, expr, sense=minimize):
        """Add an objective to the list."""
        next_idx = len(self._index_set) + self._starting_index
        self._index_set.add(next_idx)
        ans = self.__setitem__(next_idx, expr)
        if ans is not None:
            if sense not in {minimize, maximize}:
                sense = sense(self.parent_block(), next_idx)
            ans.set_sense(sense)
        return ans
