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


import pyomo.environ as pyo


def pipe_rule(pipe, i):
    m = pipe.model()
    pipe.flow = pyo.Var()
    pipe.pIn = pyo.Var(within=pyo.NonNegativeReals)
    pipe.pOut = pyo.Var(within=pyo.NonNegativeReals)
    pipe.pDrop = pyo.Constraint(
        expr=pipe.pIn - pipe.pOut == m.friction * m.pipe_length[i] * pipe.flow
    )

    pipe.IN = pyo.Connector()
    pipe.IN.add(-pipe.flow, "flow")
    pipe.IN.add(pipe.pIn, "pressure")

    pipe.OUT = pyo.Connector()
    pipe.OUT.add(pipe.flow)
    pipe.OUT.add(pipe.pOut, "pressure")


def node_rule(node, i):
    def _mass_balance(node, flows):
        return node.model().demands[i] == pyo.sum_product(flows)

    node.flow = pyo.VarList()
    node.pressure = pyo.Var(within=pyo.NonNegativeReals)
    node.port = pyo.Connector()
    # node.port.add(node.flow,
    #              aggregate=lambda n,v: n.model().demands[id] == pyo.sum_product(v))
    node.port.add(node.flow, aggregate=_mass_balance)
    node.port.add(node.pressure)


def _src_rule(model, pipe):
    return model.nodes[pyo.value(model.pipe_links[pipe, 0])].port == model.pipes[pipe].IN


def _sink_rule(model, pipe):
    return model.nodes[pyo.value(model.pipe_links[pipe, 1])].port == model.pipes[pipe].OUT


model = pyo.AbstractModel()
model.PIPES = pyo.Set()
model.NODES = pyo.Set()

model.friction = pyo.Param(within=pyo.NonNegativeReals)
model.pipe_length = pyo.Param(model.PIPES, within=pyo.NonNegativeReals)
model.pipe_links = pyo.Param(model.PIPES, [0, 1])
model.demands = pyo.Param(model.NODES, within=pyo.Reals, default=0)

model.pipes = pyo.Block(model.PIPES, rule=pipe_rule)
model.nodes = pyo.Block(model.NODES, rule=node_rule)

# Connect the network
model.network_src = pyo.Constraint(model.PIPES, rule=_src_rule)
model.network_sink = pyo.Constraint(model.PIPES, rule=_sink_rule)


# Solve so the minimum pressure in the network is 0
def _obj(model):
    return sum(model.nodes[n].pressure for n in model.NODES)


model.obj = pyo.Objective(rule=_obj)
