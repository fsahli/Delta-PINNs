import jax
from jax import core
from graphviz import Digraph
import itertools


styles = {
  'const': dict(style='filled', color='goldenrod1'),
  'invar': dict(color='mediumspringgreen', style='filled'),
  'outvar': dict(style='filled,dashed', fillcolor='indianred1', color='black'),
  'op_node': dict(shape='box', color='lightskyblue', style='filled'),
  'intermediate': dict(style='filled', color='cornflowerblue')
}

def _jaxpr_graph(jaxpr):
    id_names = (f'id{id}' for id in itertools.count())
    graph = Digraph(engine='dot')
    graph.attr(size='6,10!')
    for v in jaxpr.constvars:
        graph.node(str(v), core.raise_to_shaped(v.aval).str_short(), styles['const'])
    for v in jaxpr.invars:
        graph.node(str(v), v.aval.str_short(), styles['invar'])
    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            if isinstance(v, core.Literal):
                graph.node(str(id(v.val)), core.raise_to_shaped(core.get_aval(v.val)).str_short(), styles['const'])
        if eqn.primitive.multiple_results:
            id_name = next(id_names)
            graph.node(id_name, str(eqn.primitive), styles['op_node'])
            for v in eqn.invars:
                graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), id_name)
            for v in eqn.outvars:
                graph.node(str(v), v.aval.str_short(), styles['intermediate'])
                graph.edge(id_name, str(v))
        else:
            outv, = eqn.outvars
            graph.node(str(outv), str(eqn.primitive), styles['op_node'])
            for v in eqn.invars:
                graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), str(outv))
    for i, v in enumerate(jaxpr.outvars):
        outv = 'out_'+str(i)
        graph.node(outv, outv, styles['outvar'])
        graph.edge(str(v), outv)
    return graph


def jaxpr_graph(fun, *args):
    jaxpr = jax.make_jaxpr(fun)(*args).jaxpr
    return _jaxpr_graph(jaxpr)


def grad_graph(fun, *args):
    _, fun_vjp = jax.vjp(fun, *args)
    jaxpr = fun_vjp.args[0].func.args[1]
    return _jaxpr_graph(jaxpr)


# example use
# import jax.numpy as jnp
# f = lambda x: jnp.sum(x**2)
# x = jnp.ones(5)
# g = grad_graph(f, x)
# g = jaxpr_graph(f, x)
# g = jaxpr_graph(jax.grad(f), x)
# g # will show inline in a notebook (alt: g.view() creates & opens Digraph.gv.pdf)
