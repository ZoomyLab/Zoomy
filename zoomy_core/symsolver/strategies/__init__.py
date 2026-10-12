"""The solver strategies: a template object holds the configuration of one
solver, its methods are the graphs a backend prints (``step``, ``dt``,
``initial_aux``, ``march``, ...).  ``HyperbolicTemplate`` is the explicit
(IMEX) hyperbolic solver, ``ChorinTemplate`` the projection solver built on
it."""
