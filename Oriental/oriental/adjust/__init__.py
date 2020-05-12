# -*- coding: cp1252 -*-
"""
Non-linear, robust least-squares adjustment

oriental.adjust provides
   * a Python-interface to `ceres <http://ceres-solver.org/>`_, see the :doc:`ceres manual <ceres:index>`.
   * predefined observation types, with definitions following ORIENT
   * additional functionality to estimate a posteriori variances, etc.
"""

#__all__ = [ 'parameters', 'cost', 'local_parameterization', 'loss' ]

from oriental import config as _config, _setup_summary_docstring
if _config.debug:
    from ._adjust_d import *
    from ._adjust_d import __date__
else:
    from ._adjust import *
    from ._adjust import __date__

def importSubPackagesAndModules():
    "import all sub-packages into the namespace of this package"
    from oriental import utils
    import os.path
    # This file may be compiled as _cython_init.py -> .c
    # Thus, don't rely on __name__, but better hard-code it.
    utils.importSubPackagesAndModules( [os.path.dirname(__file__)], 'oriental.adjust'  )

from ..utils.strProperties import strProperties
from .. import Exception as _Exception, ExcInternal as _ExcInternal

# enable print(Options)
# Much more easily implemented in Python than in C++!
for _typ in ( Solver.Options, Solver.Summary, IterationSummary, Problem.EvaluateOptions, Covariance.Options ):
    _typ.__str__  = strProperties
    _typ.__repr__ = _typ.__str__

def _printRelevantSolverOptions( self ):
    def printMe(attr):
        print("{}={}".format(attr,getattr(self,attr)))

    warnings = []
    printMe("minimizer_type")

    printMe("max_num_iterations")
    printMe("max_solver_time_in_seconds")
    printMe("function_tolerance")
    printMe("gradient_tolerance")

    printMe("num_threads")

    if self.minimizer_type == MinimizerType.TRUST_REGION:
        printMe("trust_region_strategy_type")
        printMe("use_nonmonotonic_steps")

        if self.use_nonmonotonic_steps:
            printMe("max_consecutive_nonmonotonic_steps")

        printMe("initial_trust_region_radius")
        printMe("max_trust_region_radius")
        printMe("min_trust_region_radius")
        printMe("min_relative_decrease")
        printMe("max_num_consecutive_invalid_steps")

        if self.trust_region_strategy_type == TrustRegionStrategyType.LEVENBERG_MARQUARDT:
            printMe("min_lm_diagonal")
            printMe("max_lm_diagonal")
            printMe("linear_solver_type")
            if self.linear_solver_type in ( LinearSolverType.ITERATIVE_SCHUR,
                                            LinearSolverType.CGNR):
                printMe("preconditioner_type")
                if self.preconditioner_type in ( PreconditionerType.CLUSTER_JACOBI,
                                                 PreconditionerType.CLUSTER_TRIDIAGONAL):
                    printMe("visibility_clustering_type")
            if self.linear_solver_type in ( LinearSolverType.DENSE_QR,
                                            LinearSolverType.DENSE_NORMAL_CHOLESKY,
                                            LinearSolverType.DENSE_SCHUR):
                printMe("dense_linear_algebra_library_type")
            else:
                printMe("sparse_linear_algebra_library_type")
            if self.linear_solver_type in ( LinearSolverType.SPARSE_NORMAL_CHOLESKY,
                                            LinearSolverType.SPARSE_SCHUR):
                printMe("use_postordering")

            if self.linear_solver_type in ( LinearSolverType.ITERATIVE_SCHUR,
                                            LinearSolverType.CGNR):
                printMe("min_linear_solver_iterations")
                printMe("max_linear_solver_iterations")
                printMe("eta")

            printMe("jacobi_scaling")
            printMe("use_inner_iterations")
            if self.use_inner_iterations:
                printMe("inner_iteration_tolerance")
                printMe("inner_iteration_ordering")

        elif self.trust_region_strategy_type == TrustRegionStrategyType.DOGLEG:
            printMe("dogleg_type")

        try:
            printMe("linear_solver_ordering")
        except _Exception:
            pass
        printMe("trust_region_minimizer_iterations_to_dump")
        if len(self.trust_region_minimizer_iterations_to_dump):
            printMe("trust_region_problem_dump_format")
            if self.trust_region_problem_dump_format != DumpFormatType.CONSOLE:
                printMe("trust_region_problem_dump_directory")

    elif self.minimizer_type == MinimizerType.LINE_SEARCH:
        printMe("line_search_direction_type")
        printMe("line_search_type")
        if self.line_search_direction_type in (LineSearchDirectionType.BFGS, LineSearchDirectionType.LBFGS):
            if self.line_search_type != LineSearchType.WOLFE:
                warnings.append( "In order for the assumptions underlying the BFGS and LBFGS line search direction algorithms to be guaranteed to be satisifed, the WOLFE line search should be used." )
            printMe("use_approximate_eigenvalue_bfgs_scaling")
        if self.line_search_direction_type == LineSearchDirectionType.LBFGS:
            printMe("max_lbfs_rank")
        if self.line_search_direction_type == LineSearchDirectionType.NONLINEAR_CONJUGATE_GRADIENT:
            printMe("nonlinear_conjugate_gradient_type")
        printMe("line_search_interpolation_type")
        printMe("min_line_search_step_size")
        printMe("line_search_sufficient_function_decrease")
        printMe("max_line_search_step_contraction")
        printMe("min_line_search_step_contraction")
        printMe("max_num_line_search_step_size_iterations")
        printMe("max_num_line_search_direction_restarts")
        printMe("line_search_sufficient_curvature_decrease")
        printMe("max_line_search_step_expansion")

    else:
        raise _ExcInternal( "MinimizerType not implemented: '{}'".format(self.minimizer_type) )

    printMe("numeric_derivative_relative_step_size")
    printMe("callbacks")
    if len(self.callbacks):
        printMe("update_state_every_iteration")

    printMe("check_gradients")
    if self.check_gradients:
        printMe("gradient_check_relative_precision")

    printMe("logging_type")
    if self.logging_type != LoggingType.SILENT:
        printMe("minimizer_progress_to_stdout")
    printMe("solver_log")

    if len(warnings):
        warnings.insert( 0, "WARNINGS:")
        print( '\n'.join(warnings) )

Solver.Options.printRelevant = _printRelevantSolverOptions

def _summary():
    pass

_setup_summary_docstring( _summary, __name__ )