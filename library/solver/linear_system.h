#include "define.h"
#include <petscksp.h>

#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

int solve_small_linear_system(const realArr2& matrix, const realArr& rhs, realArr& solution)
{
  PetscInt n = matrix.extent(0);
  Vec         sol, b; /* approx solution, RHS, exact solution */
  Mat         A;       /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PC          pc;      /* preconditioner context */
  PetscReal   norm;    /* norm of solution error */
  PetscInt    i, col[n], its;
  PetscScalar value[n];

  /*
     Create vectors
  */
  PetscCall(VecCreate(PETSC_COMM_SELF, &sol));
  PetscCall(PetscObjectSetName((PetscObject)sol, "solution"));
  PetscCall(VecSetSizes(sol, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(sol));
  PetscCall(VecCreate(PETSC_COMM_SELF, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(VecSetSizes(b, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(b));

  /*
     Create matrix.
  */
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
    Set vector 
  */
  for (i = 0; i < n; i++) {
	  PetscCall(VecSetValue(b, i, rhs(i), INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /*
     Assemble matrix
  */
  for (i = 0; i < n; i++) {
  	for (int j = 0; j < n; j++) {
		PetscCall(MatSetValue(A, i, j, matrix(i,j), INSERT_VALUES));
	}
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, b, sol));

//   PetscCall(KSPGetIterationNumber(ksp, &its));
//   PetscCall(PetscPrintf(PETSC_COMM_SELF, "Iterations %" PetscInt_FMT "\n", its));

//   /* check that KSP automatically handles the fact that the the new non-zero values in the matrix are propagated to the KSP solver */
//   PetscCall(MatShift(A, 2.0));
//   PetscCall(KSPSolve(ksp, b, x));

  /*
	Copy the values back to the solution array
  */

  for (i = 0; i < n; i++) {
	  VecGetValues(sol, 1, &i, &solution[i]);
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&sol));
  PetscCall(MatDestroy(&A));
  return 0;
}

void test_solve_small_linear_system()
{
	realArr2 A("Q", 2, 2);
	realArr b("b", 2);
	realArr x("x", 2);
	A(0, 0) = 2.0;
	A(0, 1) = -2.0;
	A(1, 0) = -2.0;
	A(1, 1) = 2.0;
	b(0) = 1.0;
	b(1) = 1.0;
	int err = solve_small_linear_system(A, b, x);
	std::cout << "x: " << x(0) << " " << x(1) << std::endl;
}

#endif // LINEAR_SYSTEM_H