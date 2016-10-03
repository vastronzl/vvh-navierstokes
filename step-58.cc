/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *d
 * Author: Liang Zhao and Timo Heister, Clemson University, 2016
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/block_matrix_array.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparse_ilu.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace Step58
{
  using namespace dealii;

  // TODO: remove this later, when moving to deal 8.4
  Tensor<1,3> cross_product_3d(const Tensor<1,3> &A, const Tensor<1,3> &B)
  {
  	Tensor<1,3> result;
  	cross_product(result, A, B);
  	return result;
  }



  template <int dim>
  class Navier_Stokes_Newton
  {
  public:
    Navier_Stokes_Newton(const unsigned int degree);
    ~Navier_Stokes_Newton();
    void run();

  private:
    void setup_system(bool setup_dof, bool initialize_system);
    void setup_dof();
    void initialize_system();
    void assemble_NavierStokes_system(bool initial_step,
                                      bool assemble_matrix,
                                      bool assemble_rhs);
    void solve(bool initial_step);
    void refine_mesh();
    void process_solution(unsigned int refinement);
    void output_results (const unsigned int refinement_cycle) const;
    void newton_iteration(const double tolerance,
                          const unsigned int max_iteration,
                          const unsigned int n_refinements,
                          bool initial,
                          bool result);
    void search_initial_guess(double step_size);

    double viscosity;
    double gamma;
    double gamma1;
    double gamma2;
    double gamma3;
    const unsigned int           degree;
    int dof_u;
    int dof_p;
    int dof_w;


    Triangulation<dim>           triangulation;
    FESystem<dim>                fe;
    DoFHandler<dim>              dof_handler;

    ConstraintMatrix             zero_constraints;
    ConstraintMatrix             nonzero_constraints;

    BlockSparsityPattern         sparsity_pattern;
    BlockSparseMatrix<double>    system_matrix;
    SparseMatrix<double>         pressure_mass_matrix;

    BlockVector<double>          present_solution;
    BlockVector<double>          newton_update;
    BlockVector<double>          system_rhs;
    BlockVector<double>          evaluation_point;
  };

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(double viscosity) : Function<dim>(dim+dim+1), viscosity(viscosity) {}
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const;

    virtual void   vector_value(const Point <dim>    &p,
                                Vector<double> &values) const;
  private:
    double viscosity;
  };

  template <int dim>
  double ExactSolution<dim>::value(const Point<dim> &p,
                                   const unsigned int component) const
  {
    using numbers::PI;
//    const double Px = p[0];
//    const double Py = p[1];
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

// velocity
    if (component == 0)
      {
    	return 2.0*sin(PI*x);
      }
    if (component == 1)
      {
    	return -PI*y*cos(PI*x);
      }
    if (component == 2)
      {
    	return -PI*z*cos(PI * x);
      }
// vorticity
    if (component == 3)
      {
        return 0.0;
      }
    if (component == 4)
      {
    	return -PI*PI*z*sin(PI*x);
      }
    if (component == 5)
      {
    	return PI*PI*y*sin(PI*x);
      }
// pressure
    if (component == 6)
      {
    	return sin(PI*x)*cos(PI*y)*sin(PI*z);
      }
    return 0;
  }

  template <int dim>
  void ExactSolution<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = ExactSolution<dim>::value (p, c);
  }


  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide (double viscosity) : Function<dim>(dim+dim+1), viscosity (viscosity) {}
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const;
    virtual void   vector_value(const Point <dim>    &p,
                                Vector<double> &values) const;
  private:
    double viscosity;
  };

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &p,
                                   const unsigned int component) const
  {
    using numbers::PI;
//    const double Px = p[0];
//    const double Py = p[1];
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

    if (component == 0)
      {
    	return  2.0*viscosity*PI*PI*sin(PI*x)
    	      + 4*PI*sin(PI*x)*cos(PI*x)
    	      + PI * cos(PI * x) * cos(PI * y) * sin(PI * z);

      }
    if (component == 1)
      {
    	return - viscosity*PI*PI*PI*y*cos(PI*x)
    	       + PI*PI*y*sin(PI*x)*sin(PI*x) + PI*PI*y
    	       - PI*sin(PI*x)*sin(PI*y)*sin(PI*z);

      }
    if (component == 2)
      {
    	return -viscosity*PI*PI*PI*z*cos(PI*x)
    	       + PI*PI*z*sin(PI*x)*sin(PI*x) + PI*PI*z
    	       + PI*sin(PI*x)*cos(PI*y)*cos(PI*z);
      }

    if (component == 3)
      {
    	return 0.0;
      }
    if (component == 4)
      {
    	return -viscosity*PI*PI*PI*PI*z*sin(PI*x)
    	       -2.0*PI*PI*PI*z*sin(PI*x)*cos(PI*x);
      }
    if (component == 5)
      {
    	return  viscosity*PI*PI*PI*PI*y*sin(PI*x)
    	       +2*PI*PI*PI*y*sin(PI*x)*cos(PI*x);
      }
    if (component == 6)
      {
        return 0.0;
      }
    return 0;
  }

  template <int dim>
  void RightHandSide<dim>::vector_value(const Point <dim>    &p,
                                        Vector<double> &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = RightHandSide<dim>::value (p, c);
  }



  template <class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner (double                                     gamma,
                              double                                     viscosity,
                              const BlockSparseMatrix<double>            &S,
                              const SparseMatrix<double>                 &P,
                              const PreconditionerMp                     &Mppreconditioner
                             );

    void vmult (BlockVector<double>       &dst,
                const BlockVector<double> &src) const;

  private:
    const double gamma;
    const double viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double>      &pressure_mass_matrix;
    const PreconditionerMp          &mp_preconditioner;
  };

  template <class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerMp>::
  BlockSchurPreconditioner (double                           gamma,
                            double                           viscosity,
                            const BlockSparseMatrix<double>  &S,
                            const SparseMatrix<double>       &P,
                            const PreconditionerMp           &Mppreconditioner)
    :
    gamma                (gamma),
    viscosity            (viscosity),
    stokes_matrix        (S),
    pressure_mass_matrix (P),
    mp_preconditioner    (Mppreconditioner)
  {}

  template <class PreconditionerMp>
  void
  BlockSchurPreconditioner<PreconditionerMp>::
  vmult (BlockVector<double>       &dst,
         const BlockVector<double> &src) const
  {
	  SparseDirectUMFPACK  A_alpha_direct;
	  SparseDirectUMFPACK  A_beta_direct;
	  A_alpha_direct.initialize(stokes_matrix.block(0,0));
	  A_beta_direct.initialize(stokes_matrix.block(2,2));

	  {
		  A_beta_direct.vmult (dst.block(2), src.block(2));
	  }

      Vector<double> utmp1(src.block(0));
      {
    	  SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
    	  SolverCG<>    cg (solver_control);

    	  dst.block(1) = 0.0;
    	  cg.solve(pressure_mass_matrix,
    			   dst.block(1), src.block(1),
                   mp_preconditioner);
    	  dst.block(1) *= -(viscosity+gamma);

    	  stokes_matrix.block(0,1).vmult(utmp1, dst.block(1));
    	  utmp1*=-1.0;
    	  utmp1+=src.block(0);
      }

      Vector<double> utmp2(src.block(2));
      Vector<double> utmp3(src.block(2));
      Vector<double> utmp4(src.block(1));
      Vector<double> utmp5(src.block(1));

      {
    	  stokes_matrix.block(2,2).vmult(utmp2, dst.block(2));
    	  A_alpha_direct.vmult(utmp3, utmp2);
    	  stokes_matrix.block(1,0).vmult(utmp4, utmp3);
    	  {
        	  SolverControl solver_control(1000, 1e-6 * utmp4.l2_norm());
        	  SolverCG<>    cg (solver_control);
    		  utmp5 = 0.0;
        	  cg.solve(pressure_mass_matrix,
        			   utmp5, utmp4,
                       mp_preconditioner);
        	  utmp5*= -(viscosity+gamma);
    	  }
    	  dst.block(1)+= utmp5;
      }

      Vector<double> utmp6(src.block(0));
      Vector<double> utmp7(src.block(0));
      Vector<double> utmp8(src.block(1));
      Vector<double> utmp9(src.block(1));
      Vector<double> utmp10(src.block(0));
      {
    	  stokes_matrix.block(2,2).vmult(utmp6, dst.block(2));
    	  A_alpha_direct.vmult(utmp7, utmp6);
    	  stokes_matrix.block(1,0).vmult(utmp8, utmp7);
    	  {
        	  SolverControl solver_control(1000, 1e-6 * utmp8.l2_norm());
        	  SolverCG<>    cg (solver_control);
    		  utmp9 = 0.0;
        	  cg.solve(pressure_mass_matrix,
        			   utmp9, utmp8,
                       mp_preconditioner);
        	  utmp9*= -(viscosity+gamma);
    	  }
    	  stokes_matrix.block(0,1).vmult(utmp10, utmp9);
    	  utmp10 *= -1.0;
    	  utmp6  *= -1.0;
    	  utmp10 += utmp6;
    	  utmp1  += utmp10;
      }

      A_alpha_direct.vmult(dst.block(0), utmp1);
  }

  template <class PreconditionerMp>
  class BlockSchurPreconditioner1 : public Subscriptor
  {
  public:
    BlockSchurPreconditioner1 (double                                     gamma,
                              double                                     viscosity,
                              const BlockSparseMatrix<double>            &S,
                              const SparseMatrix<double>                 &P,
                              const PreconditionerMp                     &Mppreconditioner
                             );

    void vmult (BlockVector<double>       &dst,
                const BlockVector<double> &src) const;

  private:
    const double gamma;
    const double viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double>      &pressure_mass_matrix;
    const PreconditionerMp          &mp_preconditioner;
  };

  template <class PreconditionerMp>
  BlockSchurPreconditioner1<PreconditionerMp>::
  BlockSchurPreconditioner1 (double                           gamma,
                            double                           viscosity,
                            const BlockSparseMatrix<double>  &S,
                            const SparseMatrix<double>       &P,
                            const PreconditionerMp           &Mppreconditioner)
    :
    gamma                (gamma),
    viscosity            (viscosity),
    stokes_matrix        (S),
    pressure_mass_matrix (P),
    mp_preconditioner    (Mppreconditioner)
  {}
  template <class PreconditionerMp>
  void
  BlockSchurPreconditioner1<PreconditionerMp>::
  vmult (BlockVector<double>       &dst,
         const BlockVector<double> &src) const
  {
	SparseDirectUMFPACK  A_beta_direct;
	A_beta_direct.initialize(stokes_matrix.block(2,2));
	A_beta_direct.vmult (dst.block(2), src.block(2));

    Vector<double> utmp(src.block(0));
    {
      SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
      SolverCG<>    cg (solver_control);
      dst.block(1) = 0.0;
      cg.solve(pressure_mass_matrix,
               dst.block(1), src.block(1),
               mp_preconditioner);
      dst.block(1) *= -(viscosity+gamma);
    }

    {
      stokes_matrix.block(0,1).vmult(utmp, dst.block(1));
      utmp*=-1.0;
      utmp+=src.block(0);
    }

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(stokes_matrix.block(0,0));
    A_direct.vmult (dst.block(0), utmp);
  }

  //**************************************************************************//
  //**************************************************************************//
  //**************************************************************************//
  template <class MatrixType>
  class InverseMatrix : public Subscriptor
  {
  public:
	  InverseMatrix (const MatrixType   &m);
  	  void vmult(Vector<double>   &dst,
  			     const Vector<double> &src) const;
  private:
	  const SmartPointer<const MatrixType> matrix;
	  SparseDirectUMFPACK  M_direct;
  };

  template <class MatrixType>
  InverseMatrix<MatrixType>::InverseMatrix(const MatrixType &m)
  :
  matrix(&m)
  {
	  M_direct.initialize(*matrix);
  }

  template<class MatrixType>
  void InverseMatrix<MatrixType>::vmult(Vector<double> &dst,
		                                            const Vector<double> &src) const
  {
	  M_direct.vmult(dst, src);
  }

  //**************************************************************************//
  //**************************************************************************//
  //**************************************************************************//
  template<class PreconditionerA, class PreconditionerB>
  class BlockDiagonalPreconditioner : public Subscriptor
  {
  public:
	  BlockDiagonalPreconditioner(const PreconditionerA  &preconditioner_A_alpha,
			                      const PreconditionerB  &preconditioner_A_beta);

	    void vmult (BlockVector<double>       &dst,
	                const BlockVector<double> &src) const;
  private:
	    const PreconditionerA   &preconditioner_A_alpha;
	    const PreconditionerB   &preconditioner_A_beta;
  };

  template <class PreconditionerA, class PreconditionerB>
  BlockDiagonalPreconditioner<PreconditionerA, PreconditionerB>::
  BlockDiagonalPreconditioner (
    const PreconditionerA  &preconditioner_A_alpha,
    const PreconditionerB  &preconditioner_A_beta)
    :
	preconditioner_A_alpha (preconditioner_A_alpha),
    preconditioner_A_beta  (preconditioner_A_beta)
  {}

  template <class PreconditionerA, class PreconditionerB>
  void
  BlockDiagonalPreconditioner<PreconditionerA, PreconditionerB>::
  vmult (BlockVector<double>       &dst,
         const BlockVector<double> &src) const
	{
	  preconditioner_A_alpha.vmult(dst.block(0), src.block(0));
	  preconditioner_A_beta.vmult(dst.block(1), src.block(1));
	}

  //**************************************************************************//
  //**************************************************************************//
  //**************************************************************************//
  template <class Preconditioner, class MatrixType>
  class BlockTriangularPreconditioner : public Subscriptor
  {
  public:
	  BlockTriangularPreconditioner(const MatrixType &W,
			  	  	  	  	  	  	const Preconditioner &A_Alpha_Inverse,
									const Preconditioner &A_Beta_Inverse);

	  void vmult (BlockVector<double>       &dst,
			      const BlockVector<double> &src) const;
  private:
	  const MatrixType  &W;
	  const Preconditioner  &A_alpha_inverse;
	  const Preconditioner  &A_beta_inverse;
  };

  template <class Preconditioner, class MatrixType>
  BlockTriangularPreconditioner<Preconditioner, MatrixType>::
  BlockTriangularPreconditioner(const MatrixType &W,
  			  	  	  	  	  	  	const Preconditioner &A_Alpha_Inverse,
  									const Preconditioner &A_Beta_Inverse)
  :
  									W(W),
									A_alpha_inverse(A_Alpha_Inverse),
									A_beta_inverse(A_Beta_Inverse)
									{}

  template <class Preconditioner, class MatrixType>
  void
  BlockTriangularPreconditioner<Preconditioner, MatrixType>::
  vmult (BlockVector<double>       &dst,
         const BlockVector<double> &src) const
		 {
	  	      A_alpha_inverse.vmult(dst.block(0), src.block(0));
	  	  	  A_beta_inverse.vmult(dst.block(1), src.block(1));
	  	  	  Vector<double> tmp1(dst.block(1));
	  	  	  Vector<double> tmp2(dst.block(1));
	  	  	  W.vmult(tmp1, dst.block(1));
	  	  	  A_alpha_inverse.vmult(tmp2, tmp1);
	  	  	  tmp2 *= -1;
	  	  	  dst.block(0) += tmp2;
		 }



  //**************************************************************************//
  //**************************************************************************//
  //**************************************************************************//
  template <class PreconditionerMp, class PreconditionerK>
  class BlockSchurPreconditioner2 : public Subscriptor
  {
  public:
    BlockSchurPreconditioner2 (double                                     gamma,
                               double                                     viscosity,
                               const BlockSparseMatrix<double>            &S,
                               const SparseMatrix<double>                 &P,
                               const PreconditionerMp                     &Mppreconditioner,
							   const PreconditionerK                      &Kpreconditioner
                              );

    void vmult (BlockVector<double>       &dst,
                const BlockVector<double> &src) const;

  private:
    const double gamma;
    const double viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double>      &pressure_mass_matrix;
    const PreconditionerMp          &mp_preconditioner;
    const PreconditionerK           &K_preconditioner;
  };

  template <class PreconditionerMp, class PreconditionerK>
  BlockSchurPreconditioner2<PreconditionerMp, PreconditionerK>::
  BlockSchurPreconditioner2 (double                           gamma,
                             double                           viscosity,
                             const BlockSparseMatrix<double>  &S,
                             const SparseMatrix<double>       &P,
                             const PreconditionerMp           &Mppreconditioner,
							 const PreconditionerK            &Kpreconditioner)
    :
    gamma                (gamma),
    viscosity            (viscosity),
    stokes_matrix        (S),
    pressure_mass_matrix (P),
    mp_preconditioner    (Mppreconditioner),
	K_preconditioner     (Kpreconditioner)
  {}
  template <class PreconditionerMp, class PreconditionerK>
  void
  BlockSchurPreconditioner2<PreconditionerMp, PreconditionerK>::
  vmult (BlockVector<double>       &dst,
         const BlockVector<double> &src) const
  {
	  {
	     SolverControl solver_control(1000, 1e-6 * src.block(2).l2_norm());
	     SolverCG<>    cg (solver_control);

	     dst.block(2) = 0.0;
	     cg.solve(pressure_mass_matrix,
	              dst.block(2), src.block(2),
	              mp_preconditioner);
	     dst.block(2) *= -(viscosity+gamma);
	  }

	  Vector<double> utmp(src.block(0));
	  {
	     stokes_matrix.block(0,2).vmult(utmp, dst.block(2));
	     utmp*=-1.0;
	     utmp+=src.block(0);
	  }

	  BlockMatrixArray<double> K(2,2);
	  K.enter(stokes_matrix.block(0,0), 0, 0);
	  K.enter(stokes_matrix.block(0,1), 0, 1);
	  K.enter(stokes_matrix.block(1,0), 1, 0);
	  K.enter(stokes_matrix.block(1,1), 1, 1);

	  BlockVector<double> uwtmp(2);
	  uwtmp.block(0) = utmp;
	  uwtmp.block(1) = src.block(1);

	  BlockVector<double> dst_tmp(2);
	  dst_tmp = uwtmp;

	  {
		  SolverControl solver_control (1000,1e-6 * uwtmp.l2_norm());
		  SolverFGMRES<BlockVector<double> >::AdditionalData gmres_data;
		  SolverFGMRES<BlockVector<double> > gmres(solver_control,gmres_data);

		  dst_tmp = 0;
		  gmres.solve (K,
		               dst_tmp,
		               uwtmp,
					   K_preconditioner);

		  dst.block(0) = dst_tmp.block(0);
		  dst.block(1) = dst_tmp.block(1);
		  std::cout << "  ---inner iteration:  " << solver_control.last_step() << std::endl;
	  }
  }
  //**************************************************************************//
  //**************************************************************************//
  //**************************************************************************//


  template <int dim>
  Navier_Stokes_Newton<dim>::Navier_Stokes_Newton(const unsigned int degree)
    :

    viscosity(1.0/30.0),
    gamma(1.0),
	gamma1(0.0),
	gamma2(0.0),
	gamma3(1.0),
    degree(degree),
    triangulation(Triangulation<dim>::maximum_smoothing),
    fe(FE_Q<dim>(degree+1), dim,
       FE_Q<dim>(degree+1), dim,
       FE_Q<dim>(degree),   1),
    dof_handler(triangulation)
  {}


  template <int dim>
  Navier_Stokes_Newton<dim>::~Navier_Stokes_Newton()
  {
    dof_handler.clear();
  }


  template <int dim>
  void Navier_Stokes_Newton<dim>::setup_dof()
  {
    system_matrix.clear();
    pressure_mass_matrix.clear();

    dof_handler.distribute_dofs (fe);

    std::vector<unsigned int> component_u(dim, 0);
    std::vector<unsigned int> component_w(dim, 1);
    std::vector<unsigned int> component_p(1,   2);
    std::vector<unsigned int> block_component;

    block_component.insert(block_component.end(), component_u.begin(), component_u.end());
    block_component.insert(block_component.end(), component_w.begin(), component_w.end());
    block_component.insert(block_component.end(), component_p.begin(), component_p.end());

    DoFRenumbering::component_wise (dof_handler, block_component);

    FEValuesExtractors::Vector velocities(0);
    FEValuesExtractors::Vector vorticity(dim);

    {
      nonzero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);

      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ExactSolution<dim>(viscosity),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ExactSolution<dim>(viscosity),
											   nonzero_constraints,
                                               fe.component_mask(vorticity));
    }
    nonzero_constraints.close();

    {
      zero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);

      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ZeroFunction<dim>(dim+dim+1),
                                               zero_constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ZeroFunction<dim>(dim+dim+1),
                                               zero_constraints,
                                               fe.component_mask(vorticity));
    }
    zero_constraints.close();

    std::vector<types::global_dof_index> dofs_per_block (3);
    DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
    dof_u = dofs_per_block[0];
    dof_w = dofs_per_block[1];
    dof_p = dofs_per_block[2];

    std::cout <<"*****************************************" << std::endl
    		  <<"       DoF Information        " << std::endl
			  <<"*****************************************"
			  << std::endl;
    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << dof_u << '+' << dof_w << '+' << dof_p << ')' << std::endl
			  << "*****************************************"
              << std::endl;

    std::cout << "  gamma = " << gamma << " ; "
    		  << "  gamma1 = "<< gamma1<< " ; "
			  << "  gamma2 = "<< gamma2<< " ; "
			  << "  gamma3 = "<< gamma3<< std::endl
			  << "*****************************************" << std::endl;

    BlockSparseMatrix<double> J;
  }

  template <int dim>
  void Navier_Stokes_Newton<dim>::initialize_system()
  {
    {
      BlockDynamicSparsityPattern dsp (3,3);
      dsp.block(0,0).reinit (dof_u, dof_u);
      dsp.block(1,0).reinit (dof_w, dof_u);
      dsp.block(2,0).reinit (dof_p, dof_u);

      dsp.block(0,1).reinit (dof_u, dof_w);
      dsp.block(1,1).reinit (dof_w, dof_w);
      dsp.block(2,1).reinit (dof_p, dof_w);

      dsp.block(0,2).reinit (dof_u, dof_p);
      dsp.block(1,2).reinit (dof_w, dof_p);
      dsp.block(2,2).reinit (dof_p, dof_p);

      dsp.collect_sizes();

      DoFTools::make_sparsity_pattern (dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from (dsp);
    }

    system_matrix.reinit(sparsity_pattern);

    present_solution.reinit (3);
    present_solution.block(0).reinit (dof_u);
    present_solution.block(1).reinit (dof_w);
    present_solution.block(2).reinit (dof_p);
    present_solution.collect_sizes ();

    newton_update.reinit (3);
    newton_update.block(0).reinit (dof_u);
    newton_update.block(1).reinit (dof_w);
    newton_update.block(2).reinit (dof_p);
    newton_update.collect_sizes ();

    system_rhs.reinit (3);
    system_rhs.block(0).reinit (dof_u);
    system_rhs.block(1).reinit (dof_w);
    system_rhs.block(2).reinit (dof_p);
    system_rhs.collect_sizes ();

    std::cout <<"*****************************************" << std::endl
    		  <<"      Initialize system complete        " << std::endl
			  <<"*****************************************"
			  << std::endl;
  }

  template <int dim>
  void Navier_Stokes_Newton<dim>::assemble_NavierStokes_system(bool initial_step,
      bool assemble_matrix,
      bool assemble_rhs)
  {
    if (assemble_matrix)
      {
        system_matrix    = 0;
      }

    if (assemble_rhs)
      {
        system_rhs       = 0;
      }



    QGauss<dim>   quadrature_formula(degree+2);

    FEValues<dim> fe_values (fe,
                             quadrature_formula,
                             update_values |
                             update_quadrature_points |
                             update_JxW_values |
                             update_gradients );

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Vector vorticity (dim);
    const FEValuesExtractors::Scalar pressure (dim+dim);


    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs    (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


    std::vector<Tensor<1, dim>>   present_velocity_values     (n_q_points);
    std::vector<Tensor<1, dim>>   present_velocity_curls      (n_q_points);
    std::vector<Tensor<2, dim>>   present_velocity_gradients  (n_q_points);
    std::vector<double>           present_velocity_divergence (n_q_points);

    std::vector<double>           present_pressure_values     (n_q_points);

    std::vector<Tensor<1, dim>>   present_vorticity_values    (n_q_points);
    std::vector<Tensor<2, dim>>   present_vorticity_gradients (n_q_points);
    std::vector<double>           present_vorticity_divergence (n_q_points);

    const RightHandSide<dim>      right_hand_side (viscosity);
    std::vector<Vector<double>>   rhs_values(n_q_points, Vector<double>(dim+dim+1));



    std::vector<double>           div_phi_u                 (dofs_per_cell);
    std::vector<Tensor<1, dim>>   phi_u                     (dofs_per_cell);
    std::vector<Tensor<1, dim>>   curl_phi_u                (dofs_per_cell);
    std::vector<Tensor<2, dim>>   grad_phi_u                (dofs_per_cell);

    std::vector<double>           phi_p                     (dofs_per_cell);

    std::vector<double>           div_phi_w                 (dofs_per_cell);
    std::vector<Tensor<1, dim>>   phi_w                     (dofs_per_cell);
    std::vector<Tensor<2, dim>>   grad_phi_w                (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        fe_values[velocities].get_function_values(evaluation_point,
                                                  present_velocity_values);

        fe_values[velocities].get_function_gradients(evaluation_point,
                                                     present_velocity_gradients);

        fe_values[velocities].get_function_divergences(evaluation_point,
                                                       present_velocity_divergence);

        fe_values[velocities].get_function_curls(evaluation_point,
                                                 present_velocity_curls);

        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);

        fe_values[vorticity].get_function_values(evaluation_point,
                                                 present_vorticity_values);

        fe_values[vorticity].get_function_gradients(evaluation_point,
                                                    present_vorticity_gradients);

        fe_values[vorticity].get_function_divergences(evaluation_point,
                                                      present_vorticity_divergence);

        for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                div_phi_u[k]  =  fe_values[velocities].divergence (k, q);
                grad_phi_u[k] =  fe_values[velocities].gradient(k, q);
                phi_u[k]      =  fe_values[velocities].value(k, q);
                curl_phi_u[k] =  fe_values[velocities].curl(k, q);

                phi_p[k]      =  fe_values[pressure]  .value(k, q);

                div_phi_w[k]  =  fe_values[vorticity].divergence (k, q);
                phi_w[k]      =  fe_values[vorticity].value(k, q);
                grad_phi_w[k] =  fe_values[vorticity].gradient(k, q);
              }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                if (assemble_matrix)
                  {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      {
                        local_matrix(i, j) += (
                        						   viscosity*scalar_product(grad_phi_u[j], grad_phi_u[i])    //nu(grad_d_U, grad_V_u)
                                                 + present_velocity_gradients[q]*phi_u[j]*phi_u[i]           //(d_U*grad_U_old, V_u)
                                                 + grad_phi_u[j]*present_velocity_values[q]*phi_u[i]         //(U_old*grad_d_U, V_u)
                                                 - phi_p[j]*div_phi_u[i]                                     //-(d_P, div_V_u)
                                                 + 0.5*cross_product_3d(present_vorticity_values[q], phi_u[j])*phi_u[i]
                                                 + 0.5*cross_product_3d(phi_w[j], present_velocity_values[q])*phi_u[i]  //DW
                                                 - 0.5*cross_product_3d(curl_phi_u[j], present_velocity_values[q])*phi_u[i]
                                                 - 0.5*cross_product_3d(present_velocity_curls[q], phi_u[j])*phi_u[i]

												 - div_phi_u[j]*phi_p[i]                            		 //-(div_d_U, V_p)

                                                 + viscosity*scalar_product(grad_phi_w[j], grad_phi_w[i])   //nu(grad_d_W, grad_V_w)
                                                 + grad_phi_w[j]*present_velocity_values[q]*phi_w[i]        //(U_old*grad_d_W, V_w)
                                                 + present_vorticity_gradients[q]*phi_u[j]*phi_w[i]         //(d_U*grad_W_old, V_w) DN
                                                 - present_velocity_gradients[q]*phi_w[j]*phi_w[i]          //(d_W*grad_U_old, V_w)
                                                 - grad_phi_u[j]*present_vorticity_values[q]*phi_w[i]       //(W_old*grad_d_U, V_w) DN

                                                 + gamma*div_phi_u[j]*div_phi_u[i]               //(div_u_j, div_u_i) on A_alpha
												 + gamma1*div_phi_u[j]*div_phi_w[i]              //(div_u_j, div_w_i) on N
												 + gamma2*div_phi_w[j]*div_phi_u[i]              //(div_w_j, div_u_i) on W
									             + gamma3*div_phi_w[j]*div_phi_w[i]              //(div_w_j, div_w_i) on A_beta

                                                 + phi_p[i]*phi_p[j]                             //Mp-mass matrix of pressure
											  )* fe_values.JxW(q);
                      }
                  }

                if (assemble_rhs)
                  {
                	const unsigned int component_i = fe.system_to_component_index(i).first;
                    local_rhs(i) += (   fe_values.shape_value(i,q) *rhs_values[q](component_i)
                                       -viscosity*scalar_product(present_velocity_gradients[q],grad_phi_u[i]) // -nu(gradU_old, gradV_u)
                                       -present_velocity_gradients[q]*present_velocity_values[q]*phi_u[i]     // -(U_old*gradU_old, V_u)
                                       +present_pressure_values[q]*div_phi_u[i]   // +(P_old, div_V_u)
                                       +present_velocity_divergence[q]*phi_p[i]   // +(div_U_oid, V_p)
                                       +0.5*cross_product_3d(present_velocity_curls[q], present_velocity_values[q])*phi_u[i]
                                       -0.5*cross_product_3d(present_vorticity_values[q], present_velocity_values[q])*phi_u[i]
                                       -viscosity*scalar_product(present_vorticity_gradients[q],grad_phi_w[i]) //-nu(grad_W_old, grad_V_w)
                                       -present_vorticity_gradients[q]*present_velocity_values[q]*phi_w[i]     //-(U_old*grad_W_old, V_w)
                                       +present_velocity_gradients[q]*present_vorticity_values[q]*phi_w[i]     //+(W_old*grad_U_old, V_w)
                                       -gamma*present_velocity_divergence[q]*div_phi_u[i]
									   -gamma1*present_velocity_divergence[q]*div_phi_w[i]
									   -gamma2*present_vorticity_divergence[q]*div_phi_u[i]
									   -gamma3*present_vorticity_divergence[q]*div_phi_w[i]
                                    )*fe_values.JxW(q);
                  }

              }
          }

        cell-> get_dof_indices (local_dof_indices);

        if (initial_step)
          {
            if (assemble_matrix)
              {
                nonzero_constraints.distribute_local_to_global(local_matrix,
                                                               local_dof_indices,
                                                               system_matrix);
              }

            if (assemble_rhs)
              {
                nonzero_constraints.distribute_local_to_global(local_rhs,
                                                               local_dof_indices,
                                                               system_rhs);
              }
          }

        else
          {
            if (assemble_matrix)
              {
                zero_constraints.distribute_local_to_global(local_matrix,
                                                              local_dof_indices,
                                                              system_matrix);
              }

            if (assemble_rhs)
              {
                zero_constraints.distribute_local_to_global(local_rhs,
                                                              local_dof_indices,
                                                              system_rhs);;
              }
          }

      }

    if (assemble_matrix)
      {
        pressure_mass_matrix.reinit(sparsity_pattern.block(2,2));
        pressure_mass_matrix.copy_from(system_matrix.block(2,2));
        system_matrix.block(2,2) = 0;
      }

  }

  template <int dim>
  void Navier_Stokes_Newton<dim>::solve (bool initial_step)
  {
	  if (false)
	  {
		  SparseDirectUMFPACK  G_direct;
		  G_direct.initialize(system_matrix);
		  G_direct.vmult (newton_update, system_rhs);
	  }

	    SolverControl solver_control (system_matrix.m(),1e-4*system_rhs.l2_norm(), true);

	    SolverFGMRES<BlockVector<double> >::AdditionalData gmres_data;

	    SolverFGMRES<BlockVector<double> > gmres(solver_control,gmres_data);

	    SparseILU<double> pmass_preconditioner;
	    pmass_preconditioner.initialize (pressure_mass_matrix,
	                                     SparseILU<double>::AdditionalData());

    const InverseMatrix< SparseMatrix<double> > A_alpha_inverse(system_matrix.block(0,0));
    const InverseMatrix< SparseMatrix<double> > A_beta_inverse (system_matrix.block(1,1));


//    const BlockDiagonalPreconditioner< InverseMatrix< SparseMatrix<double> >,
//									   InverseMatrix< SparseMatrix<double> > >
//    													K_preconditioner(A_alpha_inverse, A_beta_inverse);
//    const BlockSchurPreconditioner2< SparseILU<double>,
//									 BlockDiagonalPreconditioner<InverseMatrix< SparseMatrix<double> >,
//																 InverseMatrix< SparseMatrix<double> > > >
//    												   preconditioner (gamma,
//                                                    		   	   	   viscosity,
//    																   system_matrix,
//    																   pressure_mass_matrix,
//                													   pmass_preconditioner,
//																	   K_preconditioner);


    const BlockTriangularPreconditioner< InverseMatrix< SparseMatrix<double> >, SparseMatrix<double> >
                                                        K_preconditioner1(system_matrix.block(0,1),
                                                        		          A_alpha_inverse,
																		  A_beta_inverse);

        const BlockSchurPreconditioner2< SparseILU<double>,
		                                 BlockTriangularPreconditioner< InverseMatrix< SparseMatrix<double> >,
										                                SparseMatrix<double> > >
        												   preconditioner (gamma,
                                                        		   	   	   viscosity,
        																   system_matrix,
        																   pressure_mass_matrix,
                    													   pmass_preconditioner,
    																	   K_preconditioner1);

        newton_update = 0;
        gmres.solve (system_matrix,
                     newton_update,
                     system_rhs,
                     preconditioner);

//
//    const BlockSchurPreconditioner<SparseILU<double>>
//                                                   preconditioner (gamma,
//                                                		   	   	   viscosity,
//																   system_matrix,
//																   pressure_mass_matrix,
//            													   pmass_preconditioner);
//    std::cout << "  Computing linear system -----" << std::endl;

//    gmres.solve (system_matrix,
//                 newton_update,
//                 system_rhs,
//                 preconditioner);



    std::cout << " ****FGMRES steps: " << solver_control.last_step() << std::endl;

    if (initial_step)
      {
        nonzero_constraints.distribute(newton_update);
      }

    else
      {
        zero_constraints.distribute(newton_update);
      }
  }

  template <int dim>
  void Navier_Stokes_Newton<dim>::newton_iteration(const double tolerance,
                                                   const unsigned int max_iteration,
                                                   const unsigned int n_refinements,
                                                   bool  is_initial_step,
                                                   bool  output_result)
  {
    double current_res;
    double last_res;
    bool   first_step = is_initial_step;

    for (unsigned int refinement = 0; refinement < n_refinements+1; ++refinement)
      {
        unsigned int outer_iteration = 0;
        last_res = 1.0;
        current_res = 1.0;
        std::cout << "*****************************************" << std::endl;
        std::cout << "************  refinement = " << refinement << " ************ " << std::endl;
        std::cout << "viscosity= " << viscosity << std::endl;
        std::cout << "*****************************************" << std::endl;

        while ((first_step || (current_res > tolerance)) && outer_iteration < max_iteration)
          {
            if (first_step)
              {
                setup_dof();
                initialize_system();
                evaluation_point = present_solution;
                assemble_NavierStokes_system(first_step, true, true);
                solve(first_step);
                present_solution = newton_update;
                nonzero_constraints.distribute(present_solution);
                first_step = false;
                evaluation_point = present_solution;
                assemble_NavierStokes_system(first_step, false, true);
                current_res = system_rhs.l2_norm();
                std::cout << "******************************" << std::endl;
                std::cout << " The residual of initial guess is " << current_res << std::endl;
                std::cout << " Initialization complete!  " << std::endl;
                std::cout << "---------------------------------------- " << std::endl;
                last_res = current_res;
              }

            else
            {
              evaluation_point = present_solution;
              if (outer_iteration == 0)
                {
                  assemble_NavierStokes_system(first_step, true, true);
                }
              else
                {
                  assemble_NavierStokes_system(first_step, true, false);
                }
              solve(first_step);
              double alpha;
              for (alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
                {
                  evaluation_point = present_solution;
                  evaluation_point.add(alpha, newton_update);
                  assemble_NavierStokes_system(first_step, false, true);
                  current_res = system_rhs.l2_norm();
                  std::cout << " alpha = " << std::setw(6) << alpha << std::setw(0)
                            << " res = " << current_res << std::endl;
                  if (current_res < last_res)
                    break;
                }

              {
                present_solution = evaluation_point;
                nonzero_constraints.distribute(present_solution);
                std::cout << " ----The " << outer_iteration << "th iteration. ---- " << std::endl;
                std::cout << " ----Residual: " << current_res << std::endl;
                std::cout << "---------------------------------------- " << std::endl;
                last_res = current_res;
              }

            }
            ++outer_iteration;

            if (output_result)
              {
                output_results (max_iteration*refinement+outer_iteration);

                if (current_res <= tolerance)
                  {
                    process_solution(refinement);
                  }
              }
          }

        if (refinement < n_refinements)
          {
            refine_mesh();
            std::cout << "*****************************************" << std::endl
                      << "        Do refinement ------   " << std::endl;
          }
      }
  }

  template <int dim>
  void Navier_Stokes_Newton<dim>::refine_mesh()
  {

//    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
//    FEValuesExtractors::Vector velocity(0);
//    KellyErrorEstimator<dim>::estimate (dof_handler,
//                                        QGauss<dim-1>(degree+1),
//                                        typename FunctionMap<dim>::type(),
//                                        present_solution,
//                                        estimated_error_per_cell,
//                                        fe.component_mask(velocity));
//
//    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
//                                                     estimated_error_per_cell,
//                                                     0.3, 0.0);

    triangulation.set_all_refine_flags();
    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, BlockVector<double>> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
    triangulation.execute_coarsening_and_refinement ();

    //  First DoF is set up and constraints are generated. The we create a temporary vector "tmp",
    //  whose size is according with the solution in refined mesh,
    //  to temporarily store the solution transfered from last mesh.

    setup_dof();

    BlockVector<double> tmp;
    tmp.reinit (3);
    tmp.block(0).reinit (dof_u);
    tmp.block(1).reinit (dof_w);
    tmp.block(2).reinit (dof_p);
    tmp.collect_sizes ();

    //  Transfer solution from coarse to fine mesh and apply boundary value constraints
    //  to the new transfered solution. Then set it to be the initial guess on the
    //  fine mesh. Then corresponding linear system is initialized.
    solution_transfer.interpolate(present_solution, tmp);
    initialize_system();
    nonzero_constraints.distribute(tmp);
    present_solution = tmp;
  }

  template <int dim>
    void Navier_Stokes_Newton<dim>::search_initial_guess(double step_size)
    {
      const double target_Re = 1.0/viscosity;

      bool is_initial_step = true;

      for (double Re=30.0; Re < target_Re; Re = std::min(Re+step_size, target_Re))
        {
          viscosity = 1/Re;
          std::cout << "*****************************************" << std::endl;
          std::cout << " Searching for initial guess with Re = " << Re << std::endl;
          std::cout << "*****************************************" << std::endl;

          newton_iteration(1e-12, 50, 0, is_initial_step, false);
          is_initial_step = false;
        }
    }

  template <int dim>
  void Navier_Stokes_Newton<dim>::output_results (const unsigned int output_index)  const
  {
    std::vector<std::string> solution_names (dim, "velocity");

    solution_names.push_back ("vorticity");
    solution_names.push_back ("vorticity");
    solution_names.push_back ("vorticity");
    solution_names.push_back ("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (1+2*dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation[dim+dim] = DataComponentInterpretation::component_is_scalar;
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (present_solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string (output_index, 2)
             << ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }

  template <int dim>
  void Navier_Stokes_Newton<dim>::process_solution(unsigned int refinement)
  {
//    std::ostringstream filename;
//    filename << "line-"
//             << refinement
//             << ".txt";
//
//    std::ofstream f (filename.str().c_str());
//    f << "# y u_x u_y" << std::endl;
//
//    Point<dim> p;
//    p(0)= 0.5;
//    p(1)= 0.5;
//
//    f << std::scientific;
//
//    for (unsigned int i=0; i<=100; ++i)
//      {
//
//        p(dim-1) = i/100.0;
//
//        Vector<double> tmp_vector(dim+1);
//        VectorTools::point_value(dof_handler, present_solution, p, tmp_vector);
//        f << p(dim-1);
//
//        for (int j=0; j<dim; j++)
//          f << " " << tmp_vector(j);
//        f << std::endl;
//      }

      Vector<float> difference_per_cell (triangulation.n_active_cells());

      ExactSolution<dim> exact_solution(viscosity);
      const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+dim+1);
      const ComponentSelectFunction<dim> vorticity_mask(std::make_pair(dim, dim+dim), dim+dim+1);
      const ComponentSelectFunction<dim> pressure_mask (dim+dim, dim+dim+1);

      VectorTools::integrate_difference(dof_handler,
                                        present_solution,
                                        exact_solution,
                                        difference_per_cell,
                                        QGauss<dim>(degree+2),
                                        VectorTools::L2_norm,
                                        &velocity_mask);

      const double L2_V_error = difference_per_cell.l2_norm();

      VectorTools::integrate_difference(dof_handler,
                                        present_solution,
                                        exact_solution,
                                        difference_per_cell,
                                        QGauss<dim>(degree+2),
                                        VectorTools::L2_norm,
                                        &vorticity_mask);
      const double L2_W_error = difference_per_cell.l2_norm();

      const double mean_pressure = VectorTools::compute_mean_value (dof_handler,
                                   QGauss<dim>(degree+2),
                                   present_solution,
                                   dim+dim);

      present_solution.block(2).add(-mean_pressure);
      VectorTools::integrate_difference(dof_handler,
                                        present_solution,
                                        exact_solution,
                                        difference_per_cell,
                                        QGauss<dim>(degree+2),
                                        VectorTools::L2_norm,
                                        &pressure_mask);
      const double L2_P_error = difference_per_cell.l2_norm();




      std::cout <<"*****************************************"<< std::endl
    		    <<"           L2_error                      "<< std::endl
				<<"*****************************************"<< std::endl
	            <<"At " << refinement << " refinement  |V|_l2 is: " << L2_V_error << std::endl
    		    <<"At " << refinement << " refinement  |P|_l2 is: " << L2_P_error << " with mean value "
				<< mean_pressure << std::endl
			    <<"At " << refinement << " refinement  |W|_l2 is: " << L2_W_error << std::endl;
  }

  template <int dim>
  void Navier_Stokes_Newton<dim>::run()
  {
    const Point<dim> bottom_left(0,0,0);
    const Point<dim> top_right(1,1,0.5);
    std::vector<unsigned int> direction(3);
    direction[0] = 2;
    direction[1] = 2;
    direction[2] = 1;
    GridGenerator::subdivided_hyper_rectangle(triangulation,
    									      direction,
                                              bottom_left,
                                              top_right);
    triangulation.refine_global(1);

    const double Reynold =  1.0/viscosity;

    // When the viscosity is larger than 1/1000, the solution to Stokes equations is good enough as an initial guess. If so,
    // we do not need to search for the initial guess via staircase. Newton's iteration can be started directly.
    if (Reynold <= 30)
      {
        newton_iteration(1e-12, 50, 3, true, true);
      }

    // If the viscosity is smaller than 1/1000, we have to first search for an initial guess via "staircase". What we
    // should notice is the search is always on the initial mesh, that is the $8 \times 8$ mesh in this program.
    // After the searching part, we just do the same as we did when viscosity is larger than 1/1000: run Newton's iteration,
    // refine the mesh, transfer solutions, and again.
    else
      {
        std::cout << "       Searching for initial guess ... " << std::endl;
        search_initial_guess(20.0);
        std::cout << "*****************************************" << std::endl
                  << "       Initial guess obtained            " << std::endl
                  << "                  *                      " << std::endl
                  << "                  *                      " << std::endl
                  << "                  *                      " << std::endl
                  << "                  *                      " << std::endl
                  << "*****************************************" << std::endl;


        std::cout << "       Computing solution with target viscosity ..." <<std::endl;
        std::cout << "       Reynold = " << Reynold << std::endl;
        viscosity = 1.0/Reynold;
        newton_iteration(1e-12, 50, 1, false, true);
      }

  }
}

int main()
{
  using namespace dealii;
  using namespace Step58;

  deallog.depth_console(0);

  Navier_Stokes_Newton<3> flow(1);
  flow.run();
}









