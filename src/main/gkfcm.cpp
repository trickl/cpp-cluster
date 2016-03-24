#include "gkfcm.h"
#include <iostream>
#include <stdexcept>
#include <cluster/partition.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>

using namespace boost::numeric::bindings;
using namespace boost::numeric::ublas;

void gkfcm(const matrix<double, column_major> &X, matrix<double, column_major>& V, matrix<double, column_major> &U, std::vector<matrix<double, column_major> > &EL, gsl_rng* rng, unsigned int &itr, unsigned int c, double m, double epsilon, unsigned int max_itr)
{
   unsigned int n = X.size1(); // Number of features
   unsigned int p = X.size2(); // Dimensions of features

   U.resize(n, c); // Partition matrix
   random_partition(U, rng);

   // Initialize variables:
   V.resize(c, p); // Prototypes

   // Begin the main loop of alternating optimization
   double step_size = epsilon;
   for (itr = 0; itr < max_itr && step_size >= epsilon; ++itr)
   {
      // Get new prototypes (v) for each cluster using weighted median
      for (unsigned int k = 0; k < c; k++)
      {

         for (unsigned int j = 0; j < p; j++)
         {
            double sum = 0;
            V(k, j) = 0; 
        
            for (unsigned int i = 0; i < n; i++)
            {
               double Um = pow(U(i, k), m);
               sum += Um;

               V(k, j) += X(i ,j) * Um;
            }

            V(k, j) /= sum;
         }
      }

      // Calculate distance measure d:
      matrix<double, column_major> D(n, c);
      EL.resize(c);   // Cluster covariance eigenvectors * eigenvalues
      for (unsigned int k = 0; k < c; k++)
      {
         // Calculate the fuzzy covariance matrix of the cluster
         matrix<double, column_major> cov(p, p); 
         for (unsigned int j = 0; j < p; j++)
         {
            for (unsigned int j2 = 0; j2 < p; j2++)
            {
              double sum = 0; 

               // Find the deviations of each point from the cluster           
               // weighted by membership
               cov(j, j2) = 0;
               for (unsigned int i = 0; i < n; i++)
               {
                  double Um = pow(U(i, k), m);
                  sum += Um;

                  cov(j, j2) += (V(k, j) - X(i, j)) * (V(k, j2) - X(i, j2)) * Um;
               } 

               cov(j, j2) /= sum;
            }
         }

         // Calculate the eigenvectors and eigenvalues of the covariance matrix
         // cov = E * L * E'
         vector<double> L(cov.size1());  // eigenvalues
         lapack::syev('V', 'U', cov, L, lapack::optimal_workspace()); // n.b. cov is lost in this call and replaced with E
         matrix<double, column_major> E = cov; // eigenvectors

         // Calculate the cov determinant using the eigenvalues
         double detL = 1;
         for (unsigned int j = 0; j < p; j++) detL *= L(j); 
         double detL_root = pow(detL, 1. / double(p)); 

         // Norm inducing matrix A is normalised covariance inverse
         matrix<double, column_major> LEinv(L.size(), E.size1());
         EL[k].resize(p, p);
         for (unsigned int i = 0; i < L.size(); i++)
         {
           
            matrix_row<matrix<double, column_major> >(LEinv, i) = (detL_root / L(i)) * matrix_row<matrix<double, column_major> >(E, i);

            // Store EL as descriptor for cluster shape
            matrix_column<matrix<double, column_major> >(EL[k], i) = L(i) * matrix_row<matrix<double, column_major> >(E, i);
         }
         
         // A = E * L^-1 * E' / det(L)
         matrix<double, column_major> A = prod(E, LEinv);

         // calculate the distance vector for this cluster
         for (unsigned int i = 0; i < n; i++)
         {
            vector<double> dev_i(p);
            for (unsigned int j = 0; j < p; j++)
            {
               dev_i(j) = V(k, j) - X(i, j);
            } 

            vector<double> Adev_i = prod(A, dev_i);

            // Calculate dot product of dev_i and Adev_i
            // Hmm an expression template for this in boost would be nice :P
            double dot = 0;
            for (unsigned int j = 0; j < p; j++)
            {
               dot += dev_i(j) * Adev_i(j);
            }

            // Euclidean distance using hyperellipsoid clusters
            D(i, k) = sqrt(dot);
         }
      }

      // Get new partition matrix U:
      step_size = 0;
      for (unsigned int k = 0; k < c; k++)
      {
         for (unsigned int i = 0; i < n; i++)
         {
            double u = 0;

            if (D(i, k) == 0) 
            {
               // Handle this awkward case
               u = 1;
            }
            else
            {
               if (m == 1)
               { 
                  double mind = D(i, k);
                  for (unsigned int j = 0; j < c; j++)
                  {
                     if (D(i, j) < mind) mind = D(i, j);
                  }

                  if (D(i, k) == mind) u = 1;
                  else u = 0;
               }
               else
               {
                  double sum = 0;
                  for (unsigned int j = 0; j < c; j++)
                  {
                     // Exact analytic solution given by Lagrange multipliers
                     sum += pow(D(i, k) / D(i, j), 2.0 / (m - 1.0));
                  }  
                  u = 1.0 / sum;
               } 
            }

            double u0 = U(i, k);
            U(i, k) = u;

            // Stepsize is max(delta(U))
            if (u - u0 > step_size) step_size = u - u0;
         }
      }
   }
}
