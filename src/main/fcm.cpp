#include "fcm.h"
#include <iostream>
#include <stdexcept>
#include <cluster/partition.h>
#include <gsl/gsl_sf_pow_int.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>

using namespace std;
using namespace boost::numeric::ublas;

/* Based on XCM algorithm.
   X is feature matrix of size n x p
   c is the number of clusters to find
   m is the fuzzification factor
   epsilon is the stopping criterion for successive partitions
   U on exit is a fuzzy clustering partition matrix */
void fcm(const matrix<double, column_major> &X, matrix<double, column_major> &V, matrix<double, column_major> &U, gsl_rng *rng, unsigned int &itr, unsigned int c, double m, double epsilon, unsigned int max_itr)
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
      for (unsigned int k = 0; k < c; k++)
      {
         for (unsigned int i = 0; i < n; i++)
         {
            // Euclidean distance calculation
            double dist = 0;
            for (unsigned int j = 0; j < p; j++)
            {
               dist += gsl_sf_pow_int(V(k, j) - X(i, j), 2);
            }
            D(i, k) = sqrt(dist);
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
                  u = 1 / sum;
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
