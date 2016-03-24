#include "km.h"
#include <cluster/partition.h>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_pow_int.h>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric::ublas;

/* K-means algorithm
   X is feature matrix of size n x p
   c is the number of clusters to find
   U on exit is a clustering partition matrix */
void km(const matrix<double, column_major> &X, matrix<double, column_major>& V, matrix<unsigned int, column_major> &U, gsl_rng *rng, unsigned int &itr, unsigned int c, unsigned int max_itr)
{
   unsigned int n = X.size1(); // Number of features
   unsigned int p = X.size2(); // Dimensions of features

   U.resize(n, c); // Partition matrix
   random_partition(U, rng);

   // Initialize variables:
   V.resize(c, p); // Prototypes

   bool U_has_changed = true;

   // Begin the main loop of alternating optimization
   for (itr = 0; itr < max_itr && U_has_changed; ++itr)
   {
      // Get new prototypes (v) for each cluster using weighted median
      for (unsigned int k = 0; k < c; k++)
      {

         for (unsigned int j = 0; j < p; j++)
         {
            unsigned int sum = 0;
            V(k, j) = 0; 
        
            for (unsigned int i = 0; i < n; i++)
            {
               unsigned int Um = U(i, k);
               sum += Um;

               V(k, j) += X(i ,j) * Um;
            }

            V(k, j) /= double(sum);
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
      U_has_changed = false;
      for (unsigned int i = 0; i < n; i++)
      {
         double mind = std::numeric_limits<double>::max();
         unsigned int closest_k = 0;

         for (unsigned int k = 0; k < c; k++)
         {
            // U = 1 for the closest prototype
            // U = 0 otherwise

            if (D(i, k) < mind)
            {
               mind = D(i, k);
               closest_k = k;
            } 
         }

         if (!U(i, closest_k))
         {
            U_has_changed = true;

            for (unsigned int k = 0; k < c; k++)
            {
                U(i, k) = (k == closest_k) ? 1 : 0; 
            }
         }
      }
   }
}
