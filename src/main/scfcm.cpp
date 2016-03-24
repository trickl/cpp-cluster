#include "scfcm.h"
#include <set>
#include <numeric>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cluster/partition.h>
#include <gsl/gsl_math.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <gsl/gsl_statistics_double.h>

using namespace std;
using namespace boost::numeric::ublas;

/* Fuzzy clustering with spatial continuity
   X is feature matrix of size n x p, where p = region size, width * height
   w is the width of the region
   c is the number of clusters to find
   m is the fuzzification factor
   epsilon is the stopping criterion for successive partitions
   U on exit is a fuzzy clustering partition matrix */
void scfcm(const matrix<double, column_major> &X, matrix<double, column_major> &V, matrix<double, column_major> &U, gsl_rng *rng, unsigned int &itr, unsigned int w, unsigned int c, double m, double epsilon, unsigned int max_itr)
{
   unsigned int n = X.size1(); // Number of features
   unsigned int p = X.size2(); // Dimensions of features

   U.resize(n, c); // Partition matrix
   random_partition(U, rng);

   // Initialize variables:
   V.resize(c, p); // Prototypes

   // Calculate the distances between neighbours
   matrix<double, column_major> D_xy(n, 8);
   int neighbour_offset[] = {-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1};

   std::multiset<double> d_avg_sorted;
   for (unsigned int i = 0; i < n; ++i)
   {
      unsigned int neighbours = 0;
      double d_avg = 0.;

      for (unsigned int y = 0; y < D_xy.size2(); ++y)
      {
         int i_offset = i + neighbour_offset[y];

         if ((i_offset < 0) || ((unsigned int) i_offset >= n) ||
             (gsl_pow_2(int(i_offset % w) - int(i % w)) * 2 > D_xy.size2()))
         {
            // Neighbour is not in the image (no wrapping allowed)
            D_xy(i, y) = numeric_limits<double>::infinity();
         }
         else
         {
            // Distance from neighbour
            double d_xy = 0., d2_xy = 0.;
            for (unsigned int j = 0; j < p; ++j)
            {
               d2_xy += gsl_pow_2(X(i_offset, j) - X(i, j));
            }

            d_xy = sqrt(d2_xy); 
            D_xy(i, y) = d_xy;
            d_avg += d_xy;
            neighbours++;
         }
      } 

      d_avg /= neighbours;
      d_avg_sorted.insert(d_avg);
   }

   // Calculate the average background randomness 'mu using D_xy
   double mu = std::accumulate(d_avg_sorted.begin(), d_avg_sorted.end(), 0.) / n;
   std::cout << "Mu : " << mu << std::endl;

   // Need to find 95th percentile of average neighbour distances
   double lambda_95 = 0.8; // Lambda corresponding to 95 percentile
   double d_avg_sorted_array[n];
   std::copy(d_avg_sorted.begin(), d_avg_sorted.end(), d_avg_sorted_array);
   double neighbour_dist95 = gsl_stats_quantile_from_sorted_data(d_avg_sorted_array, 1, n, 0.95);
   std::cout << "Distance 95 Percentile : " << neighbour_dist95 << std::endl;
   
   // Solve for steepness parameter sigma
   double sigma = (neighbour_dist95 - mu) / (log(lambda_95 / (1 - lambda_95)));
   std::cout << "Sigma : " << sigma << std::endl;

   // Write out lambda to a file
   //std::ofstream fout("lambda.dat");

   // Calculate the lambda function for all neighbours
   matrix<double, column_major> Lambda(D_xy.size1(), D_xy.size2());
   for (unsigned int i = 0; i < n; ++i)
   {
      for (unsigned int y = 0; y < D_xy.size2(); ++y)
      {
         Lambda(i, y) = 1 / (1 + exp(-(D_xy(i, y) - mu) / sigma));
         //if (i % (n / 1000) == 0) fout << D_xy(i, y) << "\t" << L(i, y) << std::endl;
      }
   }

   // Begin the main loop of alternating optimization
   double step_size = epsilon;
   for (itr = 0; itr < max_itr && step_size >= epsilon; ++itr)
   {
      // Get new prototypes (v) for each cluster using weighted median
      for (unsigned int k = 0; k < c; k++)
      {
         matrix_row<matrix<double, column_major> >(V, k) = zero_vector<double>(p);
         double denominator = 0;
         for (unsigned int i = 0; i < n; ++i)
         {
            double Um = pow(U(i, k), m);

            for (unsigned int y = 0; y < D_xy.size2(); ++y)
            {
               int i_offset = i + neighbour_offset[y];

               if ((i_offset >= 0) && ((unsigned int) i_offset < n) 
                  && (Lambda(i, y) != std::numeric_limits<double>::infinity()))
               {
                  denominator += Um;
                  matrix_row<matrix<double, column_major> >(V, k) += (Lambda(i, y) * matrix_row<const matrix<double, column_major> >(X, i) 
                                  + (1 - Lambda(i, y)) *  matrix_row<const matrix<double, column_major> >(X, i_offset) ) * Um;
               }
            } 

         }

         matrix_row<matrix<double, column_major> >(V, k) /= denominator;
      }
      
      // Calculate the prototype distances to each pixel
      matrix<double, column_major> D_kx2(n, c);
      for (unsigned int k = 0; k < c; k++)
      {
         for (unsigned int i = 0; i < n; i++)
         {
            double d_kx2 = 0;
            for (unsigned int j = 0; j < p; j++)
            {
               d_kx2 += gsl_pow_2(V(k, j) - X(i, j));
            }

            D_kx2(i, k) = d_kx2;
         }
      }

      // Dissimilarity index is a function of dissimilarity of the pixel and the prototype D_kx
      // and the dissimilarity of the pixel with it's neighbours D_xy
      matrix<double, column_major> D(n, c); 
      for (unsigned int k = 0; k < c; k++)
      {
         for (unsigned int i = 0; i < n; ++i)
         {
            unsigned int neighbours = 0;
            double dist = 0.;

            for (unsigned int y = 0; y < D_xy.size2(); ++y)
            {
               int i_offset = i + neighbour_offset[y];

               if ((i_offset >= 0) && ((unsigned int) i_offset < n) 
                  && (Lambda(i, y) != numeric_limits<double>::infinity()))
               {
                  // Distance from neighbour
                  dist += D_kx2(i, k) * Lambda(i, y) + D_kx2(i_offset, k) * (1 - Lambda(i, y)); 
                  neighbours++;
               }
            } 

            dist /= neighbours;
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
      std::cout << "Iteration: " << itr << ", step size: " << step_size << std::endl;
   }
}
