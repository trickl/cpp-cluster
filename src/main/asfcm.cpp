#include "asfcm.h"
#include <curve/bspline.h>
#include <cluster/partition.h>
#include <cluster/hardening.h>
#include <iostream/manipulators/csv.h>
#include <map>
#include <set>
#include <numeric>
#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics_double.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>

#include <iostream>
#include <iomanip>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;
using namespace boost::numeric::bindings;

/* Adaptive Fuzzy C-Means Clustering with Spatial Constraints
   Implemented based on approach in 
   "Image Segmentation Based On Adaptive Cluster Prototype Estimation"
   by A. Liew and H. Yan
   X is feature matrix of size n x 3, needs to be in Lab space
   c is the number of clusters to find
   m is the fuzzification factor
   epsilon is the stopping criterion for successive partitions
   U on exit is a fuzzy clustering partition matrix */
void asfcm(const matrix<double, column_major> &X, matrix<double, column_major> &V, matrix<double, column_major> &U, gsl_rng *rng, unsigned int &itr, unsigned int width, double knot_frequency, double smoothing, unsigned int c, double m, double epsilon, unsigned int max_itr)
{
   unsigned int n = X.size1(); // Number of features
   unsigned int p = X.size2(); // Dimensions of features
   bool adapt_prototypes = false; // Flag to adapt prototypes
   unsigned int height = X.size1() / width;
   const unsigned char order = 3;
   unsigned int x_knots = (1. / knot_frequency) + order;
   unsigned int y_knots = (1. / knot_frequency) + order;

   U.resize(n, c); // Partition matrix
   random_partition(U, rng);

   // Initialize variables:
   V.resize(c, p); // Prototypes

   // The b-spline co-efficients for the prototypes
   // x is measured from top left increasing right, y from top left increasing down
   // Separate cluster matrices are joined vertically and are contigent.
   matrix<double, column_major> V_knots(y_knots * c, x_knots);

   // The spatially adaptive multiplicative field for the prototypes
   matrix<double, column_major> W = scalar_matrix<double>(X.size1(), c, 1.);

   // Convolution matrices for the bspline solver (these are identical between calls)
   matrix<double, column_major> R = smoothing * regularization_matrix<order>(x_knots, y_knots); 
   matrix<double, column_major> LXC = Lx_convolve_matrix<order>(x_knots, width);
   matrix<double, column_major> LYC = Ly_convolve_matrix<order>(y_knots, height);
   matrix<double, column_major> BXC = Bx_convolve_matrix<order>(x_knots, width);
   matrix<double, column_major> BYC = By_convolve_matrix<order>(y_knots, height);

   matrix<double, column_major> lightness_matrix(height, width);
   for (unsigned int sx = 0; sx < width; ++sx)
   {
      for (unsigned int sy = 0; sy < height; ++sy)
      {
         lightness_matrix(sy, sx) = matrix_row<const matrix<double, column_major> >(X, sx + (sy * width))[0]; // LAB -> L
      }
   }

   // Calculate the distances between neighbours
   matrix<double, column_major> D_xy(n, 8);
   int neighbour_offset[] = {-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1};

   std::multiset<double> d_avg_sorted;
   for (unsigned int i = 0; i < n; ++i)
   {
      unsigned int neighbours = 0;
      double d_avg = 0.;

      for (unsigned int y = 0; y < D_xy.size2(); ++y)
      {
         int i_offset = i + neighbour_offset[y];

         if ((i_offset < 0) || ((unsigned int) i_offset >= n) ||
             (gsl_pow_2(int(i_offset % width) - int(i % width)) * 2 > D_xy.size2()))
         {
            // Neighbour is not in the image (no wrapping allowed)
            D_xy(i, y) = std::numeric_limits<double>::infinity();
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

   // Calculate the lambda function for all neighbours
   matrix<double, column_major> Lambda(D_xy.size1(), D_xy.size2());
   for (unsigned int i = 0; i < n; ++i)
   {
      for (unsigned int y = 0; y < D_xy.size2(); ++y)
      {
         Lambda(i, y) = 1 / (1 + exp(-(D_xy(i, y) - mu) / sigma));
      }
   }

   // Begin the main loop of alternating optimization
   double step_size = epsilon;
   for (itr = 0; itr < max_itr && step_size >= epsilon; ++itr)
   {
      if (adapt_prototypes)
      {
         // Solve for the B-Spline co-efficients for each cluster
         for (unsigned int k = 0; k < c; ++k)
         {
            // Calculate L and B (definitions are in the paper)
            double prototype_lightness = matrix_row<matrix<double, column_major> >(V, k)[0]; // LAB -> L

            // Membership matricies
            matrix<double, column_major> U2(height, width);
            matrix<double, column_major> U2L(height, width);
            for (unsigned int sx = 0; sx < width; ++sx)
            {
               for (unsigned int sy = 0; sy < height; ++sy)
               {
                  U2(sy, sx) = gsl_pow_2(U(sx + (sy * width), k));
                  U2L(sy, sx) = U2(sy, sx) * lightness_matrix(sy, sx);
               }
            } 

            // Use the precalculated convolution matrices to speed up the calculations
            matrix<double, column_major> L(x_knots * y_knots, x_knots * y_knots);
            matrix<double, column_major> L_temp(LYC.size1(), U2.size2()); 
            matrix<double, column_major> L_temp2(LYC.size1(), LXC.size2());

            opb_prod(LYC, U2, L_temp);
            opb_prod(L_temp, LXC, L_temp2);

            // Transform into the t, u ordering
            for (unsigned int p = 0; p < x_knots; ++p)
            {
               for (unsigned int q = 0; q < y_knots; ++q)
               {
                  for (unsigned int i = 0; i < x_knots; ++i)
                  {
                     for (unsigned int j = 0; j < y_knots; ++j)
                     {
                        L(q * y_knots + p, j * x_knots + i) = L_temp2(q * x_knots + j, p * y_knots + i) * prototype_lightness * prototype_lightness;
                     }
                  } 
               }
            } 

            vector<double> B(x_knots * y_knots);
            matrix<double, column_major> B_temp(BYC.size1(), U2L.size2()); 
            matrix<double, column_major> B_temp2(BYC.size1(), BXC.size2());

            opb_prod(BYC, U2L, B_temp);
            opb_prod(B_temp, BXC, B_temp2);

            // Transform into the t ordering
            for (unsigned int p = 0; p < x_knots; ++p)
            {
               for (unsigned int q = 0; q < y_knots; ++q)
               {
                  B(q * x_knots + p) = B_temp2(q, p) * prototype_lightness;
               }
            }

            matrix<double, column_major> A = L + R;
            vector<double> theta(x_knots + y_knots);

            // Use SVD to calculate the b spline coefficients (theta), since A * theta = B;
            // theta = V * S * U' * B;
            vector<double> S(A.size2());
            matrix<double, column_major> U(A.size1(), A.size2()), Vt(A.size2(), A.size2());

            lapack::gesvd(A, S, U, Vt);

            matrix<double, column_major> SUt(A.size2(), A.size1());
            for (unsigned int i = 0; i < A.size2(); i++)
            {
               matrix_row<matrix<double, column_major> >(SUt, i) = matrix_column<matrix<double, column_major> >(U, i) / S(i);
            }

            // Get V from transpose Vt
            matrix<double, column_major> V(A.size2(), A.size2());
            for (matrix<double, column_major>::const_iterator1 i_itr = Vt.begin1(), i_end = Vt.end1(); i_itr != i_end; ++i_itr)
            {
               for (matrix<double, column_major>::const_iterator2 j_itr = i_itr.begin(), j_end = i_itr.end(); j_itr != j_end; ++j_itr)
               {
                  unsigned long i = i_itr.index1(), j = j_itr.index2();
                  V(j, i) =  Vt(i, j);
               }
            }

            matrix<double, column_major> Ainv(A.size2(), A.size1());
            opb_prod(V, SUt, Ainv);
            theta = prod(Ainv,  B);

            // Store the coefficients in V_knots
            for (unsigned int i = 0; i < x_knots; ++i)
            {
	       for (unsigned int j = 0; j < y_knots; ++j)
	       {
                  unsigned int u = j * x_knots + i;

                  V_knots(j + (y_knots * k), i) = theta(u); 
               }
            }
         }
         
         // Update the spatially adaptive multiplicative field W
         for (unsigned int k = 0; k < c; k++)
         {
            for (unsigned int i = 0; i < n; i++)
            {
               // Measure to the centre of a pixel
               double tx = (double((i % width) + .5) / double(width)) * double(x_knots - order);
               double ty = (double((i / width) + .5) / double(height)) * double(y_knots - order);

               W(i, k) = tensor_bspline_value<order, 0>(tx, ty, matrix_range<matrix<double, column_major> >(V_knots, range(y_knots * k, y_knots * (k + 1)), range(0, x_knots)));
            } 
         }
         
      }
      
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
                  denominator += (Lambda(i, y) * W(i, k) * W(i, k) + (1 - Lambda(i, y)) * W(i_offset, k) * W(i_offset, k)) * Um;

                  matrix_row<matrix<double, column_major> >(V, k) += (Lambda(i, y) * matrix_row<const matrix<double, column_major> >(X, i) * W(i, k)
                                  + (1 - Lambda(i, y)) *  matrix_row<const matrix<double, column_major> >(X, i_offset) * W(i_offset, k)) * Um;
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

            // Adjust lightness value by the adaptive field
            vector<double> adjusted_prototype = matrix_row<matrix<double, column_major> >(V, k);
            vector<double> adjusted_point     = matrix_row<const matrix<double, column_major> >(X, i);

            if (adapt_prototypes)
            {
               // Adjust the lightness of the prototype according to the spatial field 
               adjusted_prototype[0] *= W(i, k); 
            }
            else
            {
               // Only compute distance based on chromacity, ignore lightness
               adjusted_point[0] = adjusted_prototype[0]; 
            } 

            for (unsigned int j = 0; j < p; j++)
            {
               d_kx2 += gsl_pow_2(adjusted_prototype(j) - adjusted_point(j));
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
                  && (Lambda(i, y) != std::numeric_limits<double>::infinity()))
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
                     sum += pow(D(i, k) / D(i, j), 2. / (m - 1.));
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

      std::cout << "Iteration:" << itr << ", Step size: " << step_size << std::endl; 

      if (step_size < epsilon && !adapt_prototypes)
      {
         adapt_prototypes = true;
         step_size = epsilon;
         std::cout << "Starting to perform prototype adaptation..." << std::endl;
      }
   }
}

