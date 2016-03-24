#include "afcm.h"
#include <curve/bspline.h>
#include <cluster/partition.h>
#include <cluster/hardening.h>
#include <iostream/manipulators/csv.h>
#include <map>
#include <gsl/gsl_math.h>
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

/* Adaptive Fuzzy C-Means Clustering
   Implemented based on approach in 
   "Image Segmentation Based On Adaptive Cluster Prototype Estimation"
   by A. Liew and H. Yan
   X is feature matrix of size n x 3, needs to be in Lab space
   c is the number of clusters to find
   m is the fuzzification factor
   epsilon is the stopping criterion for successive partitions
   U on exit is a fuzzy clustering partition matrix */
void afcm(const matrix<double, column_major> &X, matrix<double, column_major> &V, matrix<double, column_major> &U, gsl_rng *rng, unsigned int &itr, unsigned int width, double knot_frequency, double smoothing, unsigned int c, double m, double epsilon, unsigned int max_itr)
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
               denominator += Um * W(i, k) * W(i, k);
               matrix_row<matrix<double, column_major> >(V, k) += matrix_row<const matrix<double, column_major> >(X, i) * Um * W(i, k);
         }

         matrix_row<matrix<double, column_major> >(V, k) /= denominator;
      }

      if (adapt_prototypes)
      {
/*
         // Perform region relabelling to merge regions with significant transitions between their boundaries      
         // TODO : TODO : TODO: TODO
           
         // Reconstruct the image A using the prototypes
         matrix<double, column_major> A(X.size1(), X.size2());
         for (unsigned int i = 0; i < n; i++)
         {
            matrix_row<matrix<double, column_major> >(A, i) = zero_vector<double>(X.size2());
            for (unsigned int k = 0; k < c; k++)
            {
               matrix_row<matrix<double, column_major> >(A, i) += U(i, k) * W(i, k) * matrix_row<matrix<double, column_major> >(V, k);
            }

         }

         // Write out the image lightness
         std::cout << "Image: " << std::endl;
         for (unsigned int i = 0; i < n; i++)
         {
            std::cout << lightness(matrix_row<const matrix<double, column_major> >(X, i));
            if ((i + 1) % width == 0) std::cout << std::endl;
            else std::cout << ", ";
         }

         // Write out the reconstructed image
         std::cout << "Reconstruction: " << std::endl;
         for (unsigned int i = 0; i < n; i++)
         {
            std::cout << lightness(matrix_row<matrix<double, column_major> >(A, i));
            if ((i + 1) % width == 0) std::cout << std::endl;
            else std::cout << ", ";
         }

         // Write out the residuals
         std::cout << "Residuals: " << std::endl;
         for (unsigned int i = 0; i < n; i++)
         {
            std::cout << lightness(matrix_row<matrix<double, column_major> >(A, i)) -  lightness(matrix_row<const matrix<double, column_major> >(X, i));
            if ((i + 1) % width == 0) std::cout << std::endl;
            else std::cout << ", ";
         }

         // Calculate the gradient magnitude G across A
         matrix<double, column_major> G(height - 1, width - 1);
         for (unsigned int sx = 0; sx < width - 1; ++sx)
         {
            for (unsigned int sy = 0; sy < height - 1; ++sy)
            {
               unsigned int n = sx + (sy * width);

               // G = sqrt(dA'dx ^ 2 + dA'dy ^ 2)
               vector<double> dAdx = matrix_row<matrix<double, column_major> >(A, n + 1) - matrix_row<matrix<double, column_major> >(A, n) ;
               vector<double> dAdy = matrix_row<matrix<double, column_major> >(A, n + width) - matrix_row<matrix<double, column_major> >(A, n);
               G(sy, sx) = sqrt(inner_prod(dAdx, dAdx) + inner_prod(dAdy, dAdy));
            }
         }

         // Write out the residuals
         std::cout << "Gradient Image:" << std::endl;
         std::cout << csv << G << std::endl;

         // Threshold G to remove insignificant gradient
         for (unsigned int sx = 0; sx < width - 1; ++sx)
         {
            for (unsigned int sy = 0; sy < height - 1; ++sy)
            {
               // TODO
            }
         }

         // Perform a hard classification on the image according to the highest membership
         matrix<unsigned int> R(height, width);
         for (unsigned int sx = 0; sx < width; ++sx)
         {
            for (unsigned int sy = 0; sy < height; ++sy)
            {
               unsigned int n = sx + (sy * width);
               matrix_row<const matrix<double, column_major> > U_row(U, n);
               R(sy, sx) = std::max_element(U_row.begin(), U_row.end()).index());
            }
         }

         // Calculate the boundary pixels 
         matrix<unsigned int> B(height - 1, width - 1);
         for (unsigned int sx = 0; sx < width - 1; ++sx)
         {
            for (unsigned int sy = 0; sy < height - 1; ++sy)
            {
               unsigned int n = sx + (sy * width);
               if (R(n, k) != R(n + 1, k) 
                || R(n, k) != R(n + width, k)) B(sy, sx) = 1;
               else B(sy, sx) = 0;
            } 
         } 
         


         break;


         // Trace the boundary of each region
         symmetric_matrix<double, column_major> B = zero_matrix<double, column_major>(c, c);
         for (unsigned int k = 0; k < c; k++)
         {
            // Get the mean square difference between region boundaries
            
         } 

         // If the mean square difference is less than a threshold, merge these two regions
         double threshold = 0.; // TODO
         for (unsigned int k1 = 0; k1 < c; k1++)
         {
            for (unsigned int k2 = k1 + 1; k2 < c; k2++)
            {
               if (B(k1, k2) < threshold)
               {
                  // Merge these regions
               }
            } 
         }
*/
      }

      // Calculate distance measure d:
      matrix<double, column_major> D(n, c);
      for (unsigned int k = 0; k < c; k++)
      {
         for (unsigned int i = 0; i < n; i++)
         {
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

            // Euclidean distance calculation
            D(i, k) = norm_2(adjusted_prototype - adjusted_point);
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

         // Assign each pixel to the cluster with the largest membership value
         // Not sure when to do this?
         /*
         for (unsigned int i = 0; i < U.size1(); i++)
         {
            matrix_row<const matrix<double, column_major> > R(U, i);
            matrix_row<matrix<double, column_major> >(U, i) = unit_vector<double>(U.size2(), std::max_element(R.begin(), R.end()).index());
         }
         */
      }
   }
}
