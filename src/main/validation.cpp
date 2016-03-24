#include "validation.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <algorithm>
#include <numeric>

using namespace boost::numeric::ublas;

double partition_coeff(const matrix<double, column_major> &U)
{
   double pc = 0;

   for (unsigned int i = 0; i < U.size1(); i++)
   {
      for (unsigned int j = 0; j < U.size2(); j++)
      {
         pc += pow(U(i, j), 2.0);         
      }
   }

   pc /= double(U.size1());

   return pc;
}

double partition_entropy(const matrix<double, column_major> &U)
{
   double pe = 0;

   for (unsigned int i = 0; i < U.size1(); i++)
   {
      for (unsigned int j = 0; j < U.size2(); j++)
      {
         pe -= U(i, j) * log(U(i, j));
      }
   }

   pe /= double(U.size1());

   return pe;
}

double xie_beni_index(const matrix<double, column_major> &U, const matrix<double, column_major> &V, const matrix<double, column_major> &X, double m)
{
   unsigned int n = X.size1(); // Number of features
   unsigned int p = X.size2(); // Dimensions of features
   unsigned int c = U.size2(); // Number of features

   double compactness = 0;
   for (unsigned int k = 0; k < c; k++)
   {
      // calculate the distance vector for this cluster
      for (unsigned int i = 0; i < n; i++)
      {
         double Um = pow(U(i, k), m);

         vector<double> dev_i(p);
         for (unsigned int j = 0; j < p; j++)
         {
            dev_i(j) = V(k, j) - X(i, j);
         }

         identity_matrix<double> A(p);

         // Euclidean distance using hyperellipsoid clusters
         vector<double> Adev_i = prod(A, dev_i);

         // Calculate dot product of dev_i and Adev_i
         double dot = 0;
         for (unsigned int j = 0; j < p; j++)
         {
            dot += dev_i(j) * Adev_i(j);
         }

         compactness += dot * Um;
      }
   }
   compactness /= n;

   double minimal_separation = 0;
   for (unsigned int k = 0; k < c; k++)
   {
      for (unsigned int k2 = 0; k2 < c; k2++)
      {
         // Euclidean distance 
         if (k != k2)
         {
            vector<double> dev_k(c);
            for (unsigned int j = 0; j < p; j++)
            {
               dev_k(j) = V(k, j) - V(k2, j);
            }

            // Calculate magnitude of dev_k
            double separation = 0;
            for (unsigned int j = 0; j < p; j++)
            {
               separation += dev_k(j) * dev_k(j);
            }

            if (minimal_separation == 0)
            {
               minimal_separation = separation;
            }
            else
            { 
               minimal_separation = std::min<double>(separation, minimal_separation);
            }
         } 
      } 
   } 

   return compactness / minimal_separation;
}

void inclusion_measure(const matrix<double, column_major> &U, symmetric_matrix<double, lower, column_major> &I)
{
    I.resize(U.size2());
    // Iterate over all combinations of clusters
    for (symmetric_matrix<double, lower, column_major>::iterator1 i_itr = I.begin1(), i_end = I.end1(); i_itr != i_end; ++i_itr)
    {
       for (symmetric_matrix<double, lower, column_major>::iterator2 j_itr = i_itr.begin(), j_end = i_itr.end(); j_itr != j_end; ++j_itr)
       {
          unsigned int i = i_itr.index1(), j = j_itr.index2();
          if (i == j) I(i, j) = 1;
          else if (i > j)
          {
             matrix_column<const matrix<double, column_major> > ui(U, i);
             matrix_column<const matrix<double, column_major> >  uj(U, j);
             vector<double> umin(ui.size());
             for (unsigned int k = 0; k < ui.size(); ++k) umin(k) = std::min(ui(k), uj(k));

             I(i, j) = std::accumulate(umin.begin(), umin.end(), 0.) /
                       std::min(std::accumulate(ui.begin(), ui.end(), 0.), std::accumulate(uj.begin(), uj.end(), 0.));
          } 
       }
    }
}

