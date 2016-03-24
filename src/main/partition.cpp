#include "partition.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>

using namespace boost::numeric::ublas;

void random_partition(matrix<double, column_major> &U, gsl_rng *rng)
{
   // Initialise U randomly
   for (unsigned int i = 0; i < U.size1(); ++i)
   {
      // Randomise
      double sum = 0;
      for (unsigned int k = 0; k < U.size2(); ++k)
      {
         double u = gsl_rng_uniform(rng);
         U(i, k) = u;
         sum += u;
      }

      // Normalise the weights
      for (unsigned int k = 0; k < U.size2(); ++k)
      {
         U(i, k) /= sum;
      }
   }
}

void random_partition(matrix<unsigned int, column_major> &U, gsl_rng *rng)
{
   // Initialise U randomly
   U = zero_matrix<unsigned int>(U.size1(), U.size2());
   for (unsigned int i = 0; i < U.size1(); ++i)
   {
      // Randomise
      unsigned int k = gsl_rng_get(rng) % U.size2();
      U(i, k) = 1;
   }
}
