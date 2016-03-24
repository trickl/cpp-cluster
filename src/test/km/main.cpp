#include <exception>
#include <fstream>
#include <cluster/km.h>
#include <cluster/validation.h>
#include <cluster/partition.h>
#include <dataset/gaussian_circles.h>
#include <iostream/manipulators/csv.h>
#include <draw/plotter.h>
#include <draw/gnuplot.h>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;
using namespace std;

int main(int argc, char** argv)
{
   // Setup the gsl random number generator
   gsl_rng_env_setup();
   gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
   long seed = time (NULL) *getpid();
   gsl_rng_set(rng, seed);

   // Generate points
   matrix<double> X = gaussian_circles(rng, 200, 3, 1.0);
   matrix<double> V;
   matrix<unsigned int> U;

   try
   {
      // Perform cluster analysis
      unsigned int i; 
      km(X, V, U, rng, i, 3);

      // Validation measures
      cout << "FCM completed in " << i << " iterations." << endl;
      cout << "Partition index:   " << partition_coeff(U) << endl;
      cout << "Partition entropy: " << partition_entropy(U) << endl;
      cout << "Xie-Beni index:    " << xie_beni_index(U, V, X) << endl;

      // Output for gnuplot
      gnuplot("/home/tgee/projects/c++/algo/data/km-gnuplot.dat", X, V, U);

      // Quick interactive visual
      plotter(X, V, U);
   }
   catch (exception &e)
   {
      cerr << e.what() << endl;
   }
}
