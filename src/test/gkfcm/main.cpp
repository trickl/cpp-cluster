#include <exception>
#include <fstream>
#include <cluster/gkfcm.h>
#include <cluster/validation.h>
#include <cluster/partition.h>
#include <cluster/clusters_gnuplot.h>
#include <cluster/clusters_visual.h>
#include <dataset/gaussian_circles.h>
#include <iostream/manipulators/csv.h>
#include <draw/drawing_pad.h>
#include <boost/numeric/ublas/io.hpp>
#include <Fl/Fl.H>

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
   matrix<double, column_major> X = gaussian_circles(rng, 200, 3, 1.0);
   matrix<double, column_major> V, U;
   std::vector<matrix<double, column_major> > A;

   try
   {
      // Perform cluster analysis
      unsigned int i; 
      gkfcm(X, V, U, A, rng, i, 3);

      // Validation measures
      cout << "FCM completed in " << i << " iterations." << endl;
      cout << "Partition index:   " << partition_coeff(U) << endl;
      cout << "Partition entropy: " << partition_entropy(U) << endl;
      cout << "Xie-Beni index:    " << xie_beni_index(U, V, X) << endl;

      // Output for gnuplot
      clusters_gnuplot("/home/tgee/projects/c++/algo/data/gkfcm-gnuplot.dat", X, V, U, A);

      // Quick interactive visual
      drawing_pad pad;
      pad.set_scale(0, 0, 10, 10);
      clusters_visual c(X, V, U, A);   
      pad.draw(&c);
      Fl::run();
   }
   catch (exception &e)
   {
      cerr << e.what() << endl;
   }
}
