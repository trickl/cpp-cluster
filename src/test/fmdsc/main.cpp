#include <mds/rkisomap.h>
#include <cluster/fmdsc.h>
#include <cluster/validation.h>
#include <cluster/partition.h>
#include <dataset/spiral_flower.h>
#include <mds/euclidean_distance.h>
#include <iostream/manipulators/csv.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <exception>
#include <fstream>

using namespace std;

class dissimilarity_functor : public dissimilarity_function
{
public:
   dissimilarity_functor(const symmetric_matrix& R)
      : _R(R) {};

   virtual pair<double, double> operator() (unsigned int i, unsigned int j)
   {
      pair<double, double> weight_and_distance = pair<double, double>(1., _R(i, j));
      return weight_and_distance;
   };

private:
   const symmetric_matrix& _R;
};

int main(int argc, char** argv)
{
   // Setup the gsl random number generator
   gsl_rng_env_setup();
   gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
   long seed = time (NULL) *getpid();
   gsl_rng_set(rng, seed);

   // Load feature data from file 
   matrix X;
   ifstream fin("/home/Sleeper/projects/netflix/data/spiral_flower.dat");
   fin >> csv >> X;

   // Convert into a dissimilarity matrix
   cout << "Creating distance matrix..." << endl;
   symmetric_matrix R;
   euclidean_distance(X, R); 

   try
   {
      // Perform cluster analysis

      // Preprocess with a manifold detector
      cout << "Preprocessing with isomap..." << endl;
      symmetric_matrix S;
      rkisomap(R, S, 5);

      cout << "Running fuzzy cluster algorithm..." << endl;
      dissimilarity_functor ds(R);

      matrix U(R.size1(), 10); 
      random_partition(U, rng);

      fmdsc solver(ds, U, 4, 0.75, 0.01, 1e-3, 1e-3, 2., 1e-4, 100, rng);
      for (fmdsc::iterator itr = solver.begin(), end = solver.end(); itr != end; ++itr)
      {
         cout << "Iteration : " << itr.iteration() << endl;
         cout << "Clusters  : " << itr.clusters() << endl;
         cout << "Step size : " << itr.step_size() << endl;
         cout << "Fit tolerance : " << itr.applied_fit_tolerance() << endl;
      }

      // Write out the partition data to a color feature file
      matrix F;
      color_partition(X, U, F, hsv);
      ofstream fout("/home/Sleeper/projects/netflix/data/spiral_flower_fmdsc_manifold.dat");
      fout << csv << F;

      cout << "Partition index:   " << partition_coeff(U) << endl;
      cout << "Partition entropy: " << partition_entropy(U) << endl;
   }
   catch (exception &e)
   {
      cerr << e.what() << endl;
   }
}
