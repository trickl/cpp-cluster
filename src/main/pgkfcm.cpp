#include "pgkfcm.h"
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <limits>
#include <cluster/partition.h>
#include <gsl/gsl_sf_pow_int.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>

using namespace boost::numeric::bindings;
using namespace boost::numeric::ublas;

pgkfcm::pgkfcm(unsigned int clusters,
           unsigned int p,
           unsigned int decay_size,
           input_function &input_fn,
           double min_prototypes_delta,
           gsl_rng* rng,
           double fuzzy_factor)
   : _clusters(clusters),
     _p(p),
     _decay_size(decay_size),
     _prototypes_wsum(zero_matrix<double>(clusters, p)),
     _wsum(zero_vector<double>(clusters)),
     _prototypes_eigenvalues(zero_matrix<double>(p, clusters)),
     _prototypes_eigenvectors(zero_matrix<double>(p * clusters, p)),
     _input_fn(input_fn),
     _min_prototypes_delta(min_prototypes_delta),
     _rng(rng),
     _fuzzy_factor(fuzzy_factor)
{
}

pgkfcm::iterator::iterator(pgkfcm& solver, bool end)
   : _solver(solver),
     _iteration(0),
     _max_prototypes_delta(std::numeric_limits<double>::infinity()),
     _end(end)
{
}

pgkfcm::iterator::iterator(const iterator& itr)
   : _solver(itr._solver),
     _iteration(itr._iteration),
     _max_prototypes_delta(itr._max_prototypes_delta),
     _end(itr._end)
{
}

const pgkfcm::iterator& pgkfcm::iterator::operator++()
{
   matrix<double, column_major> features;
   if (_iteration == 0)
   {
      // First iteration, initialise the partition matrix to calculate the prototypes
      while(_solver._input_fn(features))
      {
         matrix<double, column_major> partition(features.size1(), _solver._clusters);
         random_partition(partition, _solver._rng);

         _solver.update_prototypes(partition, features);
      }
   }
   else
   {
      _max_prototypes_delta = 0;
      matrix<double, column_major> orig_prototypes_wsum = _solver._prototypes_wsum;
      vector<double> orig_wsum = _solver._wsum;

      // Subsequent iterations, refine the prototypes
      while (_solver._input_fn(features))
      {
         matrix<double, column_major> partition(features.size1(), _solver._clusters);

         _solver.update_partition(partition, features);
         _solver.update_prototypes(partition, features);
      }

      // Calculate prototypes movement
      for (unsigned int k = 0; k < _solver._clusters; ++k)
      {
         vector<double> orig_prototype = matrix_row<matrix<double, column_major> >(orig_prototypes_wsum, k) / orig_wsum(k);
         vector<double> prototype = matrix_row<matrix<double, column_major> >(_solver._prototypes_wsum, k) / _solver._wsum(k);
         vector<double> prototype_delta = prototype - orig_prototype;

         _max_prototypes_delta = std::max(_max_prototypes_delta, atlas::dot(prototype_delta, prototype_delta));
      }

      if (_max_prototypes_delta <= _solver._min_prototypes_delta) _end = true;
   }

  _iteration++;

  return *this;
}

void pgkfcm::iterator::get_prototypes(matrix<double, column_major>& centroids, std::vector<matrix<double, column_major> >& EL)
{
   centroids.resize(_solver._clusters, _solver._p);
   EL.resize(_solver._clusters);
   for (unsigned int k = 0; k < _solver._clusters; ++k)
   {
      matrix_row<matrix<double, column_major> >(centroids, k) = matrix_row<matrix<double, column_major> >(_solver._prototypes_wsum, k) / _solver._wsum(k);

      EL[k] = matrix<double, column_major>(_solver._p, _solver._p);
      matrix_column<matrix<double, column_major> > eigenvalues(_solver._prototypes_eigenvalues, k);
      matrix<double, column_major> eigenvectors = matrix_range<matrix<double, column_major> >(_solver._prototypes_eigenvectors, range(k * _solver._p, k * _solver._p + _solver._p), range(0, _solver._p));
      for (unsigned int j = 0; j < _solver._p; ++j)
      {
         matrix_column<matrix<double, column_major> >(EL[k], j) = eigenvalues(j) * matrix_row<matrix<double, column_major> >(eigenvectors, j);
      }
   }
}

void pgkfcm::iterator::get_partition(matrix<double, column_major>& partition)
{
   matrix<double, column_major> features;
   unsigned int position = 0;
   while (_solver._input_fn(features))
   {
      matrix<double, column_major> partition_range(features.size1(), _solver._clusters);
      _solver.update_partition(partition_range, features);

      partition.resize(position + features.size1(), _solver._clusters);
      matrix_range<matrix<double, column_major> >(partition, range(position, position + features.size1()), range(0, _solver._clusters)) = partition_range;

      position += features.size1();
   }
}

// Get new prototypes (v) for each cluster using weighted median
void pgkfcm::update_prototypes(const matrix<double, column_major>& partition, const matrix<double, column_major>& features) 
{
   matrix<double, column_major> prototypes_wsum_batch = zero_matrix<double>(_clusters, _p);
   vector<double> wsum_batch = zero_vector<double>(_clusters);

   for (unsigned int k = 0; k < _clusters; k++)
   {
      for (unsigned int i = 0, n = features.size1(); i < n; i++)
      {
         double membership_weight = pow(partition(i, k), _fuzzy_factor);
         wsum_batch(k) += membership_weight;

         matrix_row<matrix<double, column_major> >(prototypes_wsum_batch, k) += matrix_row<const matrix<double, column_major> >(features, i) * membership_weight;
      }
   }

   // Decay influence of previous feature points
   double alpha = std::min(1., double(features.size1()) / double(_decay_size));
   _prototypes_wsum = ((1 - alpha) * _prototypes_wsum) + (alpha * prototypes_wsum_batch);
   _wsum = ((1 - alpha) * _wsum) + (alpha * wsum_batch);

   // Incrementally calculate the hyperellipsoidal shape of each prototype using the distance covariances
   for (unsigned int k = 0; k < _clusters; k++)
   {
      matrix_column<matrix<double, column_major > > eigenvalues(_prototypes_eigenvalues, k);
      matrix<double, column_major> eigenvectors = matrix_range<matrix<double, column_major> >(_prototypes_eigenvectors, range(k * _p, k * _p + _p), range(0, _p));

      // Calculate the fuzzy covariance matrix of the cluster
      matrix<double, column_major> covariance_batch = zero_matrix<double>(_p, _p);

      matrix<double, column_major> LEt(_p, _p);
      for (unsigned int i = 0; i < _p; ++i)
      {
         matrix_row<matrix<double, column_major> >(LEt, i) = (1 - alpha) * (eigenvalues(i)) * matrix_column<matrix<double, column_major> >(eigenvectors, i);
      }

      matrix<double, column_major> covariance(_p, _p);
      atlas::gemm(eigenvectors, LEt, covariance);

      for (unsigned int j = 0; j < _p; ++j)
      {
         for (unsigned int j2 = 0; j2 < _p; ++j2)
         {

            // Find the deviations of each point from the cluster
            // weighted by membership
            for (unsigned int i = 0, n = features.size1(); i < n; i++)
            {
               double membership_weight = pow(partition(i, k), _fuzzy_factor);

               covariance_batch(j, j2) += ((_prototypes_wsum(k, j) / _wsum(k)) - features(i, j)) * ((_prototypes_wsum(k, j2) / _wsum(k)) - features(i, j2)) * membership_weight;
            }

            covariance_batch(j, j2) /= wsum_batch(k);
            covariance(j, j2) += alpha * covariance_batch(j, j2);
         }
      }

      // Calculate the eigenvectors and eigenvalues of the covariance matrix
      // covariance = alpha (E * L * E') + (1 - alpha) (covariance batch)
      lapack::syev('V', 'U', covariance, eigenvalues, lapack::optimal_workspace());

       // n.b. covariance is lost in the previous call and replaced with eigenvectors
      matrix_range<matrix<double, column_major> >(_prototypes_eigenvectors, range(k * _p, k * _p + _p), range(0, _p)) = covariance; 
   }
}

// Calculate the partition matrix given the prototypes and features
void pgkfcm::update_partition(matrix<double, column_major>& partition, const matrix<double, column_major>& features)
{
   unsigned int n = features.size1();

   // Calculate the distance vector for each cluster
   matrix<double, column_major> distances(n, _clusters);
   for (unsigned int k = 0; k < _clusters; k++)
   {
      // Norm inducing matrix A is normalised covariance inverse
      // A = E * L^-1 * E` / det(L)
      matrix_column<matrix<double, column_major> > eigenvalues(_prototypes_eigenvalues, k);
      matrix<double, column_major> eigenvectors = matrix_range<matrix<double, column_major> >(_prototypes_eigenvectors, range(k * _p, k * _p + _p), range(0, _p));
      double det = std::accumulate(eigenvalues.begin(), eigenvalues.end(), 1., std::multiplies<double>());
      double det_root = pow(det, 1. / double(_p));

      matrix<double, column_major> LEinv(_p, _p);
      for (unsigned int i = 0; i < _p; ++i)
      {
         matrix_row<matrix<double, column_major> >(LEinv, i) = (det_root / eigenvalues(i)) * matrix_row<matrix<double, column_major> >(eigenvectors, i);
      }

      matrix<double, column_major> A = prod(eigenvectors, LEinv); 

      // Now use A to calculate the distances from the cluster
      for (unsigned int i = 0; i < n; i++)
      {
         vector<double> dev_i(_p);
         for (unsigned int j = 0; j < _p; j++)
         {
            dev_i(j) = (_prototypes_wsum(k, j) / _wsum(k)) - features(i, j);
         }

         vector<double> Adev_i = prod(A, dev_i);

         // Euclidean distance using hyperellipsoid clusters
         double dot = atlas::dot(dev_i, Adev_i);
         distances(i, k) = sqrt(dot);
      }
   }

   // Get new partition matrix for the points
   for (unsigned int k = 0; k < _clusters; k++)
   {
      for (unsigned int i = 0; i < n; i++)
      {
         double membership = 0;

         if (distances(i, k) == 0) 
         {
            // Handle this awkward case
            membership = 1;
         }
         else
         {
            if (_fuzzy_factor == 1)
            { 
               double min_distance = distances(i, k);
               for (unsigned int j = 0; j < _clusters; j++)
               {
                  if (distances(i, j) < min_distance) min_distance = distances(i, j);
               }

               if (distances(i, k) == min_distance) membership = 1;
               else membership = 0;
            }
            else
            {
               double sum = 0;
               for (unsigned int j = 0; j < _clusters; j++)
               {
                  // Exact analytic solution given by Lagrange multipliers
                  sum += pow(distances(i, k) / distances(i, j), 2. / (_fuzzy_factor - 1.));
               }  
               membership = 1. / sum;
            } 
         }

         partition(i, k) = membership;
      }
   }
}
