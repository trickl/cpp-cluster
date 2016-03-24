#include "epgkfmdsc.h"
#include <stdexcept>
#include <numeric>
#include <functional>
#include <limits>
#include <cluster/partition.h>
#include <gsl/gsl_sf_pow_int.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/lapack/gesv.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>

/*
    Difficult Challenges:

    1) Merging of prototypes, particularly when they have different dimensionality
    a) Just ditch one of the prototypes 

    2) Incremental lmds between iterations, again if dimensionality changes
    a) Use old + new lmds alignment matrix to reproject prototype

    3) Use of partition matrix to weight mds invalidates use as online algorithm
    a) Need a low dimension version of the partition matrix, one that we can 
       project onto using prototypical information.
    b) Obtain an estimate of the membership by using known distances to non-prototypes

*/

using namespace std;
using namespace boost;
using namespace boost::numeric::bindings;
using boost::numeric::ublas::range;

epgkfmdsc::lmds_node_weighted_batch::node_weighted_ds_fn::node_weighted_ds_fn(dissimilarity_function &ds_fn, const const_matrix_column& node_weights)
   : _ds_fn(ds_fn), _node_weights(node_weights)
{
};

epgkfmdsc::lmds_node_weighted_batch::node_weighted_ds_fn::node_weighted_ds_fn(const node_weighted_ds_fn& rhs)
   : _ds_fn(rhs._ds_fn), _node_weights(rhs._node_weights)
{
};

pair<double, double> epgkfmdsc::lmds_node_weighted_batch::node_weighted_ds_fn::operator()(unsigned int i, unsigned int j)
{
   pair<double, double> w_and_d = _ds_fn(i, j);
   return pair<double, double>(w_and_d.first * _node_weights(i) * _node_weights(j), w_and_d.second);
};

epgkfmdsc::lmds_node_weighted_batch::iterator::iterator(lmds_node_weighted_batch& solver, bool end)
  : _solver(solver), 
    _iteration(0),
    _end(end)
{
};

epgkfmdsc::lmds_node_weighted_batch::iterator::iterator(const iterator& rhs)
   : _solver(rhs._solver),
     _iteration(rhs._iteration),
     _end(rhs._end)
{
};

const epgkfmdsc::lmds_node_weighted_batch::iterator & epgkfmdsc::lmds_node_weighted_batch::iterator::operator++()
{
   for (unsigned int k = 0; k < _solver._size; ++k)
   {
      if (++*(_solver._iterators[k]) == _solver._solvers[k]->end())
      {
         _end = true;

         // Can we realign the prototypes for the next iteration?
      }
   } 

   _iteration++;

   return *this;
};

epgkfmdsc::lmds_node_weighted_batch::lmds_node_weighted_batch(dissimilarity_function &ds_fn,
           unsigned int size,
           const matrix& partition,
           unsigned int max_p,
           lmds::range_callback& on_fetch,
           lmds::range_callback& on_free,
           gsl_rng* rng)
   : _ds_fn(ds_fn),
     _size(size),
     _partition(partition),
     _max_p(max_p),
     _lmds_range_cb(on_fetch, on_free), 
     _rng(rng)
{
   for (unsigned int k = 0; k < _size; ++k)
   {
      // TODO: To make this a pure incremental algorithm, the node weights must be a function of some
      // prototype parameter that describes the locality of the cluster manifold.

      _ds_fns.push_back(shared_ptr<node_weighted_ds_fn>(new node_weighted_ds_fn(_ds_fn, const_matrix_column(_partition, k))));
      _solvers.push_back(shared_ptr<lmds>(new lmds(*_ds_fns.back(), _lmds_range_cb.on_fetch(), _lmds_range_cb.on_free(), _partition.size1(), _max_p, 0.001, 0.0001, 100, _rng)));
      _iterators.push_back(shared_ptr<lmds::iterator>(new lmds::iterator(_solvers.back()->begin())));
   }
};

epgkfmdsc::epgkfmdsc(dissimilarity_function &ds_fn,
           unsigned int max_clusters,
           unsigned int max_p,
           unsigned int decay_size,
           double min_prototypes_delta,
           double min_inclusion_delta,
           double similarity_threshold,
           gsl_rng* rng,
           double fuzzy_factor,
           lmds::range_callback& on_fetch,
           lmds::range_callback& on_free)
   : _ds_fn(ds_fn),
     _clusters(max_clusters),
     _max_p(max_p),
     _decay_size(decay_size),
     _inclusion_sum(identity_matrix(_clusters, _clusters)),  
     _min_prototypes_delta(min_prototypes_delta),
     _min_inclusion_delta(min_inclusion_delta),
     _similarity_threshold(similarity_threshold),
     _rng(rng),
     _fuzzy_factor(fuzzy_factor),
     _on_fetch(on_fetch),
     _on_free(on_free)
{
   _partition.resize(decay_size, _clusters);
   random_partition(_partition, _rng);
}

epgkfmdsc::iterator::iterator(epgkfmdsc& solver, bool end)
   : _solver(solver),
     _iteration(0),
     _max_prototypes_delta(numeric_limits<double>::infinity()),
     _max_inclusion_delta(numeric_limits<double>::infinity()),
     _max_inclusion(0),
     _end(end)
{
}

epgkfmdsc::iterator::iterator(const iterator& itr)
   : _solver(itr._solver),
     _iteration(itr._iteration),
     _max_prototypes_delta(itr._max_prototypes_delta),
     _max_inclusion_delta(itr._max_inclusion_delta),
     _max_inclusion(itr._max_inclusion),
     _end(itr._end)
{
}

const epgkfmdsc::iterator& epgkfmdsc::iterator::operator++()
{
   _max_prototypes_delta = 0;
   double orig_max_inclusion = _max_inclusion;

   // Store the original prototype information, so we can later calculate a delta
   vector<prototype> orig_prototypes(_solver._clusters);
   for (unsigned int k = 0; k < _solver._clusters; ++k)
   {
      orig_prototypes[k].centroid_wsum = _solver._prototypes[k].centroid_wsum;
      orig_prototypes[k].wsum          = _solver._prototypes[k].wsum;
   }

   // Setup the lmds solver for each cluster
   lmds_node_weighted_batch solver(_solver._ds_fn,
                                 _solver._clusters,
                                 _solver._partition,
                                 _solver._max_p,
                                 _solver._on_fetch,
                                 _solver._on_free,
                                 _solver._rng);

   // Subsequent iterations, refine the prototypes
   for (lmds_node_weighted_batch::iterator itr = solver.begin(), end = solver.end(); itr != end; ++itr)
   {
      matrix prototype_distances;
      _solver.update_prototypes(itr, prototype_distances);
      _solver.update_partition(prototype_distances);
   }

   // Calculate prototype centroid movement and max inclusion since the last iteration
   pair<unsigned int, unsigned int> max_inclusion_itr(0, 0);
   _max_inclusion = 0;
   for (unsigned int k = 0; k < _solver._clusters; ++k)
   {

      // Calculate prototypes movements  
      ublas_vector orig_prototype = orig_prototypes[k].centroid_wsum / orig_prototypes[k].wsum;
      ublas_vector prototype = _solver._prototypes[k].centroid_wsum / _solver._prototypes[k].wsum;

      if (orig_prototype.size() == prototype.size())
      { 
         ublas_vector prototype_delta = prototype - orig_prototype;
         _max_prototypes_delta = max(_max_prototypes_delta, atlas::dot(prototype_delta, prototype_delta));
      }
      else
      {
         // TODO: Can we project the old prototype centroid to get an estimate?
         _max_prototypes_delta = numeric_limits<double>::quiet_NaN();
      } 

      // Calculate maximum inclusion
      for (unsigned int k2 = k + 1; k2 < _solver._clusters; ++k2)
      {
         // Calculate maximum inclusion between two clusters
         double inclusion = _solver._inclusion_sum(k, k2) / min(_solver._prototypes[k].usum, _solver._prototypes[k2].usum);
         if (inclusion > _max_inclusion)
         {
            _max_inclusion = inclusion; 
            max_inclusion_itr = make_pair(k, k2);
         } 
      }
   }

   // Merge clusters if two clusters are similar and solution is in steady state
   _max_inclusion_delta = abs(orig_max_inclusion - _max_inclusion);
   if (_max_inclusion_delta <= _solver._min_inclusion_delta &&
       _max_inclusion > _solver._similarity_threshold)
   {
      _solver.merge_clusters(max_inclusion_itr.first, max_inclusion_itr.second);
   }
   else if (_max_prototypes_delta <= _solver._min_prototypes_delta)
   {
      _end = true;
   } 

  _iteration++;

  return *this;
}

// Get new prototypes (v) for each cluster using weighted median
void epgkfmdsc::update_prototypes(lmds_node_weighted_batch::iterator& lmds_itr, matrix& prototype_distances) 
{
   // Decay influence of previous feature points
   double alpha = min(1., double(prototype_distances.size1()) / double(_decay_size));

   for (unsigned int k = 0; k < lmds_itr.size(); ++k)
   {
      const matrix& features = lmds_itr.iterators()[k]->features();

      ublas_vector prototype_wsum_batch = zero_vector(features.size2());
      double wsum_batch = 0;
      double usum_batch = 0;

      for (unsigned int i = 0, n = features.size1(); i < n; ++i)
      {
         double partition_value = _partition(i, k);
         double membership_weight = pow(partition_value, _fuzzy_factor);

         usum_batch += partition_value;
         wsum_batch += membership_weight;

         prototype_wsum_batch += const_matrix_row(features, i) * membership_weight;
      }

      _prototypes[k].centroid_wsum = ((1 - alpha) * _prototypes[k].centroid_wsum) + (alpha * prototype_wsum_batch);
      _prototypes[k].wsum = ((1 - alpha) * _prototypes[k].wsum) + (alpha * wsum_batch);
      _prototypes[k].usum = ((1 - alpha) * _prototypes[k].usum) + (alpha * usum_batch);

      // Calculate the fuzzy covariance matrix of the cluster
      unsigned int p = features.size2();
      matrix covariance_batch = zero_matrix(p, p);
      matrix covariance       = zero_matrix(p, p);
      for (unsigned int j = 0; j < p; ++j)
      {
         for (unsigned int j2 = 0; j2 < p; ++j2)
         {
            // Find the deviations of each point from the cluster
            // weighted by membership
            double pval_j = _prototypes[k].centroid_wsum(j) /  _prototypes[k].wsum;
            double pval_j2 = _prototypes[k].centroid_wsum(j2) /  _prototypes[k].wsum;
            for (unsigned int i = 0, n = features.size1(); i < n; i++)
            {
               double membership_weight = pow(_partition(i, k), _fuzzy_factor);
               covariance_batch(j, j2) += (pval_j - features(i, j)) * (pval_j2 - features(i, j2)) * membership_weight;
            }

            covariance_batch(j, j2) /= wsum_batch;
            covariance(j, j2) = (1 - alpha) * _prototypes[k].eigenvalues(j) * atlas::dot(matrix_row(_prototypes[k].eigenvectors, j), matrix_column(_prototypes[k].eigenvectors, j2));
            covariance(j, j2) += alpha * covariance_batch(j, j2);
         }
      }

      // Calculate the eigenvectors and eigenvalues of the covariance matrix
      // covariance = (1 - alpha) (E * L * E') + alpha * (covariance batch)
      lapack::syev('V', 'U', covariance,  _prototypes[k].eigenvalues, lapack::optimal_workspace());

      // n.b. covariance is lost in the previous call and replaced with eigenvectors
      _prototypes[k].eigenvectors = covariance; 

      // Norm inducing matrix A is normalised covariance inverse
      // A = E * L^-1 * E` / det(L)
      double det = accumulate(_prototypes[k].eigenvalues.begin(), _prototypes[k].eigenvalues.end(), 1., multiplies<double>());
      double det_root = pow(det, 1. / double(p));

      matrix LEinv(p, p);
      for (unsigned int i = 0; i < p; ++i)
      {
         matrix_row(LEinv, i) = (det_root / _prototypes[k].eigenvalues(i)) * matrix_row(_prototypes[k].eigenvectors, i);
      }

      matrix A(p, p);
      atlas::gemm(_prototypes[k].eigenvectors, LEinv, A);

      // Now use A to calculate the distances from the cluster
      ublas_vector dev_i(p);
      ublas_vector Adev_i(p);
      for (unsigned int i = 0, n = features.size1(); i < n; ++i)
      {
         for (unsigned int j = 0; j < p; ++j)
         {
            dev_i(j) = (_prototypes[k].centroid_wsum(j) / _prototypes[k].wsum) - features(i, j);
         }

         atlas::gemv(A, dev_i, Adev_i);

         // Euclidean distance using hyperellipsoid clusters
         double dot = atlas::dot(dev_i, Adev_i);
         prototype_distances(i, k) = sqrt(dot);
      }
   }
}

// Calculate the partition matrix given the prototypes and features
void epgkfmdsc::update_partition(const matrix& prototype_distances)
{
   // Get new partition matrix for the points
   for (unsigned int k = 0; k < _clusters; ++k)
   {
      for (unsigned int i = 0, n = prototype_distances.size1(); i < n; ++i)
      {
         double membership = 0;

         if (prototype_distances(i, k) == 0) 
         {
            // Handle this awkward case
            membership = 1;
         }
         else
         {
            if (_fuzzy_factor == 1)
            { 
               double min_distance = prototype_distances(i, k);
               for (unsigned int j = 0; j < _clusters; ++j)
               {
                  if (prototype_distances(i, j) < min_distance) min_distance = prototype_distances(i, j);
               }

               if (prototype_distances(i, k) == min_distance) membership = 1;
               else membership = 0;
            }
            else
            {
               double sum = 0;
               for (unsigned int j = 0; j < _clusters; ++j)
               {
                  // Exact analytic solution given by Lagrange multipliers
                  sum += pow(prototype_distances(i, k) / prototype_distances(i, j), 2. / (_fuzzy_factor - 1.));
               }  
               membership = 1. / sum;
            } 
         }

         _partition(i, k) = membership;
      }
   }

   // Update the inclusion sum matrix
   double alpha = min(1., double(_partition.size1()) / double(_decay_size));
   symmetric_matrix inclusion_sum_batch = identity_matrix(_clusters, _clusters);
   for (unsigned int k = 0; k < _clusters; ++k)
   {
      for (unsigned int k2 = k + 1; k2 < _clusters; ++k2)
      {
         for (unsigned int i = 0, n = _partition.size1(); i < n; ++i)
         {
            inclusion_sum_batch(k, k2) += min(_partition(i, k), _partition(i, k2));
         } 
      }
   }

   _inclusion_sum = (1 - alpha) * _inclusion_sum + alpha * inclusion_sum_batch;
}

void epgkfmdsc::merge_clusters(unsigned int k, unsigned int k2)
{
   // Given prototypes are measured with different partition weights
   // merging is not trivial. So we just eliminate prototype k2.
    _clusters--;

   // Copy last element over k2 and resize
   _prototypes[k2] = _prototypes[_clusters];
   _prototypes.pop_back();

   // Update inclusion sum matrix
   // A n (B u C) approx max(A n B, A n C)
   // Update k column
   matrix_column(_inclusion_sum, k2) = matrix_column(_inclusion_sum, _clusters);
   matrix_row(_inclusion_sum, k2) = matrix_row(_inclusion_sum, _clusters);
   _inclusion_sum(k2, k2) = .1;
   _inclusion_sum.resize(_clusters, _clusters);

   // Update the partition_matrix
   matrix_column(_partition, k2) = matrix_column(_partition, _clusters);
   _partition.resize(_partition.size1(), _clusters);
}
