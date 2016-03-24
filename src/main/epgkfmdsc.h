#ifndef _EPGKFMDSC_H
#define _EPGKFMDSC_H

#include <cluster/typedefs.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <mds/dissimilarity_function.h>
#include <gsl/gsl_rng.h>
#include <mds/lmds.h>
#include <boost/shared_ptr.hpp>

struct rng;

// Extended Progressive Gustafson-Kessel MDS Clustering
// Note that this algoritm is not incremental, as the 
// partition table is required for each iteration.
// (A true incremental algorithm would store a descriptor
// of the cluster manifold as part of the prototype information).
// However, the point of the progression is simply to decrease
// the memory footprint of mds clustering for large datasets.
// A fully incremental algorithm, i.e. usable for a online app,
// would be a challenging project in the future.

// Thoughts:
// Need to rearchitect the relationship between the local mds algorithm and the progressive
// clustering algorithm, it is very clunky.
class epgkfmdsc
{
public:
   struct prototype
   {
      // Weighted centroid
      ublas_vector centroid_wsum;

      // Covariance eigenvectors and values
      ublas_vector eigenvalues;
      matrix       eigenvectors;

      // Weight measures
      double       wsum;
      double       usum;
   };

   class lmds_node_weighted_batch
   {
   public:

      // Weight dissimilarity function by cluster membership
      class node_weighted_ds_fn : public dissimilarity_function
      {
      public:
         node_weighted_ds_fn(dissimilarity_function &ds_fn, const const_matrix_column& node_weights);
         node_weighted_ds_fn(const node_weighted_ds_fn &rhs);
         virtual ~node_weighted_ds_fn() {};

         virtual std::pair<double, double> operator()(unsigned int i, unsigned int j);

      private:
         dissimilarity_function& _ds_fn;
         const const_matrix_column &_node_weights;
      };

      class iterator
      {
      public:
         iterator(lmds_node_weighted_batch& solver, bool end = false);
         iterator(const iterator& rhs);

      const iterator &operator++();
      bool operator==(const iterator& rhs) const { return _end == rhs._end; };
      bool operator!=(const iterator& rhs) const { return !(*this == rhs); };

      unsigned int iteration() const {return _iteration; };
      unsigned int size() const {return _solver._size; };
      const std::vector<boost::shared_ptr<lmds::iterator> >& iterators() const {return _solver._iterators; };

      private:
         lmds_node_weighted_batch& _solver;
         unsigned int _iteration;
         bool _end;
      };


      lmds_node_weighted_batch(dissimilarity_function &ds_fn,
              unsigned int size,
              const matrix& partition,
              unsigned int max_p,
              lmds::range_callback& on_fetch,
              lmds::range_callback& on_free,
              gsl_rng* rng);

      iterator begin() { return iterator(*this); };
      iterator end() {return iterator(*this, true); };

      unsigned int size() {return _size; };

   private:
      dissimilarity_function& _ds_fn;
      unsigned int _size;
      const matrix& _partition;
      unsigned int _max_p;
      lmds::range_callback_buffer _lmds_range_cb;
      std::vector<boost::shared_ptr<node_weighted_ds_fn> > _ds_fns;
      std::vector<boost::shared_ptr<lmds> > _solvers;
      std::vector<boost::shared_ptr<lmds::iterator> > _iterators;
      gsl_rng* _rng;
   };

   class iterator
   {
   public:
      iterator(epgkfmdsc& solver, bool end = false);
      iterator(const iterator& itr);

      const iterator &operator++();
      bool operator==(const iterator& rhs) { return _end == rhs._end; };
      bool operator!=(const iterator& rhs) { return !(*this == rhs); };

      unsigned int iteration() {return _iteration; };
      unsigned int clusters() {return _solver._clusters; };
      double max_prototypes_delta() {return _max_prototypes_delta; };
      double max_inclusion_delta() {return _max_inclusion_delta; };
      double max_inclusion() {return _max_inclusion; };
      void get_prototypes(std::vector<prototype>& prototypes) {prototypes = _solver._prototypes;} ;
      void get_partition(matrix& partition) {partition = _solver._partition; };

   private:
      epgkfmdsc& _solver;
      unsigned int _iteration;
      double _max_prototypes_delta;
      double _max_inclusion_delta;
      double _max_inclusion;
      bool _end;
   };

   epgkfmdsc(dissimilarity_function& ds_fn,
        unsigned int max_clusters,
        unsigned int max_p,
        unsigned int decay_size, 
        double min_prototype_delta,
        double min_inclusion_delta,
        double similarity_threshold,
        gsl_rng* rng,
        double fuzzy_factor,
        lmds::range_callback& on_fetch,
        lmds::range_callback& on_free);


   iterator begin() { return iterator(*this); };
   iterator end() {return iterator(*this, true); };

private:
   void update_prototypes(lmds_node_weighted_batch::iterator& lmds_itr, matrix& prototype_distances);
   void update_partition(const matrix& prototype_distances);
   void merge_clusters(unsigned int k, unsigned int k2);

private:
   dissimilarity_function& _ds_fn;
   unsigned int _clusters;
   unsigned int _max_p;
   unsigned int _decay_size;
   std::vector<prototype> _prototypes;
   matrix       _inclusion_sum;
   matrix       _partition;
   double       _min_prototypes_delta;
   double       _min_inclusion_delta;
   double       _similarity_threshold;
   gsl_rng*     _rng; 
   double       _fuzzy_factor;
   lmds::range_callback& _on_fetch;
   lmds::range_callback& _on_free;
};

#endif
