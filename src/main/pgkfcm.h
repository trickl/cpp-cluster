#ifndef _PGKFCM_H
#define _PGKFCM_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <gsl/gsl_rng.h>

struct rng;

// Progressive Gustafson-Kessel Fuzzy C-Means Clustering
class pgkfcm
{
public:
   class input_function
   {
   public:
      virtual bool operator()(boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>& feed) = 0;
   };

   class iterator
   {
   public:
      iterator(pgkfcm& solver, bool end = false);
      iterator(const iterator& itr);

      const iterator &operator++();
      bool operator==(const iterator& rhs) { return _end == rhs._end; };
      bool operator!=(const iterator& rhs) { return !(*this == rhs); };

      unsigned int iteration() {return _iteration; };
      double max_prototypes_delta() {return _max_prototypes_delta; };
      void get_prototypes(boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>& centroids, std::vector<boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> >& EL);
      void get_partition(boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>& partition);

   private:
      pgkfcm& _solver;
      unsigned int _iteration;
      double _max_prototypes_delta;
      bool _end;
   };

   pgkfcm(unsigned int clusters,
        unsigned int p,
        unsigned int decay_size, 
        input_function &input_fn,
        double min_prototype_delta,
        gsl_rng* rng,
        double fuzzy_factor= 2.);

   iterator begin() { return iterator(*this); };
   iterator end() {return iterator(*this, true); }

private:
   void update_prototypes(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>& partition, const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>& features);
   void update_partition(boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>& partition, const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>& features);

private:
   unsigned int _clusters;
   unsigned int _p;
   unsigned int _decay_size;
   boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> _prototypes_wsum;
   boost::numeric::ublas::vector<double> _wsum;
   boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> _prototypes_eigenvalues;
   boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> _prototypes_eigenvectors;
   input_function& _input_fn;
   double       _min_prototypes_delta;
   gsl_rng*     _rng; 
   double       _fuzzy_factor;
};

#endif
