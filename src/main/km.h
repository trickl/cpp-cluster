#ifndef _KM_H
#define _KM_H

#include <boost/numeric/ublas/fwd.hpp>
#include <gsl/gsl_rng.h>

// K-Means Clustering

void km(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X,
         boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V,
         boost::numeric::ublas::matrix<unsigned int, boost::numeric::ublas::column_major> &U,
         gsl_rng *rng,
         unsigned int& itr,
         unsigned int c = 3,
         unsigned int max_itr = 100);

#endif
