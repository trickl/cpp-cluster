#ifndef _FCM_H
#define _FCM_H

#include <gsl/gsl_rng.h>
#include <boost/numeric/ublas/fwd.hpp>

// Fuzzy C-Means Clustering

void fcm(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X,
         boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V,
         boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U,
         gsl_rng *rng,
         unsigned int& itr,
         unsigned int c = 3,
         double m = 2,
         double epsilon = 0.001,
         unsigned int max_itr = 100);

#endif
