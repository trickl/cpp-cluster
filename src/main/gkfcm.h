#ifndef _GKFCM_H
#define _GKFCM_H

#include <vector>
#include <gsl/gsl_rng.h>
#include <boost/numeric/ublas/fwd.hpp>

/* Based on Gustafson Kessel FCM algorithm.
   X is feature matrix of size n x p
   c is the number of clusters to find
   m is the fuzzification factor
   epsilon is the stopping criterion for successive partitions
   U on exit is a fuzzy clustering partition matrix
   V are the cluster prototypes
   EL are the cluster covariance eigenvectors * eigenvalues */
void gkfcm(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X,
           boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V,
           boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U,
           std::vector<boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> > &EL,
           gsl_rng* rng,
           unsigned int& itr,
           unsigned int c = 3,
           double m = 2,
           double epsilon = 0.001,
           unsigned int max_itr = 100);

#endif
