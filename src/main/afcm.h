#ifndef _AFCM_H
#define _AFCM_H

#include <gsl/gsl_rng.h>
#include <boost/numeric/ublas/fwd.hpp>

// Adaptive Fuzzy C-Means Clustering
// Implemented based on approach in 
// "Image Segmentation Based On Adaptive Cluster Prototype Estimation"
// by A. Liew and H. Yan

void afcm(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X,
         boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V,
         boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U,
         gsl_rng *rng,
         unsigned int& itr,
         unsigned int width,
         double knot_frequency = 0.2,
         double smoothing = 1.,
         unsigned int c = 3,
         double m = 2,
         double epsilon = 0.001,
         unsigned int max_itr = 100);

#endif
