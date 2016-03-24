#ifndef _SCFCM_H
#define _SCFCM_H

#include <gsl/gsl_rng.h>
#include <boost/numeric/ublas/fwd.hpp>

// Fuzzy C-Means Clustering with Spatial Continuity
// Implemented based on approach in 
// "Image Segmentation Based On Adaptive Cluster Prototype Estimation"
// by A. Liew and H. Yan

void scfcm(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X,
         boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V,
         boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U,
         gsl_rng *rng,
         unsigned int& itr,
         unsigned int width,
         unsigned int c = 3,
         double m = 2,
         double epsilon = 0.001,
         unsigned int max_itr = 100);

#endif
