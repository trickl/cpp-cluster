#ifndef _VALIDATION_H
#define _VALIDATION_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>

// Want to maximise the partition coefficient (separation of clusters)
double partition_coeff(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U);

// Minimize partition entropy 
double partition_entropy(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U);

// Minimize Xie-Beni index
double xie_beni_index(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U, const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V, const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X, double m = 2.0);

// Inclusion measure
void inclusion_measure(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U, boost::numeric::ublas::symmetric_matrix<double, boost::numeric::ublas::lower, boost::numeric::ublas::column_major> &I);

#endif
