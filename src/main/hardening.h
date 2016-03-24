#ifndef _HARDENING_H
#define _HARDENING_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>

// Assign total membership to the partition with the highest fuzzy membership
void highest_membership(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U, boost::numeric::ublas::matrix<unsigned int> &T);

// Only assign one spatial region to each cluster
void highest_membership_with_spatial_constraint(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U, unsigned int width, boost::numeric::ublas::matrix<unsigned int> &T);

#endif
