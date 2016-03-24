#ifndef _PARTITION_H
#define _PARTITION_H

#include <gsl/gsl_rng.h>
#include <boost/numeric/ublas/fwd.hpp>

// Randomize a partition
void random_partition(boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U, gsl_rng *rng);
void random_partition(boost::numeric::ublas::matrix<unsigned int, boost::numeric::ublas::column_major> &U, gsl_rng *rng);

#endif
