#ifndef _CLUSTERS_GNUPLOT_H
#define _CLUSTERS_GNUPLOT_H

#include <vector>
#include <string>
#include <boost/numeric/ublas/fwd.hpp>

// gplot data file generator
void clusters_gnuplot(const char *filename,
       const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X,
       const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V,
       const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U,
       const std::vector<boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> > &EL);

#endif
