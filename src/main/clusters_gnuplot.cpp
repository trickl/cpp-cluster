#include "clusters_gnuplot.h"
#include <image/color.h>
#include <fstream>
#include <iostream/manipulators/csv.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace boost::numeric::ublas;

void clusters_gnuplot(const char *filename,
           const boost::numeric::ublas::matrix<double, column_major> &X,
           const boost::numeric::ublas::matrix<double, column_major> &V,
           const boost::numeric::ublas::matrix<double, column_major> &U,
           const std::vector<boost::numeric::ublas::matrix<double, column_major> > &EL)
{
   vector<double> prototype_hues(V.size1());
   interpolate_hues(prototype_hues);

   vector<double> hues(U.size1());
   for (unsigned int i = 0; i < U.size1(); ++i)
   {
      hues(i) = memberships_to_hue(prototype_hues, matrix_row<const matrix<double, column_major> >(U, i));
   }

   // Write the data file in the format x, y, hue or x, y, z, hue
   // TODO: Capacity for plotting prototypes shapes (using V & EL).
   matrix<double, column_major> GP(X.size1(), X.size2() + 1);
   matrix_range<matrix<double, column_major> >(GP, range(0, X.size1()), range(0, X.size2())) = X;
   matrix_column<matrix<double, column_major> >(GP, X.size2()) = hues;

   std::ofstream fout(filename);
   fout << csv << GP;
}

