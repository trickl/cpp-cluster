#ifndef _CLUSTERS_VISUAL_H
#define _CLUSTERS_VISUAL_H

#include <draw/drawing_pad.h>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

// Visualisation of clusters for the drawing pad

class clusters_visual : public drawable
{
public:
   clusters_visual(const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &X, // Points
           const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &V, // Prototypes
           const boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &U, // Memberships
           const std::vector<boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> > &EL);

   void draw();

private:
   boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> X; // points
   boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> V; // prototypes
   boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> U; // cluster memberships
   std::vector<boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> > EL; // prototype shapes
};

#endif
