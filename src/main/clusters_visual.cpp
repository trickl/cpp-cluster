#include "clusters_visual.h"
#include <stdexcept>
#include <iostream>
#include <image/color.h>
#include <image/geometric_transform.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <GL/gl.h>

using namespace boost::numeric::ublas;

clusters_visual::clusters_visual(const matrix<double, column_major> &X,
                 const matrix<double, column_major> &V,
                 const matrix<double, column_major> &U,
                 const std::vector<matrix<double, column_major> > &EL)
{
   if (X.size2() != 2) throw std::invalid_argument("X must have 2 dimensions for drawing");
   if (V.size2() != 2 && (V.size1() != 0 && V.size2() != 0)) throw std::invalid_argument("V must have 2 dimensions for plotting");
   if (!EL.empty() && EL.size() != std::vector<matrix<double, column_major> >::size_type(V.size1())) throw std::invalid_argument("EL must be the same size as V or zero");

   // Deep copy X, as I can't get the clever temporary system to link :P
   this->X = X;
   this->V = V;
   this->U = U;
   this->EL = EL;
}

void clusters_visual::draw()
{
   // Prototypes
   for (unsigned int i = 0; i < V.size1(); i++)
   {
      glColor3f(1.0, 0.0, 0.0);		
      glBegin(GL_POINTS);
      glVertex2d(V(i, 0), V(i, 1));
      glEnd();

      // Axis transformation matrix (for ellipses)
      matrix<double, column_major> T = identity_matrix<double>(2);
      if (!EL.empty()) // Elliptical prototype
      {
         T(0, 0) = EL[i](0, 0);
         T(1, 0) = EL[i](1, 0);
         T(0, 1) = EL[i](0, 1);
         T(1, 1) = EL[i](1, 1);
      } 

      // Eigenvectors (n.b supplied in columns)
      glColor3f(0.0, 0.0, 1.0);		
      glBegin(GL_LINE_STRIP);			
      glVertex2d(V(i, 0), V(i, 1));
      glVertex2d(V(i, 0) + T(0, 0), V(i, 1) + T(1, 0));
      glVertex2d(V(i, 0), V(i, 1));
      glVertex2d(V(i, 0) + T(0, 1), V(i, 1) + T(1, 1));
      glEnd();

      // Elliptical shapes
      glColor3f(1.0, 0.0, 0.0);		
      glBegin(GL_LINE_STRIP);			
      for(double angle=0.0; angle <= (2.0 * 3.14159); angle+=0.01)
      {		
         double x = V(i, 0) + (T(0, 0) * cos(angle) + T(0, 1) * sin(angle));
         double y = V(i, 1) + (T(1, 0) * cos(angle) + T(1, 1) * sin(angle));
         glVertex2d(x, y);
      }
      glEnd();
   }

   vector<double> prototype_hues(V.size1());
   interpolate_hues(prototype_hues);

   // Points 
   for (unsigned int i = 0; i < X.size1(); i++)
   {
      // Calculate hue using HSV model
      double hue = memberships_to_hue(prototype_hues, matrix_row<const matrix<double, column_major> >(U, i));
      double hsv[3] = {hue, 1., 1.};
      double rgb[3] = {0., 0., 0.};
      hsv2rgb(hsv, rgb);

      glBegin(GL_POINTS);
      glColor4f(rgb[0], rgb[1], rgb[2], 0.5); // Alpha is 0.5
      glVertex2f(X(i, 0), X(i, 1));
      glEnd();
   }
}
