#include "hardening.h"
#include <set>
#include <stack>
#include <algorithm>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace boost::numeric::ublas;

void highest_membership(const matrix<double, column_major> &U, matrix<unsigned int> &T)
{
   for (unsigned int i = 0; i < U.size1(); i++)
   {
      matrix_row<const matrix<double, column_major> > R(U, i);
      matrix_row<const matrix<double, column_major> >::const_iterator max_itr = std::max_element(R.begin(), R.end());
      matrix_row<matrix<unsigned int> >(T, i) = unit_vector<unsigned int>(U.size2(), max_itr.index());
   }
}

void highest_membership_with_spatial_constraint(const matrix<double, column_major> &U, unsigned int width, matrix<unsigned int> &T)
{
   // Start by assign those with highest memberships to a list
   unsigned int highest_membership[U.size1()];
   for (unsigned int i = 0; i < U.size1(); i++)
   {
      matrix_row<const matrix<double, column_major> > R(U, i);
      matrix_row<const matrix<double, column_major> >::const_iterator max_itr = std::max_element(R.begin(), R.end());
      highest_membership[i] = max_itr.index();
   }

   // REMOVE
   /*
   std::cout << "Highest Membership" << std::endl;
   for (unsigned int i = 0; i < U.size1(); i++)
   {
      std::cout << highest_membership[i];
      if ((i + 1) % width == 0) std::cout << std::endl;
      else std::cout << "\t";
   }   
*/

   // Now region grow each of these, merging regions
   unsigned int region_labels[U.size1()];
   std::vector<unsigned int> regions;
   int neighbour_offsets[] = {-1, 1, -width, width};

   std::set<unsigned int> pixels;
   for (unsigned int i = 0; i < T.size1(); ++i) pixels.insert(i);
   
   while (!pixels.empty())
   {
      std::stack<unsigned int> similar_neighbours;
      unsigned int current_pixel = *pixels.begin();
      region_labels[current_pixel] = regions.size();
      regions.push_back(highest_membership[current_pixel]);
      
      // Label each distinct region
      similar_neighbours.push(current_pixel);
      while (!similar_neighbours.empty())
      {
         current_pixel = similar_neighbours.top();
         similar_neighbours.pop();
         for (unsigned int j = 0; j < 4; ++j)
         {
            int neighbour_pixel = current_pixel + neighbour_offsets[j];
            if (pixels.count(neighbour_pixel) && highest_membership[current_pixel] == highest_membership[neighbour_pixel])
            {
               similar_neighbours.push(neighbour_pixel);
               region_labels[neighbour_pixel] = region_labels[current_pixel];
            }
         } 

         pixels.erase(current_pixel);
      } 
   }

   // REMOVE
   /*
   std::cout << "Region Labels" << std::endl;
   for (unsigned int i = 0; i < U.size1(); i++)
   {
      std::cout << region_labels[i];
      if ((i + 1) % width == 0) std::cout << std::endl;
      else std::cout << "\t";
   }   
   */
   
   // Find the region that have the highest cumulative cluster probability
   double cumulative_probability[regions.size()];
   std::fill(cumulative_probability, cumulative_probability + regions.size(), 0.);

   for (unsigned int i = 0; i < T.size1(); ++i)
   {
      unsigned int k = highest_membership[i];
      cumulative_probability[region_labels[i]] += U(i, k);
   } 

   double max_cumulative_probability[U.size2()]; 
   double probable_cluster_region[U.size2()]; 
   for (unsigned int r = 0; r < regions.size(); ++r)
   {
      unsigned int k = regions[r];
      if (cumulative_probability[r] > max_cumulative_probability[k])
      {
          max_cumulative_probability[k] = cumulative_probability[r];
          probable_cluster_region[k] = r;
      } 
   }

   // REMOVE
   /*
   std::cout << "Cumulative Probability" << std::endl;
   for (unsigned int r = 0; r < regions.size(); r++)
   {
      std::cout << max_cumulative_probability[r];
      if ((r + 1) % regions.size() == 0) std::cout << std::endl;
      else std::cout << "\t";
   }   
   */

   // REMOVE
   /*
   std::cout << "Most probable region" << std::endl;
   for (unsigned int k = 0; k < U.size2(); ++k)
   {
      std::cout << probable_cluster_region[k];
      if ((k + 1) % U.size2() == 0) std::cout << std::endl;
      else std::cout << "\t";
   }   
   */
   
   
   
   // Now assign regions using highest_membership, note that some regions will not be assigned to any cluster
   double background_threshold = 0.1;
   for (unsigned int i = 0; i < U.size1(); ++i)
   {
      vector<double> R = matrix_row<const matrix<double, column_major> >(U, i);

      //if (region_labels[i] != probable_cluster_region[highest_membership[i]]) R(highest_membership[i]) = 0.;
      for (unsigned int k = 0; k < U.size2(); ++k)
      {
      if (region_labels[i] != probable_cluster_region[k]) R(k) = 0.;
      }

      vector<double>::const_iterator max_itr = std::max_element(R.begin(), R.end());
      if (*max_itr < background_threshold)
      {
         // Background is not assigned to any cluster
         matrix_row<matrix<unsigned int> >(T, i) = zero_vector<unsigned int>(U.size2());
      }
      else
      { 
         matrix_row<matrix<unsigned int> >(T, i) = unit_vector<unsigned int>(U.size2(), max_itr.index());
      }
   }
}
