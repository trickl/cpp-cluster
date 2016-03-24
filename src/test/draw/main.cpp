#include <iostream>
#include <fstream>
#include <vector>
#include <exception>
#include <getopt.h>
#include <iostream/manipulators/csv.h>
#include <cluster/clusters_visual.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <Fl/Fl.H>

using namespace boost::numeric::ublas;

struct options
{
   std::string dir;
   std::string points_file;
   std::string prototypes_file;
   std::string eigenvectors_file;
};

options parse_opt(int argc, char** argv);

// Reads a point list and plots in a window
int main(int argc, char** argv)
{
   options opts = parse_opt(argc, argv);

   std::ifstream points((opts.dir + opts.points_file).c_str());
   std::ifstream prototypes((opts.dir + opts.prototypes_file).c_str());
   std::ifstream eigenvectors((opts.dir + opts.eigenvectors_file).c_str());

   matrix<double, column_major> X, V, U, A_flat;

   points >> csv >> X;
   prototypes >> csv >> V;
   eigenvectors >> csv >> A_flat;

   U = scalar_matrix<double>(X.size1(), V.size1(), 0.33);

   // Convert eigenvectors
   std::vector<matrix<double, column_major> > A(V.size1());
   for (unsigned int k = 0; k < V.size1(); k++)
   {
      A[k].resize(2, 2);
      A[k](0, 0) = A_flat(k, 0);
      A[k](0, 1) = A_flat(k, 1);
      A[k](1, 0) = A_flat(k, 2);
      A[k](1, 1) = A_flat(k, 3);
   }

   try
   {
      clusters_visual c(X, V, U, A);
      drawing_pad pad;
      pad.set_scale(0, 0, 10, 10);
      pad.draw(&c);
      Fl::run();
   }
   catch (std::exception &e)
   {
      std::cerr << e.what() << std::endl;
   }
}

options parse_opt(int argc, char **argv)
{
   options opts = 
   {
      "/home/tgee/projects/cluster/data/", // directory
      "circular009.dat", // points filename
      "circular009-c.dat", // prototypes filename
      "circular009-e.dat" // prototypes filename
   };

   static struct option long_options[] =
   {
      /* These options don't set a flag.
        We distinguish them by their indices. */
      {"filename", required_argument, 0, 'f'},
      {"prototypes", required_argument, 0, 'p'},
      {"eigenvectors", required_argument, 0, 'e'},
      {"dir", required_argument, 0, 'd'},
      {0, 0, 0, 0}
   };

   while (1)
   {
      /* getopt_long stores the option index here. */
      int option_index = 0;
     
      int c = getopt_long (argc, argv, "d:f:p:e:",
                       long_options, &option_index);
     
      /* Detect the end of the options. */
      if (c == -1)
             break;
     
      std::stringstream argstream;
      switch (c)
      {
       case 0:
         /* If this option set a flag, do nothing else now. */
         if (long_options[option_index].flag != 0)
         break;

         // Long option with no short code equivalent
         std::cout << "Unhandled option: " << long_options[option_index].name << std::endl;
         if (optarg)
            std::cout << " with arg " << optarg << std::endl;
         break;
     
       case 'd':
         opts.dir = optarg;
         break;
     
       case 'f':
         opts.points_file = optarg;
         break;
     
       case 'p':
         opts.prototypes_file = optarg;
         break;

       case 'e':
         opts.eigenvectors_file = optarg;
         break;

       case '?':
         /* getopt_long already printed an error message. */
         break;
     
       default:
         abort ();
      }
   }
     
   /* Print any remaining command line arguments (not options). */
   if (optind < argc)
   {
       std::cout << "non-option ARGV-elements: ";
       while (optind < argc) std::cout << argv[optind++] << '\t';
       std::cout << std::endl;
   }

   return opts;
}

