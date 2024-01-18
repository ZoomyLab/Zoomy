#include "../../.tmp/Model_1/model_code.h"

#include <iostream>

// settings struct
// load settings
	// requires a save function on the python side
// load_ic
	// requires a save function on the python side

// problem: boundary conditions:
	// can I get them as sympy?





int main(int argc, char** argv) {

  std::cout << "MAIN" << std::endl;

	// Read command line arguments 
	//  if (argc != 4){
	//   std::cerr << RERROR "The program is run as: ./nprogram inputFolder/ outputFolder/ Nthreads" << std::endl;
	// 	  return 0;
	// }

	// {
	// SERGHEI serghei;
	
	// serghei.inFolder = argv[1];
	// serghei.outFolder = argv[2];

	// serghei.par.nthreads = atoi(argv[3]);
	
	// if(!serghei.start(argc, argv)) return 0;
	// if(!serghei.compute()) return 0;
	// if(!serghei.finalise()) return 0;
  
	// } // scope guard required to ensure serghei destructor is called

	// Kokkos::finalize();
 //  MPI_Finalize();

	return 1;
}
