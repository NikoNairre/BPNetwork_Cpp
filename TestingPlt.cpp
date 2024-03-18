#include <iostream>
#include <armadillo>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;


void try_Plt()
{
	plt::plot({1, 2, 3, 4});
	plt::show();
	
}