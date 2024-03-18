#pragma once
#include <cuchar>

using std::size_t;

namespace Config
{
	const size_t InNode = 3;
	const size_t HiddenNode = 6;
	const size_t OutNode = 1;

	const double lr = 0.8;
	const double errEps = 1e-3;
	const double zeroEps = 1e-6;
	const size_t max_epoch = 1e6;
}