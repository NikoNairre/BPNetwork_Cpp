#pragma once
#include <armadillo>
class CloudData
{
public:
	arma::ivec Time;	//观测时间

	arma::ivec Bottom1;	//第一层云的底部高度，顶部高度和厚度
	arma::ivec Top1;
	arma::ivec Thick1;

	arma::ivec Bottom2;	//第二层云
	arma::ivec Top2;
	arma::ivec Thick2;

	arma::ivec Bottom3;	//第三层云
	arma::ivec Top3;
	arma::ivec Thick3;
};