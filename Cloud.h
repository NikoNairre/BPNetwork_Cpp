#pragma once
#include <armadillo>
class CloudData
{
public:
	arma::ivec Time;	//�۲�ʱ��

	arma::ivec Bottom1;	//��һ���Ƶĵײ��߶ȣ������߶Ⱥͺ��
	arma::ivec Top1;
	arma::ivec Thick1;

	arma::ivec Bottom2;	//�ڶ�����
	arma::ivec Top2;
	arma::ivec Thick2;

	arma::ivec Bottom3;	//��������
	arma::ivec Top3;
	arma::ivec Thick3;
};