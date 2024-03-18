#pragma once
#include "Config.h"
#include "Utils.h"
#include <iostream>

struct Node				//��Ԫ
{
	double value = 0, bias = 0, bias_delta = 0;		//��Ԫ��ֵ��ƫ��ֵ��ƫ��ֵ�ĸı���
	std::vector<double> weight, weight_delta;		//�����Ԫ�����г��ߵ�Ȩֵ�Լ�Ȩֵ�ı���
	Node(int nextLayerSize) {
		weight.resize(nextLayerSize);				//������һ��������Ԫ����������
		weight_delta.resize(nextLayerSize);
	}
};

class BPNet
{
private:
	Node* inputLayer[Config::InNode]{};			//BP���������е�������Ԫ���ڶ�����̬��������ռ���
	Node* hiddenLayer[Config::HiddenNode]{};
	Node* outputLayer[Config::OutNode]{};

	//�����������е�weight_delta,bias_deltaȫ������
	void grad_zero();

	//ǰ�򴫲�
	void forward();

	//������ʧ
	double calculateLoss(arma::vec& label);

	//BP����洫��
	void backward(arma::vec& label);

	//����BP�����ǰ�������������Ȩֵ��ƫ��ֵ
	void revise(int batch_size);

public:

	//��ʼ�������磬�����������в����ƫ��ֵ����Ȩ�����ʼ��
	BPNet();

	//�������ݼ�����ʼѵ��
	bool train(vector<Sample>& trainDataSet);

	//����������Ԥ��
	Sample predict(arma::vec& feature);

	//�������Լ��ϵ�Ԥ��
	vector<Sample> predict(vector<Sample>& predictDataSet);

	void testing(vector<Sample>& testSample);
};