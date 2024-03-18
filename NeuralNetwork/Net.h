#pragma once
#include "Config.h"
#include "Utils.h"
#include <iostream>

struct Node				//神经元
{
	double value = 0, bias = 0, bias_delta = 0;		//神经元的值，偏置值和偏置值的改变量
	std::vector<double> weight, weight_delta;		//存放神经元的所有出边的权值以及权值改变量
	Node(int nextLayerSize) {
		weight.resize(nextLayerSize);				//根据下一层连接神经元的数量设置
		weight_delta.resize(nextLayerSize);
	}
};

class BPNet
{
private:
	Node* inputLayer[Config::InNode]{};			//BP神经网络所有的三层神经元，在堆区动态分配数组空间存放
	Node* hiddenLayer[Config::HiddenNode]{};
	Node* outputLayer[Config::OutNode]{};

	//将网络中所有的weight_delta,bias_delta全部清零
	void grad_zero();

	//前向传播
	void forward();

	//计算损失
	double calculateLoss(arma::vec& label);

	//BP误差逆传播
	void backward(arma::vec& label);

	//利用BP误差向前更新神经网络结点的权值，偏置值
	void revise(int batch_size);

public:

	//初始化神经网络，给网络中所有层结点的偏置值，边权随机初始化
	BPNet();

	//传入数据集，开始训练
	bool train(vector<Sample>& trainDataSet);

	//单个样本的预测
	Sample predict(arma::vec& feature);

	//样本测试集合的预测
	vector<Sample> predict(vector<Sample>& predictDataSet);

	void testing(vector<Sample>& testSample);
};