#pragma once
#include <vector>
#include <cmath>
#include <armadillo>
#include "../Cloud.h"

using std::vector;

struct Sample			//样本
{
	arma::vec feature, label;				//样本有特征向量和标签向量
	Sample(const arma::vec& feature, const arma::vec& label) {	//构造函数
		this->feature = feature;
		this->label = label;
	}
	Sample() = default;
	void display() {
		std::cout << "features:\n";
		feature.t().brief_print();
		std::cout << "label: \n";
		label.t().brief_print();
		std::cout << std::endl;
	}
};

namespace Utils
{
	static double sigmoid(double x) {		//不加static的话，sigmoid回和main中系统已有的函数冲突，使用static关键字使得它仅在当前文件有效
		return 1.0 / (1 + std::exp(-x));
	}

	//把csv文件加载到一个整数矩阵中，第0列用整数生成关于时刻的信息，方便作图使用
	void LoadCsvFile(arma::imat& aim, std::string file_name);	
	std::vector<int> armaivecToVector(arma::ivec simple);
	void GenerateSrcData(CloudData& aim, arma::imat sourceMat);

	//把所有有云时刻的数据合并到一起，去掉云厚度外的其他信息，用三层云的厚度作为后续神经网络的训练集和测试集
	void GetIntCloudThick(arma::imat& aim, arma::ivec thick1, arma::ivec thick2, arma::ivec thick3);
	arma::mat Normalize(arma::imat source);				//数据归一化，归一化后的数据才可以当作数据集使用

	//输入用于训练的数据以及训练标签，返回打包好的数据集合(传入的数据必须是归一化后的)
	std::vector<Sample> GetTrainData(arma::mat trainData, arma::mat trainLabel);

	//计算测试集预测的准确率
	double CalCorrectRate(arma::vec prdedictLabel, arma::vec testLabel);
}

