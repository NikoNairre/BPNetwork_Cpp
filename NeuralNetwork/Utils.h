#pragma once
#include <vector>
#include <cmath>
#include <armadillo>
#include "../Cloud.h"

using std::vector;

struct Sample			//����
{
	arma::vec feature, label;				//���������������ͱ�ǩ����
	Sample(const arma::vec& feature, const arma::vec& label) {	//���캯��
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
	static double sigmoid(double x) {		//����static�Ļ���sigmoid�غ�main��ϵͳ���еĺ�����ͻ��ʹ��static�ؼ���ʹ�������ڵ�ǰ�ļ���Ч
		return 1.0 / (1 + std::exp(-x));
	}

	//��csv�ļ����ص�һ�����������У���0�����������ɹ���ʱ�̵���Ϣ��������ͼʹ��
	void LoadCsvFile(arma::imat& aim, std::string file_name);	
	std::vector<int> armaivecToVector(arma::ivec simple);
	void GenerateSrcData(CloudData& aim, arma::imat sourceMat);

	//����������ʱ�̵����ݺϲ���һ��ȥ���ƺ�����������Ϣ���������Ƶĺ����Ϊ�����������ѵ�����Ͳ��Լ�
	void GetIntCloudThick(arma::imat& aim, arma::ivec thick1, arma::ivec thick2, arma::ivec thick3);
	arma::mat Normalize(arma::imat source);				//���ݹ�һ������һ��������ݲſ��Ե������ݼ�ʹ��

	//��������ѵ���������Լ�ѵ����ǩ�����ش���õ����ݼ���(��������ݱ����ǹ�һ�����)
	std::vector<Sample> GetTrainData(arma::mat trainData, arma::mat trainLabel);

	//������Լ�Ԥ���׼ȷ��
	double CalCorrectRate(arma::vec prdedictLabel, arma::vec testLabel);
}

