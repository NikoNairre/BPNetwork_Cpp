#include "Net.h"
#include <random>
using std::vector;

BPNet::BPNet()
{
	std::mt19937 rd;	//mt19937��һ��Ԥ�����������㷨
	rd.seed(std::random_device()());	//ѡ�����������
	std::uniform_real_distribution<double> distribution(-1, 1);		//����(-1,1)֮���ʵ����������ȷֲ�

	//��ʼ�������
	for (int i = 0; i < Config::InNode; i++) {
		inputLayer[i] = new Node(Config::HiddenNode);	//ÿ��������ϵĽ����������е㶼������
		for (int j = 0; j < Config::HiddenNode; j++) {
			inputLayer[i]->weight[j] = distribution(rd);	//�����������ʼ�������~�����Ȩ
			inputLayer[i]->weight_delta[j] = 0;				//���Ȩֵ�ı�����ʼ��Ϊ0
		}
	}	//����������ò���ƫ��ֵ�����������ƫ��ֵ���Բ��ó�ʼ��

	//��ʼ������
	for (int j = 0; j < Config::HiddenNode; j++) {
		hiddenLayer[j] = new Node(Config::OutNode);

		//��ʼ��ƫ��ֵ
		hiddenLayer[j]->bias = distribution(rd);
		hiddenLayer[j]->bias_delta = 0;

		//��ʼ������~������Ȩ
		for (int k = 0; k < Config::OutNode; k++) {
			hiddenLayer[j]->weight[k] = distribution(rd);
			hiddenLayer[j]->weight_delta[k] = 0;
		}
	}

	//��ʼ�������
	for (int k = 0; k < Config::OutNode; k++) {
		outputLayer[k] = new Node(0);		//��������迼�ǵ���һ��ı�Ȩ
		
		//�����ֻ��Ҫ��ʼ��ƫ��ֵ����
		outputLayer[k]->bias = distribution(rd);
		outputLayer[k]->bias_delta = 0;
	}
}


void BPNet::grad_zero()
{
	//�������~����Ȩֵ�ı���weight_delta(����Ȩֵweight)����
	for (auto inputLayerNode : inputLayer) {
		//��vector.assign()һ�����������б�
		inputLayerNode->weight_delta.assign(inputLayerNode->weight_delta.size(), 0);	
	}

	//������~�����Ȩֵ�ı������㣬������ƫ��ֵ�ı�������
	for (auto hiddenLayerNode : hiddenLayer) {
		hiddenLayerNode->bias_delta = 0;
		hiddenLayerNode->weight_delta.assign(hiddenLayerNode->weight_delta.size(), 0);
	}

	//�������ƫ��ֵ�ı�������(����㲻�ؿ�����һ���Ȩ)
	for (auto outputLayerNode : outputLayer) {
		outputLayerNode->bias_delta = 0;
	}
}


void BPNet::forward()
{
	//�����->����
	for (int j = 0; j < Config::HiddenNode; j++) {		//������ÿ������ֵ
		double input_sum = 0;							//��¼f(x) = sigmoid(x)��xֵ
		for (int i = 0; i < Config::InNode; i++) {
			input_sum += inputLayer[i]->value * inputLayer[i]->weight[j];	//�����ۼ�������ֵ*��Ȩֵ
		}
		input_sum -= hiddenLayer[j]->bias;				//��������������󣬻����ȥ��ǰ����ƫ��ֵ
		hiddenLayer[j]->value = Utils::sigmoid(input_sum);		//����sigmoid�������㵱ǰ�������ֵ
	}

	//����->����㣬�������������
	for (int k = 0; k < Config::OutNode; k++) {
		double hidden_sum = 0;
		for (int j = 0; j < Config::HiddenNode; j++) {
			hidden_sum += hiddenLayer[j]->value * hiddenLayer[j]->weight[k];
		}
		hidden_sum -= outputLayer[k]->bias;
		outputLayer[k]->value = Utils::sigmoid(hidden_sum);
	}
}


double BPNet::calculateLoss(arma::vec& label)
{
	double loss = 0;
	//����ƽ����������
	for (int k = 0; k < label.n_cols; k++) {
		loss += (outputLayer[k]->value - label(k)) * (outputLayer[k]->value - label(k));
	}
	loss /= 2;
	return loss;
}


void BPNet::backward(arma::vec& label)
{
	//BP����洫���㷨����������ʵ��ʱ�ݲ���ѧϰ�ʺ����������м�����㣬��ؼ����ں���revise������ʵ��
	//���������ƫ�øı���(���������ǰ�����ܵߵ�˳��)
	for (int k = 0; k < Config::OutNode; k++) {
		//�ݶ��½������Ƴ������bias_delta�Ĺ�ʽ
		double bias_delta = -(label(k) - outputLayer[k]->value) * outputLayer[k]->value
			* (1.0 - outputLayer[k]->value);
		outputLayer[k]->bias_delta += bias_delta;		//��ʱ�Ȱ�����delta�ۼ���������revise�����л����ѧϰ�ʣ�����������ȡ��ֵ
	}

	//��������~������Ȩ�ı�ֵ
	for (int j = 0; j < Config::HiddenNode; j++) {
		for (int k = 0; k < Config::OutNode; k++) {
			//��ʽ���������ݶ��½����Ƶ�
			double weight_delta = (label(k) - outputLayer[k]->value) * outputLayer[k]->value
				* (1.0 - outputLayer[k]->value) * hiddenLayer[j]->value;
			hiddenLayer[j]->weight_delta[k] += weight_delta;
		}
	}

	//���������ƫ��ֵ�ı�ֵ
	for (int j = 0; j < Config::HiddenNode; j++) {
		double bias_delta = 0;
		for (int k = 0; k < Config::OutNode; k++) {
			bias_delta += -(label(k) - outputLayer[k]->value) * outputLayer[k]->value
				* (1.0 - outputLayer[k]->value) * hiddenLayer[j]->weight[k];
		}
		//��ʽ�и�����k����޹أ��������ŵ������
		bias_delta *= hiddenLayer[j]->value * (1.0 - hiddenLayer[j]->value);
		hiddenLayer[j]->bias_delta += bias_delta;
	}

	//���������~�����Ȩ�ı�ֵ
	for (int i = 0; i < Config::InNode; i++) {
		for (int j = 0; j < Config::HiddenNode; j++) {
			double weight_delta = 0;
			for (int k = 0; k < Config::OutNode; k++) {
				weight_delta += (label(k) - outputLayer[k]->value) * outputLayer[k]->value
					* (1.0 - outputLayer[k]->value) * hiddenLayer[j]->weight[k];
			}
			weight_delta *= hiddenLayer[j]->value * (1.0 - hiddenLayer[j]->value)
				* inputLayer[i]->value;
			inputLayer[i]->weight_delta[j] += weight_delta;
		}
	}
}


void BPNet::revise(int batch_size)
{
	double batch_size_double = (double)batch_size * 1.0;

	//������������Ȩֵ
	for (int i = 0; i < Config::InNode; i++) {
		for (int j = 0; j < Config::HiddenNode; j++) {
			//ÿһ�ָı�һ�Σ�������+=
			inputLayer[i]->weight[j] += Config::lr * inputLayer[i]->weight_delta[j] / batch_size_double;
		}
	}

	//���������ƫ��ֵ�ͽ���Ȩֵ
	for (int j = 0; j < Config::HiddenNode; j++) {
		hiddenLayer[j]->bias += Config::lr * hiddenLayer[j]->bias_delta / batch_size_double;
		for (int k = 0; k < Config::OutNode; k++) {
			hiddenLayer[j]->weight[k] += Config::lr * hiddenLayer[j]->weight_delta[k] / batch_size_double;
		}
	}

	//����������ƫ��ֵ
	for (int k = 0; k < Config::OutNode; k++) {
		outputLayer[k]->bias += Config::lr * outputLayer[k]->bias_delta / batch_size_double;
	}
}


bool BPNet::train(vector<Sample>& trainDataSet)
{
	for (int epoch = 0; epoch <= Config::max_epoch; epoch++) {	//ѭ������ѵ������
		grad_zero();	//�������ݸ��ţ�ÿһ�ֿ�ʼʱ�����һ��ѵ���ĸ����ı���(�����Ѿ���ǰһ�ּ��뵽�˽����)
		double max_loss = 0;	//��¼�����ʧ

		for (Sample trainSample : trainDataSet) {	//ÿ��ѵ������Ҫѭ�����еĲ�������
			//��ÿ�����������ݷ��������
			for (int i = 0; i < Config::InNode; i++) {
				inputLayer[i]->value = trainSample.feature(i);
			}
			
			//���������ݽ���ǰ�򴫲�
			forward();

			//�������������ʧ����ά����ǰ�ִ�ѵ����������
			double loss = calculateLoss(trainSample.label);
			max_loss = std::max(max_loss, loss);

			//����洫��
			backward(trainSample.label);
		}

		//�����ʧ�Ѿ�С����������ֵ��˵��ѵ���ɹ���Ϊ��ֹ����ϣ�ֹͣѵ��
		if (max_loss < Config::errEps) {
			std::cout << "Training Success.\n";
			printf("Finished in %d epochs.\n", epoch);
			std::cout << "Final training loss: " << max_loss << std::endl;
			return true;
		}
		else if (epoch % 100 == 0) {		//ÿ100�����һ�ε�ǰѵ���Ľ���
			printf("epoch %d, max_loss: %lf\n", epoch, max_loss);
		}

		//ÿһ��ѵ�������õ�һ���µ�����Ȩֵ
		revise(trainDataSet.size());
	}

	std::cout << "Training Failed.\n";
	return false;
}


Sample BPNet::predict(arma::vec& feature)
{
	for (int i = 0; i < Config::InNode; i++) {		//������������
		inputLayer[i]->value = feature(i);
	}
	forward();										//ǰ�򴫲�
	arma::vec predictLabel(Config::OutNode);
	for (int k = 0; k < Config::OutNode; k++) {
		predictLabel(k) = outputLayer[k]->value;
	}
	Sample pred = Sample(feature, predictLabel);	//�������
	return pred;
}


vector<Sample> BPNet::predict(vector<Sample>& predictDataSet)
{
	std::vector<Sample> predSet;
	for (auto sample : predictDataSet) {
		Sample pres = predict(sample.feature);		//��ÿ��������һԤ�⣬Ԥ����ɺ��������������
		predSet.push_back(sample);
	}
	return predSet;
}

void BPNet::testing(vector<Sample>& testSample)
{
	grad_zero();
	for (Sample s : testSample) {
		for (int i = 0; i < Config::InNode; i++) {
			inputLayer[i]->value = s.feature(i);
		}
		forward();
		double ls = calculateLoss(s.label);
		std::cout << ls << std::endl;
		backward(s.label);
		revise(testSample.size());
	}
}

