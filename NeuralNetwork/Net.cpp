#include "Net.h"
#include <random>
using std::vector;

BPNet::BPNet()
{
	std::mt19937 rd;	//mt19937是一种预定义的随机数算法
	rd.seed(std::random_device()());	//选择随机数引擎
	std::uniform_real_distribution<double> distribution(-1, 1);		//产生(-1,1)之间的实数，满足均匀分布

	//初始化输入层
	for (int i = 0; i < Config::InNode; i++) {
		inputLayer[i] = new Node(Config::HiddenNode);	//每个输入层上的结点和隐层所有点都有连接
		for (int j = 0; j < Config::HiddenNode; j++) {
			inputLayer[i]->weight[j] = distribution(rd);	//产生随机数初始化输入层~隐层边权
			inputLayer[i]->weight_delta[j] = 0;				//相关权值改变量初始化为0
		}
	}	//由于输入层用不到偏置值，因此输入层的偏置值可以不用初始化

	//初始化隐层
	for (int j = 0; j < Config::HiddenNode; j++) {
		hiddenLayer[j] = new Node(Config::OutNode);

		//初始化偏置值
		hiddenLayer[j]->bias = distribution(rd);
		hiddenLayer[j]->bias_delta = 0;

		//初始化隐层~输出层边权
		for (int k = 0; k < Config::OutNode; k++) {
			hiddenLayer[j]->weight[k] = distribution(rd);
			hiddenLayer[j]->weight_delta[k] = 0;
		}
	}

	//初始化输出层
	for (int k = 0; k < Config::OutNode; k++) {
		outputLayer[k] = new Node(0);		//输出层无需考虑到下一层的边权
		
		//输出层只需要初始化偏置值即可
		outputLayer[k]->bias = distribution(rd);
		outputLayer[k]->bias_delta = 0;
	}
}


void BPNet::grad_zero()
{
	//把输入层~隐层权值改变量weight_delta(不是权值weight)清零
	for (auto inputLayerNode : inputLayer) {
		//用vector.assign()一次性清零所有边
		inputLayerNode->weight_delta.assign(inputLayerNode->weight_delta.size(), 0);	
	}

	//把隐层~输出层权值改变量清零，把隐层偏置值改变量清零
	for (auto hiddenLayerNode : hiddenLayer) {
		hiddenLayerNode->bias_delta = 0;
		hiddenLayerNode->weight_delta.assign(hiddenLayerNode->weight_delta.size(), 0);
	}

	//把输出层偏置值改变量清零(输出层不必考虑下一层边权)
	for (auto outputLayerNode : outputLayer) {
		outputLayerNode->bias_delta = 0;
	}
}


void BPNet::forward()
{
	//输入层->隐层
	for (int j = 0; j < Config::HiddenNode; j++) {		//算隐层每个结点的值
		double input_sum = 0;							//记录f(x) = sigmoid(x)的x值
		for (int i = 0; i < Config::InNode; i++) {
			input_sum += inputLayer[i]->value * inputLayer[i]->weight[j];	//不断累加输入层的值*边权值
		}
		input_sum -= hiddenLayer[j]->bias;				//所有输入层计算玩后，还需减去当前结点的偏置值
		hiddenLayer[j]->value = Utils::sigmoid(input_sum);		//利用sigmoid函数计算当前隐层结点的值
	}

	//隐层->输出层，步骤和上面类似
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
	//利用平方差计算误差
	for (int k = 0; k < label.n_cols; k++) {
		loss += (outputLayer[k]->value - label(k)) * (outputLayer[k]->value - label(k));
	}
	loss /= 2;
	return loss;
}


void BPNet::backward(arma::vec& label)
{
	//BP误差逆传播算法，本函数内实现时暂不对学习率和样本数进行加入计算，相关计算在后续revise函数中实现
	//计算输出层偏置改变量(从输出层向前，不能颠倒顺序)
	for (int k = 0; k < Config::OutNode; k++) {
		//梯度下降法，推出输出层bias_delta的公式
		double bias_delta = -(label(k) - outputLayer[k]->value) * outputLayer[k]->value
			* (1.0 - outputLayer[k]->value);
		outputLayer[k]->bias_delta += bias_delta;		//暂时先把所有delta累加起来，在revise函数中会乘上学习率，除以样本数取均值
	}

	//计算隐层~输出层边权改变值
	for (int j = 0; j < Config::HiddenNode; j++) {
		for (int k = 0; k < Config::OutNode; k++) {
			//公式可类似用梯度下降法推导
			double weight_delta = (label(k) - outputLayer[k]->value) * outputLayer[k]->value
				* (1.0 - outputLayer[k]->value) * hiddenLayer[j]->value;
			hiddenLayer[j]->weight_delta[k] += weight_delta;
		}
	}

	//计算隐层的偏置值改变值
	for (int j = 0; j < Config::HiddenNode; j++) {
		double bias_delta = 0;
		for (int k = 0; k < Config::OutNode; k++) {
			bias_delta += -(label(k) - outputLayer[k]->value) * outputLayer[k]->value
				* (1.0 - outputLayer[k]->value) * hiddenLayer[j]->weight[k];
		}
		//公式中该项与k项和无关，可以最后放到外面乘
		bias_delta *= hiddenLayer[j]->value * (1.0 - hiddenLayer[j]->value);
		hiddenLayer[j]->bias_delta += bias_delta;
	}

	//计算输入层~隐层边权改变值
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

	//更新输入层结点边权值
	for (int i = 0; i < Config::InNode; i++) {
		for (int j = 0; j < Config::HiddenNode; j++) {
			//每一轮改变一次，所以是+=
			inputLayer[i]->weight[j] += Config::lr * inputLayer[i]->weight_delta[j] / batch_size_double;
		}
	}

	//更新隐层的偏置值和结点边权值
	for (int j = 0; j < Config::HiddenNode; j++) {
		hiddenLayer[j]->bias += Config::lr * hiddenLayer[j]->bias_delta / batch_size_double;
		for (int k = 0; k < Config::OutNode; k++) {
			hiddenLayer[j]->weight[k] += Config::lr * hiddenLayer[j]->weight_delta[k] / batch_size_double;
		}
	}

	//更新输出层的偏置值
	for (int k = 0; k < Config::OutNode; k++) {
		outputLayer[k]->bias += Config::lr * outputLayer[k]->bias_delta / batch_size_double;
	}
}


bool BPNet::train(vector<Sample>& trainDataSet)
{
	for (int epoch = 0; epoch <= Config::max_epoch; epoch++) {	//循环增加训练轮数
		grad_zero();	//避免数据干扰，每一轮开始时清楚上一轮训练的各个改变量(它们已经在前一轮加入到了结果中)
		double max_loss = 0;	//记录最大损失

		for (Sample trainSample : trainDataSet) {	//每次训练，需要循环所有的测试样本
			//把每个样本的数据放入输入层
			for (int i = 0; i < Config::InNode; i++) {
				inputLayer[i]->value = trainSample.feature(i);
			}
			
			//对样本数据进行前向传播
			forward();

			//计算该样本的损失，并维护当前轮次训练的最大误差
			double loss = calculateLoss(trainSample.label);
			max_loss = std::max(max_loss, loss);

			//误差逆传播
			backward(trainSample.label);
		}

		//最大损失已经小于允许的最大值，说明训练成功，为防止过拟合，停止训练
		if (max_loss < Config::errEps) {
			std::cout << "Training Success.\n";
			printf("Finished in %d epochs.\n", epoch);
			std::cout << "Final training loss: " << max_loss << std::endl;
			return true;
		}
		else if (epoch % 100 == 0) {		//每100轮输出一次当前训练的进度
			printf("epoch %d, max_loss: %lf\n", epoch, max_loss);
		}

		//每一轮训练结束得到一个新的网络权值
		revise(trainDataSet.size());
	}

	std::cout << "Training Failed.\n";
	return false;
}


Sample BPNet::predict(arma::vec& feature)
{
	for (int i = 0; i < Config::InNode; i++) {		//传入样本数据
		inputLayer[i]->value = feature(i);
	}
	forward();										//前向传播
	arma::vec predictLabel(Config::OutNode);
	for (int k = 0; k < Config::OutNode; k++) {
		predictLabel(k) = outputLayer[k]->value;
	}
	Sample pred = Sample(feature, predictLabel);	//结果样本
	return pred;
}


vector<Sample> BPNet::predict(vector<Sample>& predictDataSet)
{
	std::vector<Sample> predSet;
	for (auto sample : predictDataSet) {
		Sample pres = predict(sample.feature);		//对每个样本逐一预测，预测完成后加入样本集合中
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

