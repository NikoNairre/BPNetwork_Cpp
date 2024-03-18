#include <iostream>
#include <vector>
#include <armadillo>
#include <matplotlibcpp.h>
#include <Windows.h>
#include "NeuralNetwork/Utils.h"
#include "NeuralNetwork/Config.h"
#include "NeuralNetwork/Net.h"

//using namespace Utils;

namespace plt = matplotlibcpp;

arma::imat SourceData0715;
arma::imat SourceData0716;
arma::imat SourceData0717;
arma::imat SourceData0718;
arma::imat SourceData0719;

CloudData CloudData0715;
CloudData CloudData0716;
CloudData CloudData0717;
CloudData CloudData0718;
CloudData CloudData0719;

int main()
{
	Utils::LoadCsvFile(SourceData0715, "SourceData/Z_RADA_54511_20180715_P_YCCR_HTMW_CP.csv");
	Utils::LoadCsvFile(SourceData0716, "SourceData/Z_RADA_54511_20180716_P_YCCR_HTMW_CP.csv");
	Utils::LoadCsvFile(SourceData0717, "SourceData/Z_RADA_54511_20180717_P_YCCR_HTMW_CP.csv");
	Utils::LoadCsvFile(SourceData0718, "SourceData/Z_RADA_54511_20180718_P_YCCR_HTMW_CP.csv");
	Utils::LoadCsvFile(SourceData0719, "SourceData/Z_RADA_54511_20180719_P_YCCR_HTMW_CP.csv");
	//std::vector<int> t = armaivecToVector(SourceData0715.col(1));
	//plt::plot(t);
	//plt::show();
	Utils::GenerateSrcData(CloudData0715, SourceData0715);
	Utils::GenerateSrcData(CloudData0716, SourceData0716);
	Utils::GenerateSrcData(CloudData0717, SourceData0717);
	Utils::GenerateSrcData(CloudData0718, SourceData0718);
	Utils::GenerateSrcData(CloudData0719, SourceData0719);
	

	//plt::plot(Utils::armaivecToVector(CloudData0715.Top1));
	//plt::title("CloudData0715");
	//plt::xlabel("Time");
	//plt::ylabel("Top1");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0715.Top2));
	//plt::title("CloudData0715");
	//plt::xlabel("Time");
	//plt::ylabel("Top2");
	//plt::figure_size(50, 100);

	//plt::show();
	//plt::plot(Utils::armaivecToVector(CloudData0715.Top3));
	//plt::title("CloudData0715");
	//plt::xlabel("Time");
	//plt::ylabel("Top3");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0715.Thick1));
	//plt::title("CloudData0715");
	//plt::xlabel("Time");
	//plt::ylabel("Thick1");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0715.Thick2));
	//plt::title("CloudData0715");
	//plt::xlabel("Time");
	//plt::ylabel("Thick2");
	//plt::figure_size(50, 100);


	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0715.Thick3));
	//plt::title("CloudData0715");
	//plt::xlabel("Time");
	//plt::ylabel("Thick3");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0716.Bottom1));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Bottom1");
	//plt::figure_size(50, 100);

	//plt::show();


	//plt::plot(Utils::armaivecToVector(CloudData0716.Bottom2));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Bottom2");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0716.Bottom3));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Bottom3");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0716.Top1));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Top1");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0716.Top2));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Top2");
	//plt::figure_size(50, 100);

	//plt::show();
	//plt::plot(Utils::armaivecToVector(CloudData0716.Top3));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Top3");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0716.Thick1));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Thick1");
	//plt::figure_size(50, 100);

	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0716.Thick2));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Thick2");
	//plt::figure_size(50, 100);


	//plt::show();

	//plt::plot(Utils::armaivecToVector(CloudData0716.Thick3));
	//plt::title("CloudData0716");
	//plt::xlabel("Time");
	//plt::ylabel("Thick3");
	//plt::figure_size(50, 100);

	//plt::show();



	arma::imat IntThickTrain;
	Utils::GetIntCloudThick(IntThickTrain, CloudData0715.Thick1, CloudData0715.Thick2, CloudData0715.Thick3);
	Utils::GetIntCloudThick(IntThickTrain, CloudData0716.Thick1, CloudData0716.Thick2, CloudData0716.Thick3);
	Utils::GetIntCloudThick(IntThickTrain, CloudData0717.Thick1, CloudData0717.Thick2, CloudData0717.Thick3);
	Utils::GetIntCloudThick(IntThickTrain, CloudData0718.Thick1, CloudData0718.Thick2, CloudData0718.Thick3);
	//IntThickTrain.raw_print();

	arma::mat RawTrainData = Utils::Normalize(IntThickTrain);
	//RawTrainData.brief_print();

	//人为构造数据集
	//arma::vec TrainLabel(RawTrainData.n_rows);
	//for (int i = 0; i < RawTrainData.n_rows; i++) {
	//	double label_value = RawTrainData(i, 0) * 0.4 + RawTrainData(i, 1) * 0.35 + RawTrainData(i, 2) * 0.25;
	//	TrainLabel(i) = label_value;
	//	}
	//RawTrainData.save("DataSet/trainData.csv", arma::csv_ascii);
	//TrainLabel.save("DataSet/trainLabel.csv", arma::csv_ascii);
	arma::mat RawTrainLabel; RawTrainLabel.load("DataSet/trainLabel.csv");

	//RawTrainLabel.brief_print();
	std::cout << "Train Data:\n";
	std::vector<Sample> TrainData = Utils::GetTrainData(RawTrainData, RawTrainLabel);
	for (Sample x : TrainData) {
		x.display();
	}
	Sleep(5000);

	std::vector<Sample> testTrain;
	Sample s1, s2;
	s1.feature = { 0.4, 0.2, 0.5 };
	s1.label = { 0.3 };
	s2.feature = { 0.7, 0.2, 0.4 };
	s2.label = { 0.36 };
	testTrain.push_back(s1); testTrain.push_back(s2);


	//传入训练集进行神经网络模型训练
	BPNet* Net = new BPNet();
	Net->train(TrainData);
	Sleep(5000);

	arma::imat IntThickTest;
	Utils::GetIntCloudThick(IntThickTest, CloudData0719.Thick1, CloudData0719.Thick2, CloudData0719.Thick3);
	arma::mat RawTestData = Utils::Normalize(IntThickTest);
	//人为构造测试数据集
	//arma::vec TestLabel(RawTestData.n_rows);
	//for (int i = 0; i < RawTestData.n_rows; i++) {
	//	double label_value = RawTestData(i, 0) * 0.4 + RawTestData(i, 1) * 0.35 + RawTestData(i, 2) * 0.25;
	//	TestLabel(i) = label_value;
	//}
	//RawTestData.save("DataSet/testData.csv", arma::csv_ascii);
	//TestLabel.save("DataSet/testLabel.csv", arma::csv_ascii);
	arma::mat RawTestLabel; RawTestLabel.load("Dataset/testLabel.csv");
	std::vector<Sample> TestData = Utils::GetTrainData(RawTestData, RawTestLabel);
	std::cout << "Test Data:\n";
	//for (auto x : TestData) {
	//	x.display();
	//}
	//模型测试
	std::cout << "Start predicting.\n";
	std::vector<Sample> predictRes = Net->predict(TestData);
	for (Sample x : predictRes) {
		x.display();
	}

	arma::vec PredictLabel(predictRes.size());
	for (int i = 0; i < predictRes.size(); i++) {
		PredictLabel(i) = predictRes[i].label(0);
	}
	double correct_rate = Utils::CalCorrectRate(PredictLabel, RawTestLabel);
	std::cout << "Predict finished\n";
	std::cout << "Correct rate: " << correct_rate << std::endl;

	return 0;
}