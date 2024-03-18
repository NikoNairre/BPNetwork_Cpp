#include "Utils.h"
#include "../Cloud.h"
#include <cmath>

void Utils::LoadCsvFile(arma::imat& aim, std::string file_name)
{
	std::cout << "loading " + file_name << std::endl;
	aim.load(file_name, arma::csv_ascii);				//加载csv文件
	aim.shed_row(0);									//去除标题行
	for (int i = 0, idx = 0; i < aim.n_rows; i++) {		//把第0列的时间改成从0开始的数字
		aim(i, 0) = idx++;
	}
	aim.brief_print();
	std::cout << "load finished." << std::endl;
}


void Utils::GenerateSrcData(CloudData& aim, arma::imat sourceMat)
{
	aim.Time = sourceMat.col(0);

	aim.Bottom1 = sourceMat.col(1);
	aim.Top1 = sourceMat.col(2);
	aim.Thick1 = sourceMat.col(3);

	aim.Bottom2 = sourceMat.col(4);
	aim.Top2 = sourceMat.col(5);
	aim.Thick2 = sourceMat.col(6);

	aim.Bottom3 = sourceMat.col(7);
	aim.Top3 = sourceMat.col(8);
	aim.Thick3 = sourceMat.col(9);
}


std::vector<int> Utils::armaivecToVector(arma::ivec simple)
{
	std::vector<int> res;
	for (int i = 0; i < simple.n_rows; i++) {
		res.push_back(simple(i));
	}
	return res;
}

void Utils::GetIntCloudThick(arma::imat& aim, arma::ivec thick1, arma::ivec thick2, arma::ivec thick3)
{
	for (int i = 0; i < thick1.n_rows; i++) {
		if (thick1(i) or thick2(i) or thick3(i)) {
			arma::irowvec tempInsert = { thick1(i), thick2(i), thick3(i) };
			if (aim.n_rows == 0) {
				aim.insert_rows(0, tempInsert);
			}
			else {
				aim.insert_rows(aim.n_rows - 1, tempInsert);
			}
		}
	}
}

arma::mat Utils::Normalize(arma::imat source)
{
	arma::mat res(source.n_rows, source.n_cols, arma::fill::zeros);
	int maxThick = source.max();
	for (int i = 0; i < source.n_rows; i++) {
		for (int j = 0; j < source.n_cols; j++) {
			res(i, j) = (double)(log10(source(i, j))) / (double)log10(maxThick);	//对数函数归一化
			if (res(i, j) < 0) res(i, j) = 0;			//小于0的值，还有-inf，都认为是0
		}
	}
	return res;
}

std::vector<Sample> Utils::GetTrainData(arma::mat trainData, arma::mat trainLabel)
{
	vector<Sample>res;
	for (int i = 0; i < trainData.n_rows; i++) {
		Sample cu;
		arma::vec cu_feaure = trainData.row(i).t();			//每一行的数据是一个样本
		arma::vec cu_label = trainLabel.row(i).t();			//由于Sample的feature和label都是列向量，所以需要转置	
		cu.feature = cu_feaure;
		cu.label = cu_label;
		res.push_back(cu);								//样本“集合”
	}
	return res;
}


double Utils::CalCorrectRate(arma::vec prdedictLabel, arma::vec testLabel)
{
	int test_num = prdedictLabel.n_rows;
	int correct_num = 0;
	for (int i = 0; i < test_num; i++) {
		int lpre = 0;
		if (prdedictLabel(i) < 0.5) {			//<0.5则认为是少云，>0.5则认为是多云
			lpre = 0;
		}
		else {
			lpre = 1;
		}
		int ltest = 0;
		if (testLabel(i) < 0.5) {
			ltest = 0;
		}
		else {
			ltest = 1;
		}
		std::cout << "Predict: " << lpre << " RealLabel: " << ltest << std::endl;
		if (lpre == ltest) {
			correct_num++;
		}
	}
	return (double)(correct_num) * 1.0 / ((double)(test_num) * 1.0);
}

void test_dataProcess()
{
	arma::imat Data0715;
	std::string file0715 = "SourceData/Z_RADA_54511_20180715_P_YCCR_HTMW_CP.csv";
	Utils::LoadCsvFile(Data0715, file0715);
	Data0715.raw_print();
}


