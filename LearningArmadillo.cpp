#include <iostream>
#include <armadillo>
#include <iomanip>
//using namespace arma;

void show_DataTypes()
{
	//矩阵类型
	std::cout << "Part1: Data Types" << std::endl;
	arma::mat A(5, 5, arma::fill::randu);		//5*5大小，值在0-1之间的小数矩阵
	/*
		fill::ones
		fill::zeros
		fill::eye
		fill::value(10)		赋值成具体数值
		···  etc
	*/
	/*
		mat	 = 	Mat<double>
		dmat	 = 	Mat<double>
		fmat	 = 	Mat<float>
		cx_mat	 = 	Mat<cx_double>
		cx_dmat	 = 	Mat<cx_double>
		cx_fmat	 = 	Mat<cx_float>
		umat	 = 	Mat<unsigned int> or Mat<arma::uword>
		imat	 = 	Mat<int>		  or Mat<arma::sword>
	*/
	//since Armadillo 10.5, the elements are initialised to zero by default
	arma::Mat<int> testB(4, 3, arma::fill::value(7));	//4行3列的整数矩阵  //mat A = Mat<double> A

	A.print();				//直接打印出矩阵
	testB.brief_print();	//矩阵的大小信息也会被打印,但矩阵较大时不会全部输出
	std::cout << "A[4][4]: " << A(4, 4) << std::endl;	//打出矩阵具体位置的元素数值,括号内的下标从0开始

	arma::vec testC(10);	//列向量,不赋初值时默认为0  //vec A = Col<double> A
	testC(2) = 3.33;		//操作具体位置上的元素
	testC.brief_print();
	arma::vec colA = A.col(2);		//取出矩阵A的第3列构成列向量
	colA.print();

	arma::Row<double> testD(8, arma::fill::randu);	//行向量	 //rowvec A = Row<double> A
	testD(0) = 0;
	testD.brief_print();
	arma::rowvec rowA = A.row(3);	//取出矩阵A的第4行构成行向量
	rowA.print();

	arma::Cube<double> testE(3, 3, 3, arma::fill::randu);	//三维矩阵
	testE.brief_print();	//三维矩阵也可以完整打印

}

void show_Mat_operations()
{
	/*
		operators:  +  −  *  %  /  ==  !=  <=  >=  <  >  &&  ||
		Overloaded operators for Mat, Col, Row and Cube classes
	*/
	arma::mat A = { {1.13, 2.22}, {2.4, 1.57} };	//给矩阵直接初始化成一个具体的
	arma::mat B = { {2.08, 1.99}, {0.78, 2.1} };
	arma::Mat<double> addAB = A + B;				//两矩阵相加
	addAB.brief_print();
	arma::mat subAB = A - B;
	subAB.brief_print();
	arma::mat mulAB = A * B;						//矩阵乘法,(m*k)矩阵 * (k*n)矩阵 = (m*n)矩阵
	mulAB.brief_print();

	std::cout << (A==B) << std::endl;				//会比较A和B每个位置上的值是否相等，并且会全部输出
	std::cout << (A != B) << std::endl;				//会比较A和B每个位置上的值是否不相等，并且会全部输出
	//因此不能直接写if (A==B)来给两个矩阵判等
	A /= 0.5;											//矩阵和数字可以做乘除法
	arma::Mat<int> C = { {14, 22}, {7, 9} };
	arma::Mat<int> D = { {3, 5}, {7, 11} };
	C *= 2;
	A.brief_print();
	C.brief_print();
	C %= D;				//不是取余，是将矩阵C每个位置上的数字乘上D对应位置上的数
						// “/”符号将是将矩阵C每个位置上的数字除以D对应位置上的数
	//  ==,!=,>=,<=,&&,||都是矩阵中的每个对应位置的数字做相应运算
	C.brief_print();
}

void show_Mat_attributes_part1()
{
	arma::arma_rng::set_seed_random();		//随机数种子随机化

	arma::Mat<int> A(6, 6, arma::fill::value(3));
	A(0, 2) = A(0, 5) = A(1, 1) = A(2, 0) = A(2, 5) = A(4, 4) = A(5, 0) = A(5, 1) = A(5, 4) = 0;
	A.raw_print();
	std::cout << "size of A: " << A.n_rows << " * " << A.n_cols << std::endl;	//获取矩阵的行数和列数
	std::cout << arma::size(A) << std::endl;	//获取矩阵的大小	
	std::cout << A.n_elem << std::endl;		//获取矩阵A内部的元素的个数
	
	//获取矩阵具体位置上的值
	std::cout << A.at(2, 2) << " " << A(2, 2) << std::endl;		//两种方式都可以获取矩阵具体位置上的值
	arma::Cube<double> B(3, 3, 3, arma::fill::randu);			
	std::cout << B.at(1, 1, 1) << " " << B(1, 1, 1) << std::endl;	//三维矩阵可类似获取
	std::cout << A.row(1) << "---" << A.col(4) << std::endl;	//获取二维矩阵中的一行或一列
	//std::cout << B.row(1);		//三维矩阵也可以获取到

	arma::mat testA(2, 3, arma::fill::randu);
	testA.zeros();	testA.brief_print();	//将矩阵置为全0
	testA.ones(3, 3); testA.brief_print();	//将矩阵转成3*3全1矩阵
	testA.eye(3, 4); testA.brief_print();	//非方阵也能产生对角线是1的矩阵，但不是单位阵
	testA.eye(arma::size(A)); testA.brief_print();	//A的大小是6*6，也可传入(6, 6),但不能写成(6 * 6)
	//A.randu(),  A.randn()
	testA.fill(123.0);   testA.brief_print();// or:  mat testA(6, 6, fill::value(123.0));
	
	testA.randu(3, 3); 
	testA(0, 0) = arma::datum::eps;		//趋向于0，大于0的无穷小量
	testA(2, 2) = -arma::datum::eps;	//趋向于0，小于0的无穷小量
	testA.brief_print();
	testA.clean(arma::datum::eps);		//把无穷小量变成0
	testA.brief_print();

}

void show_Mat_attributes_part2()
{
	//矩阵的一些变换
	arma::imat A = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
	A.clamp(3, 6); A.brief_print();		//限制A中值的范围在3-6之间，<3的数会变成3，大于6的数会变成6
	A.transform([](int val) {return val + 1; }); A.brief_print();	//利用transform函数将矩阵中每个元素加1
	//transform需要传入一个函数，上面用了lambda函数实现(就像sort(...,...,[](){;})) 自写比较函数那样
	//A.set_size(5, 10);  //重新设置A的大小，但是初始化时没有赋值，与创建A时默认全是0是不一样的
	A.reshape(2, 3); A.brief_print();	//重新设置A的大小，未改动的位置值不变，若大小扩大，补0
	A.reshape(3, 2); A.brief_print();
	A.reshape(3, 3); A.brief_print();
	A(0, 2) = A.at(1, 2) = 4; A(2, 2) = 8;
	A.resize(3, 4); A.brief_print();	//也可以重新设置A的大小，和reshape有不同，会保存原来的值
	arma::imat B;
	B.copy_size(A); B.brief_print();	//Set the size to be the same as object A, but not initialized
	//A.reset(); A.brief_print();		//把A重新设置为大小0的矩阵，即为空
	
	//获取子矩阵
	arma::imat testA = A.cols(0, 1);   testA.brief_print();		//取出A的第0列到第1列
	//类似地也有A.rows()
	arma::mat C(10, 10, arma::fill::randu); C *= 10; C.print();
	arma::Mat<double> testC = C.submat(1, 3, 4, 4);		//获取C以第二行第三列为左上角，第五行第五列为右下角的子矩阵
	testC.print();

	//矩阵插入行，列
	arma::mat D(4, 4, arma::fill::randu);
	D.insert_rows(1, 2);	//在第2行前插入2行空行
	D.insert_cols(2, 1);	//在第3列前插入1列空列
	D.print();
	arma::vec vD(D.n_rows, arma::fill::value(1.14514));		//在D的第0列插入列向量vD
	//arma::mat vD(D.n_rows, 2, arma::fill::value(1.14514));	//插入矩阵也可
	//同理也可以插入行向量，或者在行间插入矩阵
	D.insert_cols(0, vD);
	D.insert_rows(6, 1);
	D.print();

	//矩阵删除行，列
	arma::mat E = D;
	E.shed_col(0);		//删除第一列
	E.shed_rows(1, 2);	//删除第二行到第三行
	//同理也有.shed_cols()和.shed_row(),用于删除多列和删除单行
	//针对三维矩阵有.shed_slice()
	E.brief_print();
	arma::uvec de = { 1, 3 };		//这里的类型需要是uvec,不能是ivec
	E.shed_cols(de);		//传入列向量参数，删除E的第2和第4列
	E.brief_print();

	//矩阵交换行，列
	arma::imat F(6, 6);
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			F(i, j) = i * 6 + j;
		}
	}
	F.print(); std::cout << std::endl;
	F.swap_cols(1, 5);		//交换矩阵的第2列和第6列
	F.swap_rows(2, 3);		//交换矩阵的第3行和第4行
	F.print();

	arma::imat T1 = { {1, 2}, {3, 4} };
	arma::imat T2 = { {5, 6}, {7, 8} };
	T1.swap(T2);							//交换矩阵T1和T2
	T1.brief_print(), T2.brief_print();
}

void show_Mat_attributes_part3()
{
	//arma::Mat<arma::sword> A = { {1, 2, 3}, {4, 5, 6} };
	//arma::ivec B = A.as_col();
	arma::Mat<int> A = { {1, 2, 3}, {4, 5, 6} };
	arma::Col<int> B = A.as_col();					//转化成列向量，数据类型需要相同，arma::sword和int不能直接转化
	B.brief_print();	
	arma::Row<int> RowA = A.as_row(); RowA.brief_print();	//转化成行向量
	
	//转置矩阵
	arma::Mat<int> TA = A.t(); TA.brief_print();		//获取A的转置矩阵
	arma::Col<int> TRowA = RowA.t(); TRowA.brief_print();	//行向量，列向量之间也可以转置

	//逆矩阵
	arma::mat TestB(3, 3, arma::fill::randu); TestB *= 5; TestB.brief_print();
	arma::mat iTestB = TestB.i();	iTestB.brief_print();	//iTestB是TestB的逆矩阵
	arma::mat mulB = TestB * iTestB;	
	//原矩阵*逆矩阵得到单位矩阵，利用lambda表达式将0附近的误差清除
	mulB.transform([](double val) {if (abs(val) < 1e-10) { return 0; }}); mulB.brief_print();

	//矩阵最值
	arma::imat TestC = { {6, 1, 2, -1}, {-4, 2, 5, 1}, {3, 7, -8, 4} };
	arma::sword maxC = TestC.max();						//找出矩阵TestC中的最大值和最小值
	int minC = TestC.min();
	std::cout << "max_value: " << maxC << " " << "min_value: " << minC << std::endl;
	int maxidx = TestC.index_max();						////找出矩阵TestC中的最大值和最小值的位置，按列优先方式存储
	arma::uword minidx = TestC.index_min();
	std::cout << "max_index: " << maxidx << " " << "min_index: " << minidx << std::endl;
	int maxp = TestC(maxidx), minp = TestC(minidx);		//通过下标定位找值
	std::cout << maxp << ' ' << minp << std::endl;
}

void show_Mat_attributes_part4()
{
	arma::ivec A = arma::regspace<arma::ivec>(0, 10);	//产生从0到10，默认间隔为1的ivec型列向量
	A.t().brief_print();
	arma::ivec A1 = arma::regspace<arma::ivec>(5, -5);	//产生从5到-5，默认间隔为-1的ivec型列向量
	A1.t().brief_print();
	arma::ivec A3 = arma::regspace<arma::ivec>(0, 2, 11);	//产生从0到11，指定间隔为2的ivec型列向量
	A3.t().brief_print();
	arma::irowvec A4 = arma::regspace<arma::irowvec>(13, -3, -5);	//产生从13到-5，指定间隔为-3的行向量
	A4.brief_print();

	arma::umat TestB(6, 0);
	for (int i = 0; i < 10; i++) {
		arma::arma_rng::set_seed_random();		//产生随机种子
		arma::uvec BB = arma::randperm(10, TestB.n_rows);	//产生0-9之间的排列，排列长度为TestB行数
		//BB.brief_print();
		TestB.insert_cols(0, BB);	//把BB插入到TestB中
	}
	TestB.brief_print();	//结合randperm和insert_cols,可以产生整数排列的矩阵

}

void show_Mat_pointers_iterators()
{
	//关于指针
	arma::mat A(5, 5, arma::fill::randu);
	double* A_pot = A.memptr();				//获取指针，A_pot指向A的第一个元素
	A.brief_print();
	std::cout << *(A_pot + 4) << std::endl;				//指向了第5行的第一个元素
	//*(A_pot+6)会输出第二列的第二个元素，说明了armadillo的mat型矩阵采用了列优先的方式来进行存储
	double* A_cpt = A.colptr(2);	//指针A_cpt指向A的第三列的第一个元素
	std::cout << *(A_cpt  + 5) << std::endl;	//armadillo的mat型矩阵采用了列优先的方式来进行存储，会按列一个个找

	//关于迭代器
	arma::mat X(5, 6, arma::fill::randu);
	X *= 10;
	X.print();
	arma::mat::iterator it;
	for (it = X.begin(); it != X.end(); it++) {				//使用迭代器遍历矩阵中的每个元素，按列优先输出
		std::cout << std::setprecision(3) << *it << ' ';
	}
	std::cout << std::endl;
	arma::mat::col_iterator it_col;
	for (it_col = X.begin_col(1); it_col != X.end_col(4); it_col++) {	//使用列迭代器遍历第2列到第5列之间的元素，按列优先输出
		std::cout << std::setprecision(3) << *it_col << ' ';
	}
	std::cout << std::endl;
	arma::mat::row_iterator it_row;						//使用行迭代器遍历第3行到第4行之间的元素，此时是按行优先来输出的
	for (it_row = X.begin_row(2); it_row != X.end_row(3); it_row++) {	
		std::cout << std::setprecision(3) << *it_row << ' ';
	}

	//对于Cube类型，也有对应的迭代器
}

void show_Mat_saving_loading()
{
	arma::mat A(5, 5, arma::fill::randu);
	
	//文件保存
	A.save("testData/A.txt", arma::raw_ascii);		//将矩阵A写到一个txt文件里，不加raw_ascii的话里面是乱码
	A.save("testData/A.csv", arma::csv_ascii);		//将矩阵A写到csv文件中
	A.brief_print();
	arma::field<std::string> header(A.n_cols);		//存放每一列的标题
	header(0) = "first", header(1) = "second", header(2) = "third", header(3) = "fourth", header(4) = "fifth";
	A.save(arma::csv_name("testData/Aheader.csv", header));		//保存带有列标题的csv文件

	//文件读取
	arma::Mat<double> B;							
	B.load("testData/A.txt", arma::raw_ascii);		//读取A.txt中的数据，如果A.txt中格式正确，不写arma::raw_ascii也可以
	B.raw_print();
	std::cout << std::endl;
	arma::mat C;
	C.load("testData/Aheader.csv");					//读取带列标题的csv文件，这样读，第一行是全0		
	C.raw_print(); std::cout << std::endl;
	C.shed_row(0); C.raw_print();
	
}

int main2()
{
	std::cout << "Armadillo version: " << arma::arma_version::as_string() << std::endl;
	//show_DataTypes();
	//show_Mat_operations();
	//show_Mat_attributes_part1();
	//show_Mat_attributes_part2();
	//show_Mat_attributes_part3();
	show_Mat_attributes_part4();
	//show_Mat_pointers_iterators();
	//show_Mat_saving_loading();
	return 0;
}