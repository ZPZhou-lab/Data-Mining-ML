#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <math.h>
#include <tuple>
#include "numerical_cpp.cpp"
using namespace std;

// 随机数生成器
default_random_engine random_engine;
uniform_real_distribution<double> uniform(-1,1);
numcpp nc;

//定义Sigmoid函数
double sigmoid(double x){
	return 1/(1+exp(-x));
}

class binary_logistic{
public:
	//构造函数
	binary_logistic(int n_d){
		d = n_d + 1;
		for(int i=0;i<d;i++){
			vector<double> col(1);
			beta.push_back(col);
			beta[i][0] = uniform(random_engine);
		}
		cout<<"construction fuuction!"<<endl;
	}

	//预测函数
	vector<vector<double>> predict(vector<vector<double>> X,string mode = "label",double threshold = 0.5){
		vector<vector<double>> p(X.size(),vector<double>(1));
		//添加常数项维度
		for(int i=0;i<X.size();i++){
			X[i].push_back(1.0);
		}
		//矩阵乘法
		p = nc.matmul(X,beta);
		for(int i=0;i<p.size();i++){
			p[i][0] = sigmoid(p[i][0]);
		}
		if(mode.compare("label")==0){
			for(int i=0;i<p.size();i++){
				if(p[i][0] > threshold){
					p[i][0] = 1;
				}
				else{
					p[i][0] = 0;
				}
			}
			return p;
		}
		else{
			vector<vector<double>> pdf(p.size(),vector<double>(2));
			for(int i=0;i<p.size();i++){
				pdf[i][0] = p[i][0];
				pdf[i][1] = 1 - p[i][0];
			}
			return pdf;
		}
	}

	//梯度函数
	vector<vector<double>> gradient(vector<vector<double>> X,vector<vector<double>> y){
		vector<vector<double>> grads;
		vector<vector<double>> pdf = predict(X);
		vector<vector<double>> p(pdf.size(),vector<double>(1));
		for(int i=0;i<p.size();i++){
			p[i][0] = pdf[i][0];
		}
		vector<vector<double>> X_one(X.size(),vector<double>(1,1.0));
		X = nc.concatenate(X,X_one,1);
		grads = nc.mean(nc.dot((p-y),X),0);
		grads = nc.transpose(grads);
		return grads;
	}

	void gradient_descent(vector<vector<double>> X,vector<vector<double>> y,double lr=0.1){
		vector<vector<double>> grads = gradient(X,y);
		beta = beta - lr*grads;
	}

	void train_model(vector<vector<double>> X,vector<vector<double>> y,int EPOCHS=1000,double lr=0.1){
		for(int epoch=0;epoch<EPOCHS;epoch++){
			gradient_descent(X,y,lr);
			double acc = accuracy(X,y);
			cout<<"epoch: "<<epoch+1<<"   acc: "<<acc<<endl;
		}
	}

	double accuracy(vector<vector<double>> X,vector<vector<double>> y_true){
		vector<vector<double>> y_pred = predict(X,"label");
		vector<vector<double>> acc_count = nc.sum((y_pred == y_true),0);
		double acc = acc_count[0][0] / y_pred.size();
		return acc;
	}

private:
	int d;
	vector<vector<double>> beta;

};

void load_data(vector<vector<double>> &X_data, vector<string> &target){
	//读取训练数据
	ifstream infile("letter_recognition.csv");
	if(!infile){
		cout<<"open file error!"<<endl;
		exit(1);
	}
	else{
		cout<<"open file successfully!"<<endl;
	}
	int i = 0;
	string line;
	while(getline(infile,line)){
		string field;
		istringstream sin(line);
		if(i == 0){
			getline(sin,field,',');
			i++;
			continue;
		}
		getline(sin,field,',');
		getline(sin,field,',');
		target[i-1] = field.c_str();
		for(int j=0;j<16;j++){
			getline(sin,field,',');
			X_data[i-1][j] = atof(field.c_str());
		}
		i++;
	}
}

void OvO(void){
	
}

int main()
{
	//创建特征和标签
	vector<vector<double>> X_data(20000,vector<double>(16));
	vector<string> target(20000);
	load_data(X_data,target);
	cout<<X_data[0][0]<<endl;
	cout<<target[0]<<endl;

	//划分训练集和测试集
	double train_size = 0.6;
	vector<vector<double>> X_train(X_data.begin(),X_data.begin()+int(train_size*20000));
	vector<vector<double>> X_test(X_data.begin()+int(train_size*20000),X_data.end());
	vector<string> y_train(target.begin(),target.begin()+int(train_size*20000));
	vector<string> y_test(target.begin()+int(train_size*20000),target.end());
	cout<<"data processed!"<<endl;

	binary_logistic model(16);
	vector<vector<double>> p(X_train.size(),vector<double>(1));
	vector<vector<double>> train_target;
	vector<vector<double>> y_pred;

	//OvO test
	string letter1 = "A";
	string letter2 = "B";
	vector<vector<double>> ovo_X_train;
	vector<string> ovo_y_train;
	for(int i=0;i<12000;i++){
		if(y_train[i] == letter1 || y_train[i] == letter2){
			ovo_X_train.push_back(X_train[i]);
			ovo_y_train.push_back(y_train[i]);
		}
	}
	train_target = (letter1 == ovo_y_train);
	double acc = model.accuracy(ovo_X_train,train_target);
	cout<<"acc0: "<<acc<<endl;
	y_pred = model.predict(ovo_X_train,"label");
	int EPOCHS = 100;
	double lr = 0.1;
	model.train_model(ovo_X_train,train_target,EPOCHS,lr);

	
	system("pause");
	return 0;
}
