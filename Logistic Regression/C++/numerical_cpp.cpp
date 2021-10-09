#include <iostream>
#include <cstdlib>
#include <ctime>
#include <time.h>
#include <cmath>
#include <iomanip>
#include <vector>
#include <assert.h>
using namespace std;

template<typename T, size_t... Ns>
auto vector_n = T{};

template<typename T, size_t N>
auto vector_n<T, N> = vector<T>(N);

template<typename T, size_t N, size_t ...Ns>
auto vector_n<T, N, Ns...> = vector<decltype(vector_n<T, Ns...>)> (N, vector_n<T, Ns...>);

//重载运算符
template<typename T>
vector<vector<T>> operator+(vector<vector<T>> A,vector<vector<T>> B){
	vector<vector<T>> C(A.size(),vector<T>(A[0].size()));
	for(int i=0;i<A.size();i++){
		for(int j=0;j<A[0].size();j++){
			C[i][j] = A[i][j] + B[i][j];
		}
	}
	return C;
}

vector<vector<double>> operator-(vector<vector<double>> A,vector<vector<double>> B){
	vector<vector<double>> C(A.size(),vector<double>(A[0].size()));
	for(int i=0;i<A.size();i++){
		for(int j=0;j<A[0].size();j++){
			C[i][j] = A[i][j] - B[i][j];
		}
	}
	return C;
}

vector<vector<double>> operator*(double a,vector<vector<double>> A){
	vector<vector<double>> C(A.size(),vector<double>(A[0].size()));
	for(int i=0;i<A.size();i++){
		for(int j=0;j<A[0].size();j++){
			C[i][j] = a*A[i][j];
		}
	}
	return C;
}

vector<vector<double>> operator==(string a,vector<string> A){
	vector<vector<double>> C(A.size(),vector<double>(1));
	for(int i=0;i<A.size();i++){
            if(A[i] == a){
                C[i][0] = 1;
            }
            else{
                C[i][0] = 0;
            }
    }
	return C;
}

vector<vector<double>> operator==(double a,vector<vector<double>> A){
	vector<vector<double>> C(A.size(),vector<double>(A[0].size()));
	for(int i=0;i<A.size();i++){
		for(int j=0;j<A[0].size();j++){
            if(A[i][j] == a){
                C[i][j] = 1;
            }
            else{
                C[i][j] = 0;
            }
			
		}
	}
	return C;
}

vector<vector<double>> operator==(vector<vector<double>> A,vector<vector<double>> B){
	vector<vector<double>> C(A.size(),vector<double>(A[0].size()));
	for(int i=0;i<A.size();i++){
		for(int j=0;j<A[0].size();j++){
            if(A[i][j] == B[i][j]){
                C[i][j] = 1;
            }
            else{
                C[i][j] = 0;
            }
			
		}
	}
	return C;
}

class numcpp{
public:
    //矩阵乘法
    vector<vector<double>> matmul(vector<vector<double>> A,vector<vector<double>> B){
        int M = A.size();
        int K = A[0].size();
        int N = B[0].size();
        double s;
        vector<vector<double>> C(M,vector<double>(N,0));
        for(int i=0;i<M;i++){
            for(int k=0;k<K;k++){
                s = A[i][k];
                for(int j=0;j<N;j++){
                    C[i][j] += s*B[k][j];
                }
            }
        }
        return C;
    }

    //打印矩阵
    void print(vector<vector<double>> A,int limit=100){
        for(int i=0;i<A.size();i++){
            for(int j=0;j<A[0].size();j++){
                cout<<setw(8)<<A[i][j];
            }
            cout<<endl;
            if(i == limit){
                break;
            }
        }
        cout<<endl;
    }

    //矩阵转置
    vector<vector<double>> transpose(vector<vector<double>> A){
        vector<vector<double>> B(A[0].size(),vector<double>(A.size()));
        for(int i=0;i<A.size();i++){
            for(int j=0;j<A[0].size();j++){
                B[j][i] = A[i][j];
            }
        }
        return B;
    }

    //拼接矩阵
    vector<vector<double>> concatenate(vector<vector<double>> A,vector<vector<double>> B,int axis=0){
        //行拼接
        if(axis == 0){
            assert(A[0].size() == B[0].size());
            vector<vector<double>> C = A;
            for(int i=0;i<B.size();i++){
                C.push_back(B[i]);
            }
            return C;
        }
        //行拼接
        else{
            assert(A.size() == B.size());
            A = transpose(A);
            B = transpose(B);
            vector<vector<double>> C = A;
            for(int i=0;i<B.size();i++){
                C.push_back(B[i]);
            }
            C = transpose(C);
            return C;
        }
    }

    //点乘
    vector<vector<double>> dot(vector<vector<double>> A,vector<vector<double>> B){
        int m1 = A.size();
        int n1 = A[0].size();
        int m2 = B.size();
        int n2 = B[0].size();
        if(m1 == m2 && n1 == n2){
            vector<vector<double>> C(m1,vector<double>(n1));
            for(int i=0;i<m1;i++){
                for(int j=0;j<n1;j++){
                    C[i][j] = A[i][j]*B[i][j];
                }
            }
            return C;
        }
        //广播机制
        else if (m1 == m2 && n1 != n2){
            if(n1 == 1){
                vector<vector<double>> C;
                A = transpose(A);
                for(int i=0;i<n2;i++){
                    C.push_back(A[0]);
                }
                C = transpose(C);
                return dot(C,B);
            }
            else{
                vector<vector<double>> C;
                B = transpose(B);
                for(int i=0;i<n1;i++){
                    C.push_back(B[0]);
                }
                C = transpose(C);
                return dot(A,C);
            }
        }
        else{
            if(m1 == 1){
                vector<vector<double>> C;
                for(int i=0;i<m2;i++){
                    C.push_back(A[0]);
                }
                return dot(C,B);
            }
            else{
                vector<vector<double>> C;
                for(int i=0;i<m1;i++){
                    C.push_back(B[0]);
                }
                return dot(A,C);
            }
        }
    }

    //均值
    vector<vector<double>> mean(vector<vector<double>> A,int axis=0){
        int m = A.size();
        int n = A[0].size();
        
        if(axis == 0){
            vector<vector<double>> Mean(1,vector<double>(n,0));
            for(int j=0;j<n;j++){
                for(int i=0;i<m;i++){
                    Mean[0][j] += A[i][j];
                }
                Mean[0][j] /= m;
            }
            return Mean;
        }
        else{
            vector<vector<double>> Mean;
            A = transpose(A);
            Mean = mean(A,0);
            Mean = transpose(Mean);
            return Mean;
        }
    }

    //求和
    vector<vector<double>> sum(vector<vector<double>> A,int axis=0){
        int m = A.size();
        int n = A[0].size();
        
        if(axis == 0){
            vector<vector<double>> Mean(1,vector<double>(n,0));
            for(int j=0;j<n;j++){
                for(int i=0;i<m;i++){
                    Mean[0][j] += A[i][j];
                }
            }
            return Mean;
        }
        else{
            vector<vector<double>> Mean;
            A = transpose(A);
            Mean = mean(A,0);
            Mean = transpose(Mean);
            return Mean;
        }
    }
};
