#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <assert.h>
#include<bits/stdc++.h>
#include <chrono>
#include "tnt_array1d.h"
#include "tnt_array2d.h"

#include "jama_eig.h"


using namespace std;
using namespace TNT;
using namespace JAMA;

namespace PCA {
	bool debug = false;
	
	void load_data_from_file(Array2D<double>& d) {
		vector<vector<double>> data(20000,vector<double>(16));
		ifstream inputFile("eegData.csv");
        string line;
		string number;
        int r = 0;
		stringstream rowDataStream;
        if (inputFile.is_open()) {
            while (!inputFile.eof()) {
				int col = 0;
                getline(inputFile, line);
                if (line.empty()) continue;
				rowDataStream.clear();
				rowDataStream.str(line);
				while (rowDataStream.good()) {
					getline(rowDataStream,number, ',');
					if (!number.empty()) {
						double v=0.0;
						v=stod(number);
						//row.push_back(v);
						data[r][col]=v;
					}
					col += 1;
					if(col==16){
						break;
					}
				}
				r += 1;
				if(r==20000){
					break;
				}
            }
            inputFile.close();
        }
		for(int i=0;i<20000;i++){
			for(int j=0;j<16;j++){
				d[i][j] =data[i][j];
			}
		}
	}
	
	void adjust_data(Array2D<double>& d, Array1D<double>& means) {
	   for (int i=0; i<d.dim2(); ++i) { 
		   double mean = 0;
		   for (int j=0; j<d.dim1(); ++j) {
			   mean += d[j][i];
		   }

		   mean /= d.dim1();

		   // store the mean
		   means[i] = mean;

		   // subtract the mean
		   for (int j=0; j<d.dim1(); ++j) {
			   d[j][i] -= mean;
		   }
	   }
	}

	double compute_covariance(const Array2D<double>& d, int i, int j) {
	   double cov = 0;
	   for (int k=0; k<d.dim1(); ++k) {
		   cov += d[k][i] * d[k][j];
	   }

	   return cov / (d.dim1() - 1);
	}

	void compute_covariance_matrix(const Array2D<double> & d, Array2D<double> & covar_matrix) {
		int dim = d.dim2();
		assert(dim == covar_matrix.dim1());
		assert(dim == covar_matrix.dim2());
		for (int i=0; i<dim; ++i) {
			for (int j=i; j<dim; ++j) {
				covar_matrix[i][j] = compute_covariance(d, i, j);
			}
		}


		// fill the Left triangular matrix
		for (int i=1; i<dim; i++) {
			for (int j=0; j<i; ++j) {
				covar_matrix[i][j] = covar_matrix[j][i];
			}
		}

	}

	// Calculate the eigenvectors and eigenvalues of the covariance
	// matrix
	void eigen(const Array2D<double> & covar_matrix, Array2D<double>& eigenvector, Array2D<double>& eigenvalue) {
		Eigenvalue<double> eig(covar_matrix);
		eig.getV(eigenvector);
		eig.getD(eigenvalue);
	}


	void transpose(const Array2D<double>& src, Array2D<double>& target) {
		for (int i=0; i<src.dim1(); ++i) {
			for (int j=0; j<src.dim2(); ++j) {
				target[j][i] = src[i][j];
			}
		}
	}

	// z = x * y
	void multiply(const Array2D<double>& x, const Array2D<double>& y, Array2D<double>& z) {
		assert(x.dim2() == y.dim1());
		for (int i=0; i<x.dim1(); ++i) {
			for (int j=0; j<y.dim2(); ++j) {
				double sum = 0;
				int d = y.dim1();
				for (int k=0; k<d; k++) {
					sum += x[i][k] * y[k][j];
				}
				z[i][j] = sum;
			}
		}
	}
}

int main() {
	chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	using namespace PCA;
	
    const int row = 20000;
    const int col = 16;
    Array2D<double> d(row, col);
    load_data_from_file(d);
    Array1D<double> means(col);
    adjust_data(d, means);
    Array2D<double> covar_matrix(col, col);
    compute_covariance_matrix(d, covar_matrix);

    // get the eigenvectors
    Array2D<double> eigenvector(col, col);
    // get the eigenvalues
    Array2D<double> eigenvalue(col, col);
    eigen(covar_matrix, eigenvector, eigenvalue);
    // restore the old data
    // final_data = RowFeatureVector * RowDataAdjust
    Array2D<double> transpose_data(col, row);
    transpose(d, transpose_data);
    Array2D<double> transpose_projected_data(col, row);
	PCA::multiply(eigenvector, transpose_data, transpose_projected_data);

	Array2D<double> projected_data(row, col);
	PCA::transpose(transpose_projected_data, projected_data);

	// Reconstruct the adjusted data from the projected data
	Array2D<double> final_data(row, col);
	PCA::multiply(projected_data, eigenvector, final_data);
	vector<vector<double>> result;
	ofstream outputFile("Result.csv");
	if (outputFile.is_open())
	{
		for (int i=0; i<20000; ++i)
		{
			outputFile << final_data[i][0] << "," << final_data[i][1] << endl;
			result.push_back({final_data[i][0],final_data[i][1]});
		}
		outputFile.close();
	}
	for(int i=0;i<5;i++){
		for(int j=0;j<2;j++){
			cout<<result[i][j]<<"  ";
		}
		cout<<endl;
	}
	chrono::steady_clock::time_point end=chrono::steady_clock::now();
	cout<<"Time difference = "<<chrono::duration_cast<chrono::microseconds>(end - begin).count()<<"[micro seconds]"<<endl;
}
