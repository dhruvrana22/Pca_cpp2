#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include "qbMatrix.h"
#include "qbVector.h"
#include "qbPCA.h"

using namespace std;

int main()
{
	cout << "Testing with 20000 observations of 16 variables:" << endl;
	// Read data from the .CSV file.
	string rowData;
	string number;
	stringstream rowDataStream;
	std::vector<vector<double>> Data(20000,vector<double>(16));
	int numRows = 0;
	int numCols = 0;
	ifstream inputFile("eegData.csv");
	/* If the file open successfully then do stuff. */
	if (inputFile.is_open())
	{		
		cout << "Opened file successfully..." << endl;
		while (!inputFile.eof())
		{
			// Read the next line.
			getline(inputFile, rowData);
			
			// Loop through and extract the individual numbers.
			rowDataStream.clear();
			rowDataStream.str(rowData);
			numCols = 0;
			while (rowDataStream.good())
			{
				getline(rowDataStream,number, ',');
				if (!number.empty()) {
					double v=0.0;
					v=stod(number);
					//row.push_back(v);
					Data[numRows][numCols]=v;
				}
				numCols += 1;
				if(numCols==16){
					break;
				}
			}
			numRows += 1;
			if(numRows==20000){
				break;
			}
		}
		// Close the file.
		inputFile.close();
	}
	chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	vector<double> testData;
	for(int i=0;i<20000;i++){
		for(int j=0;j<16;j++){
			testData.push_back(Data[i][j]);
		}
	}
	chrono::steady_clock::time_point end=chrono::steady_clock::now();
	cout<<"Time1= "<<chrono::duration_cast<chrono::microseconds>(end - begin).count()<<"[micro seconds]"<<endl;
	//cout << "Completed reading file..." << endl;
	//cout << "Read " << numRows << " observations of " << numCols << " variables." << endl;
	//cout << "Constituting " <<Data.size() << " elements in total." << endl;
	
	// Form into a matrix.
	qbMatrix2<double> X (numRows, numCols,testData);
	// Compute the covariance matrix.
	std::vector<double> columnMeans = qbPCA::ComputeColumnMeans(X);
	qbMatrix2<double> X2 = X;
	qbPCA::SubtractColumnMeans(X2, columnMeans);
	
	qbMatrix2<double> covX = qbPCA::ComputeCovariance(X2);
	//cout << endl;
	//cout << "Giving the covariance matrix as: " << endl;
	//covX.PrintMatrix();
	
	// Compute the eigenvectors.
	qbMatrix2<double> eigenvectors;
	int testResult = qbPCA::ComputeEigenvectors(covX, eigenvectors);
	//cout << endl;
	//cout << "And the eigenvectors as: " << endl;
	//eigenvectors.PrintMatrix();
	
	// Test the overall function.
	//cout << endl;
	//cout << "Testing overall function..." << endl;
	qbMatrix2<double> eigenvectors2;
	int testResult2 = qbPCA::qbPCA(X, eigenvectors2);
	//cout << "testResult2 = " << testResult2 << endl;
	//cout << "And the final eigenvectors are:" << endl;
	//eigenvectors2.PrintMatrix();
	
	// Test dimensionality reduction.
	//cout << endl;
	//cout << "Testing dimensionality reduction." << endl;
	//cout << "Starting with X which has " << X.GetNumRows() << " rows and " << X.GetNumCols() << " columns." << endl;
	//cout << endl;
	//cout << "Using only the first two principal components:" << endl;
	qbMatrix2<double> V, part2;
	eigenvectors.Separate(V, part2, 2);
	//V.PrintMatrix(8);
	//cout << endl;
	
	qbMatrix2<double> newX = (V.Transpose() * X.Transpose()).Transpose();
	vector<vector<double>> result(20000,vector<double>(2));
	//cout << "Result has " << newX.GetNumRows() << " rows and " << newX.GetNumCols() << " columns." << endl;
	chrono::steady_clock::time_point begini= std::chrono::steady_clock::now();
	for(int i=0;i<20000;i++){
		for(int j=0;j<2;j++){
			result[i][j]=newX.GetElement(i,j);
		}
	}
	chrono::steady_clock::time_point endi=std::chrono::steady_clock::now();
	cout<<"Time2= "<<chrono::duration_cast<chrono::nanoseconds>(endi-begini).count()<<"[nano seconds]"<<endl;
	// Open a file for writing
	ofstream outputFile("Result.csv");
	if (outputFile.is_open())
	{
		for (int i=0; i<newX.GetNumRows(); ++i)
		{
			outputFile << newX.GetElement(i, 0) << "," << newX.GetElement(i, 1) << endl;
		}
		outputFile.close();
	}		
}
	
