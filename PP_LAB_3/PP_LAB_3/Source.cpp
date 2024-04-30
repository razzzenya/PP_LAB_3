#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <algorithm> 
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

#define PATH_TO_FIRST_MATRIX "C:\\vscode_repos\\PP_LAB_3\\first matrix.txt"
#define PATH_TO_SECOND_MATRIX "C:\\vscode_repos\\PP_LAB_3\\second matrix.txt"
#define PATH_TO_RESULT_MATRIX_FILE "C:\\vscode_repos\\PP_LAB_3\\result matrix.txt"
#define PATH_TO_RESULT_FILE "C:\\vscode_repos\\PP_LAB_3\\result.txt"
#define ROW_START_TAG 0    //tag for communicating the start row of the workload for a slave
#define ROW_END_TAG 1      //tag for communicating the end row of the workload for a slave
#define A_ROWS_TAG 2       //tag for communicating the address of the data to be worked on to slave
#define C_ROWS_TAG 3       //tag for communicating the address of the calculated data to master
#define LOCAL_TIME_TAG 4   //tag for communicating the address of the local matrix calculation time to master

int _rank;                  // mpi: process id number
int nProcesses;            // mpi: number of total processess 
MPI_Status status;         // mpi: store status of a MPI_Recv
MPI_Request request;       // mpi: capture request of a MPI_Isend
int rowStart, rowEnd;      // which rows of A that are calculated by the slave process
int granularity; 	   // granularity of parallelization (# of rows per processor) 

double start_time, end_time;
double localTimeSaver;

int randomHigh = 100;
int randomLow = 1;

template <class T>
class Matrix {
public:
    Matrix(int numrows, int numcols)
        :Nrow(numrows), Ncol(numcols), elements(Nrow* Ncol) {}

    Matrix(int numrows, int numcols, T* data)
        :Nrow(numrows), Ncol(numcols), elements(data, data + numrows * numcols) {}

    int rows() { return Nrow; }
    int cols() { return Ncol; }

    T operator() (int row, int col) const { return elements[Ncol * row + col]; }
    T& operator() (int row, int col) { return elements[Ncol * row + col]; }


    T* data() { return elements.data(); }
    const vector<T>& elem() { return elements; }

private:
    int Nrow, Ncol;
    vector<T> elements;
};

void FillMatricesRandomly(Matrix <int>& A, Matrix <int>& B) {
    srand(time(NULL));
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            A(i, j) = rand() % (randomHigh - randomLow) + randomLow;
        }
    }
    for (int i = 0; i < B.rows(); i++) {
        for (int j = 0; j < B.cols(); j++) {
            B(i, j) = rand() % (randomHigh - randomLow) + randomLow;
        }
    }
}

void PrintMatrices(Matrix <int>& A, Matrix <int>& B, Matrix <int>& C) {
    cout << "\n\nMatrix A" << endl;
    for (int i = 0; i < A.rows(); i++) {
        cout << endl << endl;
        for (int j = 0; j < A.cols(); j++)
            cout << A(i, j) << " ";
    }

    cout << "\n\n\n\nMatrix B" << endl;

    for (int i = 0; i < B.rows(); i++) {
        cout << "\n" << endl;
        for (int j = 0; j < B.cols(); j++)
            cout << B(i, j) << " ";
    }

    cout << "\n\n\n\nMultiplied Matrix C" << endl;

    for (int i = 0; i < C.rows(); i++) {
        cout << "\n" << endl;
        for (int j = 0; j < C.cols(); j++)
            cout << C(i, j) << " ";
    }

    cout << endl << endl << endl;
}

void SaveMatrix(Matrix<int>& matrix, const char* path) {
    ofstream file(path);
    for (size_t i = 0; i < matrix.rows() * matrix.cols(); ++i) {
        if (i % matrix.rows() == 0) {
            file << endl;
        }
        file << matrix.data()[i] << " ";
    }
    file.close();
}

void SaveTime(double time) {
    ofstream file(PATH_TO_RESULT_FILE);
    file << time << endl;
    file.close();
}

int main(int argc, char* argv[]) {
    if (argv[1] == NULL) {
        cout << "ERROR: The program must be executed in the following way  \n\n  \t \"mpiexec -n NumberOfProcesses mpi N \"  \n\n where N is an integer. \n \n " << endl;
        return 1;
    }

    int N = atoi(argv[1]);

    int numberOfRowsA = N;
    int numberOfColsA = N;
    int numberOfRowsB = N;
    int numberOfColsB = N;

    Matrix <int> A = Matrix <int>(numberOfRowsA, numberOfColsA);
    Matrix <int> B = Matrix <int>(numberOfRowsB, numberOfColsB);
    Matrix <int> C = Matrix <int>(numberOfRowsA, numberOfColsB);

    MPI_Init(&argc, &argv);		                
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);	    
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

    if (_rank == 0) {
        cout << "Starting an MPI parallel matrix multiplication. \n  " << endl;
        cout << "The matrices are: " << N << "x" << N << endl;
        FillMatricesRandomly(A, B);

        start_time = MPI_Wtime();
        for (int i = 1; i < nProcesses; i++) {  
            granularity = (numberOfRowsA / (nProcesses - 1));
            rowStart = (i - 1) * granularity;

            if (((i + 1) == nProcesses) && ((numberOfRowsA % (nProcesses - 1)) != 0)) { 
                rowEnd = numberOfRowsA;
            }
            else {
                rowEnd = rowStart + granularity;
            }

            MPI_Isend(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&rowEnd, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&A(rowStart, 0), (rowEnd - rowStart) * numberOfColsA, MPI_INT, i, A_ROWS_TAG, MPI_COMM_WORLD, &request);
        }
    }
    MPI_Bcast(&B(0, 0), numberOfRowsB * numberOfColsB, MPI_INT, 0, MPI_COMM_WORLD);

    if (_rank > 0) {
        MPI_Recv(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&A(rowStart, 0), (rowEnd - rowStart) * numberOfColsA, MPI_INT, 0, A_ROWS_TAG, MPI_COMM_WORLD, &status);

        localTimeSaver = MPI_Wtime();

        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = 0; j < B.cols(); j++) {
                for (int k = 0; k < B.rows(); k++) {
                    C(i, j) += (A(i, k) * B(k, j));
                }
            }
        }
        localTimeSaver = MPI_Wtime() - localTimeSaver;

        MPI_Isend(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&C(rowStart, 0), (rowEnd - rowStart) * numberOfColsB, MPI_INT, 0, C_ROWS_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&localTimeSaver, 1, MPI_INT, 0, LOCAL_TIME_TAG, MPI_COMM_WORLD, &request);


    }

    if (_rank == 0) {
        for (int i = 1; i < nProcesses; i++) {
            MPI_Recv(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&rowEnd, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&C(rowStart, 0), (rowEnd - rowStart) * numberOfColsB, MPI_INT, i, C_ROWS_TAG, MPI_COMM_WORLD, &status);

        }
        end_time = MPI_Wtime();
        double totalMultiplicationTime = end_time - start_time;

        vector<double> LocalMultiplicationTimes = vector<double>(nProcesses);

        for (int i = 1; i < nProcesses; i++) {
            MPI_Recv(&LocalMultiplicationTimes[i], 1, MPI_INT, i, LOCAL_TIME_TAG, MPI_COMM_WORLD, &status);
        }
        double maxLocalMultiplicationTime = *max_element(LocalMultiplicationTimes.begin(), LocalMultiplicationTimes.end());

        cout << "Total multiplication time =  " << totalMultiplicationTime << "\n" << endl;
        cout << "Longest multiplication time =  " << maxLocalMultiplicationTime << "\n" << endl;
        cout << "Approximate communication time =  " << totalMultiplicationTime - maxLocalMultiplicationTime << "\n\n" << endl;
        SaveTime(totalMultiplicationTime);

        cout << "Saving matrices..." << endl;
        SaveMatrix(A, PATH_TO_FIRST_MATRIX);
        SaveMatrix(B, PATH_TO_SECOND_MATRIX);
        SaveMatrix(C, PATH_TO_RESULT_MATRIX_FILE);
        if (N <= 10) {
            PrintMatrices(A, B, C);
        }

    }
    MPI_Finalize();
    return 0;
}