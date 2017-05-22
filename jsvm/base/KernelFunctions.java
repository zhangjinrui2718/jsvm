package jsvm.base;

import Jama.Matrix;

/**
 * Created by zjr on 2017/5/19.
 */
public class KernelFunctions {
    public static Matrix linearKernel(Matrix X, Matrix A) {
        return X.times(A.transpose());
    }

    /**
     * 进行RBF核函数转化
     * 实际上该操作将一个1*n的样本升高了维度
     * 变成了一个m*1的列向量
     *
     * @param X xi m*n
     * @param A xj 1*n
     * @return k(xi, xj) = exp(||xi - xj||^2 / -1*theta^2)
     * k(xi, xj) = exp(-1 * ||xi - xj||^2  / theta^2)
     */
    public static Matrix rbfKernel(Matrix X, Matrix A, double... param) {
        int m = X.getRowDimension();
        int n = X.getColumnDimension();

        // 1*m
        double[] k = new double[m];

        for (int i = 0; i < m; i++) {
            // 1*n
            Matrix deltaRow = X.getMatrix(i, i, 0, n - 1).minus(A);
            double ki = deltaRow.times(deltaRow.transpose()).get(0, 0);
            k[i] = Math.exp(ki * -1.0 / Math.pow(param[0], 2));
        }
        // k : m*1
        return new Matrix(k, 1).transpose();
    }
}
