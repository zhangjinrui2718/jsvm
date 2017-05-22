package jsvm.base;

import Jama.Matrix;

/**
 * Created by zjr on 2017/5/7.
 */
public interface Kernelable {
    default Matrix kernelTrans(Matrix X, Matrix A, double... param) {
        return KernelFunctions.rbfKernel(X, A, param);
    }
}
