package jsvm;

import Jama.Matrix;
import jsvm.base.Kernelable;
import jsvm.base.SupportVectorMachine;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by zjr on 2017/5/7.
 */

public class SoftMarginSvm extends SupportVectorMachine implements Kernelable {

    //正则参数
    private double C;

    //支持向量产生的w
    private Matrix wSv;

    //核函数参数
    private double[] kernelParam;

    public SoftMarginSvm(Matrix data, Matrix labels) {
        super(data, labels);
    }

    /**
     * @param i 样本序号
     * @return 样本对应的边界
     */
    @Override
    protected double calcBoundC(int i) {
        return C;
    }

    /**
     * @param maxIteration 最大迭代次数
     * @param param        所需要的参数 C rbf-theta
     */
    @Override
    public void train(int maxIteration, double... param) {
        if (param.length != 2) System.out.println("参数长度不为2 ! param 的长度实际上为" + param.length);

        C = param[0];
        kernelParam = Arrays.copyOfRange(param, 1, param.length);

        //核函数转化矩阵K
        //m*m
        for (int i = 0; i < sampleCount; i++) {
            K.setMatrix(0, sampleCount - 1, i, i,
                    kernelTrans(data, data.getMatrix(i, i, 0, featureCount - 1), kernelParam));
        }

        smo(maxIteration);
        calcSV();
        calcW();

        //p*1
        wSv = svLabelsMat.arrayTimes(svAlphasMat);
    }

    /**
     * 单个样本的预测
     *
     * @param x 单个样本
     * @return 预测标签
     */
    public int predict(Matrix x) {
        Matrix X = kernelTrans(svMat, x, kernelParam);//p*1
        double res = X.transpose().times(wSv).get(0, 0) + b;

        if (res > 0) return 1;
        else return -1;
    }


    public static void main(String[] args) {
        double rate = 0.1;
        String trainFileName = "data/en_lt_" + rate + "_" + 0;
        String testFileName = "data/en_lt_" + rate + "_" + 1;

        HashMap<String, Matrix> trainMap = SvmUtil.loadSet(trainFileName, " ");
        Matrix data = trainMap.get("data");//m*n
        Matrix labels = trainMap.get("labels").transpose();//m*1

        SoftMarginSvm ssvm = new SoftMarginSvm(data, labels);
        ssvm.train(100, 10, 0.17);//10 - 0.1

        HashMap<String, Matrix> testMap = SvmUtil.loadSet(testFileName, " ");
        Matrix x = testMap.get("data");//m*n
        Matrix y = testMap.get("labels").transpose();//m*1
        Map<String, Integer> testMap1 = ssvm.evaluate(x,y);

        System.out.println(testMap1.toString());
        SvmUtil.showSvmPerformance(testMap1);
    }
}
