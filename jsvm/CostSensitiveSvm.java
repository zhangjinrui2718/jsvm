package jsvm;

import Jama.Matrix;
import jsvm.base.Kernelable;
import jsvm.base.SupportVectorMachine;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by zjr on 2017/5/19.
 */
public class CostSensitiveSvm extends SupportVectorMachine implements Kernelable {
    //代价敏感支持向量机参数
    private double Cp;
    private double Cn;

    //支持向量产生的w
    private Matrix wSv;

    //核函数参数
    private double[] kernelParam;

    public CostSensitiveSvm(Matrix data, Matrix labels) {
        super(data, labels);
    }

    @Override
    protected double calcBoundC(int i) {
        if (labels.get(i, 0) == 1) {
            return Cn;
        } else {
            return Cp;
        }
    }

    @Override
    public void train(int maxIteration, double... param) {
        Cp = param[0];
        Cn = param[1];
        kernelParam = Arrays.copyOfRange(param, 2, param.length);

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

    @Override
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

        CostSensitiveSvm cssvm = new CostSensitiveSvm(data, labels);
        //cp cn rbf
        cssvm.train(100, 600, 100, 0.6);//10 - 0.1

        HashMap<String, Matrix> testMap = SvmUtil.loadSet(testFileName, " ");
        Matrix x = testMap.get("data");//m*n
        Matrix y = testMap.get("labels").transpose();//m*1
        Map<String, Integer> testMap1 = cssvm.evaluate(x, y);

        System.out.println(testMap1.toString());
        SvmUtil.showSvmPerformance(testMap1);
    }
}
