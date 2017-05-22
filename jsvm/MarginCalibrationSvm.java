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
public class MarginCalibrationSvm extends SupportVectorMachine implements Kernelable {
    //代价敏感支持向量机参数
    private double Cp;
    private double Cn;
    private double Ap;
    private double An;

    //支持向量产生的w
    private Matrix wSv;

    //存放间隔支持向量信息
    private int msvCount;
    private Matrix msvMat;
    private Matrix msvLabelsMat;
    private Matrix msvAlphasMat;

    //核函数参数
    private double[] kernelParam;

    public MarginCalibrationSvm(Matrix data, Matrix labels) {
        super(data, labels);
    }

    @Override
    protected double calcBoundC(int i) {
        if (labels.get(i, 0) == 1) {
            return Cn * An;
        } else {
            return Cp * Ap;
        }
    }

    @Override
    public void train(int maxIteration, double... param) {
        Cp = param[0];
        Cn = param[1];
        Ap = param[2];
        An = param[3];

        kernelParam = Arrays.copyOfRange(param, 4, param.length);

        //核函数转化矩阵K
        //m*m
        for (int i = 0; i < sampleCount; i++) {
            K.setMatrix(0, sampleCount - 1, i, i,
                    kernelTrans(data, data.getMatrix(i, i, 0, featureCount - 1), kernelParam));
        }

        smo(maxIteration);
        calcSV();

        //p*1
        wSv = svLabelsMat.arrayTimes(svAlphasMat);
        calcMSV();
        calcW();
        calcLopsidedMargin();
    }

    @Override
    public int predict(Matrix x) {
        Matrix X = kernelTrans(svMat, x, kernelParam);//p*1
        double res = X.transpose().times(wSv).get(0, 0) + b;

        if (res > 0) return 1;
        else return -1;
    }

    /**
     * 计算间隔支持向量
     */
    private void calcMSV() {
        //统计间隔支持向量的个数
        int count = 0;
        for (int i = 0; i < svCount; i++) {
            double alpha = svAlphasMat.get(i, 0);
            double C = calcBoundC(i);

            if (alpha > 0 && alpha < C) {
                //获得间隔支持向量
                count++;
            }
        }

        msvCount = count;
        //初始化存放信息的矩阵
        msvMat = new Matrix(msvCount, featureCount);
        msvLabelsMat = new Matrix(msvCount, 1);
        msvAlphasMat = new Matrix(msvCount, 1);

        int msvSampleCount = 0;
        for (int i = 0; i < sampleCount; i++) {
            double C = calcBoundC(i);
            double alphaI = alphas.get(i, 0);
            if (alphaI > 0 && alphaI < C) {
                //此时为间隔支持向量
                msvMat.setMatrix(msvSampleCount, msvSampleCount, 0, featureCount - 1,
                        data.getMatrix(i, i, 0, featureCount - 1));
                msvLabelsMat.set(msvSampleCount, 0, labels.get(i, 0));
                msvAlphasMat.set(msvSampleCount, 0, alphas.get(i, 0));
                msvSampleCount++;
            }
        }
    }

    /**
     * 通过计算边界漂移来修正决策边界
     */
    private void calcLopsidedMargin() {
        double bp = 0;
        double bn = 0;
        int bpCount = 0;
        int bnCount = 0;

        for (int i = 0; i < msvCount; i++) {
            double label = msvLabelsMat.get(i, 0);
            Matrix x = msvMat.getMatrix(i, i, 0, featureCount - 1);
            if (label == 1) {
                //正类
                bpCount++;
                bp += 1 / Ap - calcFxi(x);
            } else {
                //负类
                bnCount++;
                bn += -An - -calcFxi(x);
            }
        }

        bp /= (bpCount * 1.0);
        bn /= (bnCount * 1.0);

        b = (Ap * bp + An * bn) / (Ap + An);
    }

    private double calcFxi(Matrix x) {
        Matrix X = kernelTrans(svMat, x, kernelParam);//p*1
        return X.transpose().times(wSv).get(0, 0);
    }

    public static void main(String[] args) {
        double rate = 1.0;
        String trainFileName = "data/en_fa_" + rate + "_" + 0;
        String testFileName = "data/en_fa_" + rate + "_" + 1;

        HashMap<String, Matrix> trainMap = SvmUtil.loadSet(trainFileName, " ");
        Matrix data = trainMap.get("data");//m*n
        Matrix labels = trainMap.get("labels").transpose();//m*1

        MarginCalibrationSvm svm = new MarginCalibrationSvm(data, labels);
        //cp cn rbf
        svm.train(100, 20, 20, 0.1, 0.1, 0.2);//10 - 0.1

        HashMap<String, Matrix> testMap = SvmUtil.loadSet(testFileName, " ");
        Matrix x = testMap.get("data");//m*n
        Matrix y = testMap.get("labels").transpose();//m*1
        Map<String, Integer> testMap1 = svm.evaluate(x, y);

        System.out.println(testMap1.toString());
        SvmUtil.showSvmPerformance(testMap1);
    }
}
