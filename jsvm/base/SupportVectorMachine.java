package jsvm.base;

import Jama.Matrix;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;

/**
 * Created by zjr on 2017/5/7.
 */

public abstract class SupportVectorMachine {
    //用于存放训练样本集 - m*n
    protected Matrix data;
    //用于存放样本集对应的标签集 - m*1
    protected Matrix labels;

    //样本的个数 - m
    protected int sampleCount;
    //特征的个数 - n
    protected int featureCount;
    //正类和父类的比例
    protected double rate;

    //拉格朗日算子 - m*1
    protected Matrix alphas;
    //b
    protected double b;
    //w - n*1
    protected Matrix w;

    //SVM-SMO算法精确度
    protected double tolerance;
    //缓存，用于存放计算过的偏差 m*2 flag : value
    protected Matrix ECache;
    //矩阵K - m*m
    protected Matrix K;
    //Sv - count
    protected int svCount;
    //Sv - p*n
    protected Matrix svMat;
    //Svlabels - p*1
    protected Matrix svLabelsMat;
    //SvAlphas - p*1
    protected Matrix svAlphasMat;

    public SupportVectorMachine(Matrix data, Matrix labels) {
        this.data = data;
        this.labels = labels;
        this.sampleCount = data.getRowDimension();
        this.featureCount = data.getColumnDimension();

        initSvm();
    }

    /**
     * 初始化支持向量机
     */
    public void initSvm() {
        this.alphas = new Matrix(sampleCount, 1, 0);//m*1
        this.w = new Matrix(featureCount, 1);//n*1
        this.b = 0;
        this.tolerance = 0.0001;
        this.ECache = new Matrix(sampleCount, 2, 0);
        this.K = new Matrix(sampleCount, sampleCount);
    }

    /**
     * 限制alpha的范围
     *
     * @param alpha alpha
     * @param low   下限
     * @param hi    上限
     * @return 修正后的alpha
     */
    protected double limitAlpha(double alpha, double low, double hi) {
        double res;
        if (alpha > hi) res = hi;
        else if (alpha < low) res = low;
        else res = alpha;
        return res;
    }

    /**
     * 随机选择另一个样本 J
     *
     * @param i 待优化的样本I
     * @return J
     */
    protected int selectRandomJ(int i) {
        Random random = new Random(System.currentTimeMillis());
        int j = random.nextInt(sampleCount);
        while (j == i) {
            j = random.nextInt(sampleCount);
        }
        return j;
    }

    /**
     * 计算第i个样本的估计偏差
     *
     * @param i 样本序号
     * @return 偏差
     */
    protected double calcEk(int i) {
        Matrix x = alphas.arrayTimes(labels).transpose();//1*m
        Matrix Ki = K.getMatrix(0, sampleCount - 1, i, i);//K矩阵的第i列
        double f_Xk = x.times(Ki).get(0, 0) + b;
        return f_Xk - labels.get(i, 0);
    }

    /**
     * 更新误差缓存
     *
     * @param i 样本序号
     */
    protected void updateECache(int i) {
        double Ei = calcEk(i);
        ECache.set(i, 0, 1);
        ECache.set(i, 1, Ei);
    }

    /**
     * 为i选择另一个待优化的样本序号j
     *
     * @param i  i
     * @param Ei i的偏差
     * @return j
     */
    private int selectJ(int i, double Ei) {
        int maxIdx = -1;
        double maxDeltaE = -1;

        ECache.set(i, 0, 1);
        ECache.set(i, 1, Ei);

        LinkedList<Integer> validECacheList = new LinkedList<>();

        for (int k = 0; k < sampleCount; k++) {
            if (ECache.get(k, 0) == 1) {
                validECacheList.add(k);
            }
        }

        //validECacheList中序号为i的样本为计算过的
        //因此需要validECacheList的长度大于1
        if (validECacheList.size() > 1) {
            for (Integer idx : validECacheList) {
                if (idx == i) continue;
                double Ek = calcEk(idx);
                double deltaE = Math.abs(Ei - Ek);
                if (deltaE > maxDeltaE) {
                    maxDeltaE = deltaE;
                    maxIdx = idx;
                }
            }
            //存在一个步长最大的j
            return maxIdx;
        } else {
            return selectRandomJ(i);
        }
    }

    /**
     * 计算导数
     *
     * @param i i
     * @param j j
     * @return 导数
     */
    protected double calcEta(int i, int j) {
        return 2 * K.get(i, j) - K.get(i, i) - K.get(j, j);
    }


    /**
     * @param i 样本序号
     * @return 边界参数C
     */
    protected abstract double calcBoundC(int i);

    /**
     * 优化拉格朗日算子的值 i - j
     *
     * @param i 样本序号
     * @return 更新算子对的个数
     */
    protected int optAlphaPairs(int i) {
        double Ei = calcEk(i);
        //此时开始判断KKT条件，因为platt的SMO算法考虑了只边界内的alpha，因此分为以下几种情况
        // ui 为预测值 ui = sum(alphak*labelk*K(k,j))
        // alpha > 0 此时为支持向量，但是yi*ui > 1 <-> yi*ei > 0，实际上为正确分类的样本
        // alpha < C 此时为支持向量，但是yi*ui < 1 <-> yi*ei < 0, 实际上为加入松弛变量的样本
        // yi*ei = yi*(ui - yi) = yi*ui - 1 因此只需要判断yi*ei的正负即可
        //在一定精度内满足条件即可
        double alphaIOld = alphas.get(i, 0);
        double labelI = labels.get(i, 0);
        double YiEi = labelI * Ei;

        //判断KKT之前，需要计算alpha的边界
        //软间隔的c为C
        //代价敏感的c = C+ | C-
        //间隔校正的c = (C+*A+) | (C-*A-)
        double Ci = calcBoundC(i);

        //判断KKT条件是否满足
        if ((YiEi > tolerance && alphaIOld > 0) || (YiEi < -tolerance && alphaIOld < Ci)) {
            //样本i对应的拉格朗日算子不符合KKT条件
            //选择此时开始将工作集大小限定为2
            int j = selectJ(i, Ei);
            double Ej = calcEk(j);
            double labelJ = labels.get(j, 0);
            double alphaJOld = alphas.get(j, 0);
            double Cj = calcBoundC(j);

            //确定alphaJ的上下界
            double L, H;
            double s = labelI * labelJ;

            if (s == -1) {
                //yi != yj
                L = Math.max(0, alphaJOld - alphaIOld);
                H = Math.min(Cj, Ci + alphaJOld - alphaIOld);
            } else {
                //yi == yj
                //Ci == Cj
                L = Math.max(0, alphaJOld + alphaIOld - Ci);
                H = Math.min(Ci, alphaJOld + alphaIOld);
            }

            //此时alpha不能被优化
            if (L == H) return 0;

            //确定下来的两个样本的alpha在一条直线上
            //此时只需要将一个变量固定，便成为了单变量的线性规划问题
            double eta = calcEta(i, j);

            //不存在极大值
            if (eta >= 0) return 0;

            double alphaINew;
            double alphaJNew;


            if (eta == 0) {
                //目标函数由凸函数退化直线
                //W(alphaJ) = yj*(Ej-Ei)*alphaJ+const
                //带入alphaJ的两个端点L/H计算取最小值
                double p = labelJ * (Ej - Ei) * L;
                double q = labelJ * (Ej - Ei) * H;

                alphaJNew = p < q ? L : H;

            } else {
                //eta < 0
                alphaJNew = alphaJOld + labelJ * (Ej - Ei) / eta;
                alphaJNew = limitAlpha(alphaJNew, L, H);
            }

            //更新alpha及缓存Ek
            alphas.set(j, 0, alphaJNew);
            updateECache(j);

            //优化量太小
            if (Math.abs(alphaJNew - alphaJOld) < 0.00001) return 0;

            alphaINew = alphaIOld + (alphaJOld - alphaJNew) * s;
            alphas.set(i, 0, alphaINew);
            updateECache(i);

            double deltaI = alphaINew - alphaIOld;
            double deltaJ = alphaJNew - alphaJOld;

            //double b1 = b - (Ei + deltaI * labelI * K.get(i, i) + deltaJ * labelJ * K.get(j, i))
            double b1 = b - Ei - deltaI * labelI * K.get(i, i) - deltaJ * labelJ * K.get(i, j);
            double b2 = b - Ej - deltaI * labelI * K.get(i, j) - deltaJ * labelJ * K.get(j, j);

            //优化后的b由不在边界上的算子决定
            if (alphaINew > 0 && alphaINew < Ci) {
                b = b1;
            } else if (alphaJNew > 0 && alphaJNew < Cj) {
                b = b2;
            } else {
                b = (b1 + b2) / 2.0;
            }

            //此时完成了alpha-pairs以及b的更新
            return 1;
        } else {
            //满足KKT条件
            return 0;
        }
    }

    /**
     * platt Smo 算法
     */
    protected void smo(int maxIteration) {
        boolean entireSet = true;
        int alphaPairsChanged = 0;

        int iter = 0;
        while (entireSet || alphaPairsChanged > 0) {
            alphaPairsChanged = 0;
            if (entireSet) {
                //全集合遍历优化
                for (int i = 0; i < sampleCount; i++) {
                    alphaPairsChanged += optAlphaPairs(i);
                }
            } else {
                //边界内算子优化
                for (int i = 0; i < sampleCount; i++) {
                    double alpha = alphas.get(i, 0);
                    if (alpha > 0 && alpha < calcBoundC(i)) {
                        //边界算子
                        alphaPairsChanged += optAlphaPairs(i);
                    }
                }
            }

            //如果此次优化为全集合优化，则下一次不为全集合优化
            if (entireSet) entireSet = false;
                //如果此次优化不为全集合优化，且没有alpha算子更新，则进行全集合优化
            else if (alphaPairsChanged == 0) entireSet = true;

            //终止条件： 如果此次优化为全集合优化，且在优化中没有alpha算子更新，则SMO算法终止
            iter++;
            if (iter >= maxIteration) break;
        }

        System.out.println("训练结束，迭代次数：" + iter);
    }

    /**
     * 计算SV
     */
    protected void calcSV() {
        svCount = 0;
        for (int i = 0; i < sampleCount; i++) {
            if (alphas.get(i, 0) > 0) {
                svCount++;
            }
        }

        svMat = new Matrix(svCount, featureCount);
        svLabelsMat = new Matrix(svCount, 1);
        svAlphasMat = new Matrix(svCount, 1);

        int svSampleCount = 0;
        for (int i = 0; i < sampleCount; i++) {
            double C = calcBoundC(i);
            double alphaI = alphas.get(i, 0);
            if (alphaI > 0) {
                //此时为支持向量
                svMat.setMatrix(svSampleCount, svSampleCount, 0, featureCount - 1,
                        data.getMatrix(i, i, 0, featureCount - 1));
                svLabelsMat.set(svSampleCount, 0, labels.get(i, 0));
                svAlphasMat.set(svSampleCount, 0, alphas.get(i, 0));
                svSampleCount++;
            }
        }
    }

    /**
     * 计算w
     */
    protected void calcW() {
        for (int i = 0; i < sampleCount; i++) {
            // 1*m -> m*1
            Matrix wi = data.getMatrix(i, i, 0, featureCount - 1).times(
                    alphas.get(i, 0) * labels.get(i, 0)).transpose();

            w.plusEquals(wi);
        }
    }

    /**
     * 训练
     *
     * @param param 所需要的参数
     */
    public abstract void train(int maxIteration, double... param);


    /**
     * @return TP TN FP FN ---> Map
     */
    public HashMap<String, Integer> evaluateSelf() {
        return this.evaluate(data, labels);
    }

    /**
     * 评估
     *
     * @param x 输入
     * @param y 标签
     * @return TP TN FP FN
     */
    public HashMap<String, Integer> evaluate(Matrix x, Matrix y) {
        HashMap<String, Integer> map = new HashMap<>();

        //样本个数
        int m = x.getRowDimension();
        //特征个数
        int n = x.getColumnDimension();

        int positive_count = 0;
        int negative_count = 0;

        int error_positive = 0;
        int error_negative = 0;

        for (int i = 0; i < m; i++) {
            double labelI = y.get(i, 0);
            if (labelI == 1) {
                positive_count++;
            } else {
                negative_count++;
            }

            int res = this.predict(x.getMatrix(i, i, 0, n - 1));

            if (res != labelI) {
                if (labelI == 1) {
                    error_positive++;
                } else {
                    error_negative++;
                }
            }
        }

        rate = (positive_count + 1) / (negative_count + 1);

        map.put("TP", (positive_count - error_positive));
        map.put("TN", (negative_count - error_negative));
        map.put("FP", (error_negative));
        map.put("FN", (error_positive));

        return map;
    }

    /**
     * 预测某个样本
     *
     * @param x 样本
     * @return 标签
     */
    public abstract int predict(Matrix x);
}
