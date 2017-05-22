package jsvm;

import Jama.Matrix;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * Created by zjr on 2017/5/5.
 */
public class SvmUtil {
    /**
     * @param fileName  fileName
     * @param splitChar splitChar
     * @return {data : Matrix data,labels(m*n) : Matrix labels(1*m)}
     */
    public static HashMap<String, Matrix> loadSet(String fileName, String splitChar) {
        HashMap<String, Matrix> map = new HashMap();
        LinkedList<double[]> dataList = new LinkedList<>();
        LinkedList<Double> labelList = new LinkedList<>();

        int sampleCount = 0;
        int featureCount = 0;

        try (
                FileInputStream fis = new FileInputStream(fileName);
                InputStreamReader isr = new InputStreamReader(fis);
                BufferedReader bis = new BufferedReader(isr)
        ) {
            String line = bis.readLine().trim();

            while (line != null) {
                String[] element = line.trim().split(splitChar);
                int length = element.length;
                sampleCount += 1;
                featureCount = length - 1;

                double[] sampleData = new double[length - 1];
                for (int i = 0; i < length - 1; i++) {
                    sampleData[i] = Double.valueOf(element[i]);
                }
                dataList.add(sampleData);
                labelList.add(Double.valueOf(element[length - 1]));

                line = bis.readLine();
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
            e.printStackTrace();
        }

        double[][] data = new double[sampleCount][featureCount];
        double[] labels = new double[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            data[i] = dataList.get(i);
            labels[i] = labelList.get(i);
        }

        map.put("data", new Matrix(data));//m*n
        map.put("labels", new Matrix(labels, 1));//1*m
        return map;
    }

    /**
     * 将数据储存到硬盘
     *
     * @param map      map
     * @param filePath path
     */
    public static void saveDataToDisk(HashMap<String, Matrix> map, String filePath) {
        Matrix data = map.get("data");
        Matrix labels = map.get("labels");

        File file = new File(filePath);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                System.out.println(e.getMessage());
                e.printStackTrace();
            }
        }

        int m = data.getRowDimension();
        int n = data.getColumnDimension();

        try (FileWriter fwr = new FileWriter(file)) {
            for (int i = 0; i < m; i++) {
                String line = Arrays.toString(data.getMatrix(i, i, 0, n - 1).getArray()[0]);
                line = line.substring(1, line.length() - 1) + "," + labels.get(0, i) + "\r\n";
                fwr.write(line);
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 对发动机的数据进行处理
     *
     * @param map map
     * @return map
     */
    public static HashMap<String, Matrix> dataFilter(HashMap<String, Matrix> map) {
        Matrix data = map.get("data");
        Matrix labels = map.get("labels");

        //0 1 2 3   4   5   6   7    8     9    10   11
        //0 1 2 2/1 5/3 6/4 9/7 10/8 11/10 12/9 13/1 14
        int m = data.getRowDimension();
        int n = data.getColumnDimension();

        Matrix newData = new Matrix(m, 12);
        //0-油门杆角度、1-风扇转速、2-压气机转速
        newData.setMatrix(0, m - 1, 0, 2, data.getMatrix(0, m - 1, 0, 2));

        //3-压气机转速/风扇转速
        newData.setMatrix(0, m - 1, 3, 3,
                data.getMatrix(0, m - 1, 2, 2).arrayRightDivide(
                        data.getMatrix(0, m - 1, 1, 1)
                ));

        //4- 5/3
        newData.setMatrix(0, m - 1, 4, 4,
                data.getMatrix(0, m - 1, 5, 5).arrayRightDivide(
                        data.getMatrix(0, m - 1, 3, 3)
                ));

        //5- 6/4
        newData.setMatrix(0, m - 1, 5, 5,
                data.getMatrix(0, m - 1, 6, 6).arrayRightDivide(
                        data.getMatrix(0, m - 1, 4, 4)
                ));

        //6- 9/7
        newData.setMatrix(0, m - 1, 6, 6,
                data.getMatrix(0, m - 1, 9, 9).arrayRightDivide(
                        data.getMatrix(0, m - 1, 7, 7)
                ));

        //7- 10/8
        newData.setMatrix(0, m - 1, 7, 7,
                data.getMatrix(0, m - 1, 10, 10).arrayRightDivide(
                        data.getMatrix(0, m - 1, 8, 8)
                ));

        //8- 11/10
        newData.setMatrix(0, m - 1, 8, 8,
                data.getMatrix(0, m - 1, 11, 11).arrayRightDivide(
                        data.getMatrix(0, m - 1, 10, 10)
                ));

        //9- 12/9
        newData.setMatrix(0, m - 1, 9, 9,
                data.getMatrix(0, m - 1, 12, 12).arrayRightDivide(
                        data.getMatrix(0, m - 1, 9, 9)
                ));

        //10- 13/1
        newData.setMatrix(0, m - 1, 10, 10,
                data.getMatrix(0, m - 1, 13, 13).arrayRightDivide(
                        data.getMatrix(0, m - 1, 1, 1)
                ));

        //11- 14
        newData.setMatrix(0, m - 1, 11, 11,
                data.getMatrix(0, m - 1, 14, 14));

        boolean[] sampleAvailable = new boolean[m];
        int count = 0;
        for (int i = 0; i < m; i++) {
            Matrix x = newData.getMatrix(i, i, 0, 11);
            boolean flag = SvmUtil.isSampleAvailable(x);
            sampleAvailable[i] = flag;
            if (flag) count++;
            else {
                System.out.println("Unavailable :" + i);
                x.print(x.getColumnDimension(), 5);
            }
        }

        Matrix res = new Matrix(count, 12);
        Matrix res_labels = new Matrix(1, count);
        count = 0;
        for (int i = 0; i < m; i++) {
            if (sampleAvailable[i]) {
                Matrix x = newData.getMatrix(i, i, 0, 11);
                res.setMatrix(count, count, 0, 11, x);
                res_labels.set(0, count, labels.get(0, i));
                count++;
            }
        }

        map.put("data", res);
        map.put("labels", res_labels);
        return map;
    }

    /**
     * 检查样本中的数据是否都合法
     *
     * @param x x
     * @return 是否合法
     */
    private static boolean isSampleAvailable(Matrix x) {
        double[] arr = x.getArray()[0];
        for (double ele : arr) {
            if (Double.isNaN(ele) || Double.isInfinite(ele))
                return false;
        }
        return true;
    }

    public static void showSvmPerformance(Map<String, Integer> map) {
        int TP = map.get("TP");
        int TN = map.get("TN");
        int FP = map.get("FP");
        int FN = map.get("FN");

        //精确率
        double P = TP * 1.0 / (TP + FP);

        //召回率
        double R = TP * 1.0 / (TP + FN);

        //F1
        double PR = 2 * P * R / (P + R);

        System.out.println("精确率 、 召回率 、 F1值分别为：");
        System.out.println(P + " " + R + " " + PR);
    }
}