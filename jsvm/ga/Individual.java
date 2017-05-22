package jsvm.ga;


import java.util.Arrays;

/**
 * Created by zjr on 2017/4/29.
 */
public class Individual {
    //表示每一个待优化参数的上限
    public static double[] upLimit;
    //每一个待优化参数的下限
    public static double[] lowLimit;

    //染色体
    private double[] chromosome;

    //适应度
    private double fitness = -1;

    public Individual(double[] chromosome) {
        this.chromosome = chromosome;
    }

    /**
     * 初始化染色体 - 随机
     *
     * @param chromosomeLength 待优化染色体的长度-基因的个数
     */
    public Individual(int chromosomeLength) {
        /*
         if (lowLimit == null || upLimit == null) {
         throw new Exception("Please input limit!");
         }

         if (lowLimit.length != chromosomeLength || upLimit.length != chromosomeLength) {
         throw new Exception("Please input right limit!");
         }

         for (int i = 0; i < chromosomeLength; i++) {
         if (lowLimit[i] > upLimit[i]) {
         throw new Exception("UpLimit must be greater than lowLimit!");
         }
         }
         */
        this.chromosome = new double[chromosomeLength];
        for (int i = 0; i < chromosomeLength; i++) {
            double width = upLimit[i] - lowLimit[i];
            setGene(i, lowLimit[i] + Math.random() * width);
        }
    }

    public double[] getChromosome() {
        return chromosome;
    }

    public double getGene(int offset) {
        return chromosome[offset];
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public double getFitness() {
        return this.fitness;
    }

    public String toString() {
        String val = Arrays.toString(chromosome);
        return "Fitness : " + getFitness() + "Chromosome : " + val;
    }

    public int getChromosomeLength() {
        return chromosome.length;
    }

    public void setGene(int offset, double gene) {
        this.chromosome[offset] = gene;
    }
}
