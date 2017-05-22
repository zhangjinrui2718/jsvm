package jsvm.ga;


import java.util.*;

/**
 * 总体思路为：
 * -排序
 * -交叉
 * -变异
 * -评估
 * -->下一代
 * Created by zjr on 2017/4/29.
 */
public abstract class GeneticAlgorithm {
    //种群大小
    private int populationSize;
    //变异概率
    private double mutationRate;
    //交配概率
    private double crossoverRate;
    //交配参数
    private double r;
    //变异参数
    private double k;
    //精英个数
    private int elitismCount;

    public GeneticAlgorithm(int populationSize, double mutationRate, double crossoverRate,
                            double r, double k, int elitismCount) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.crossoverRate = crossoverRate;
        this.r = r;
        this.k = k;
        this.elitismCount = elitismCount;
    }

    /**
     * 初始化种群
     *
     * @param chromosomeLength 待优化参数的个数
     * @return 随机初始化后的种群
     */
    public Population initPopulation(int chromosomeLength) {
        return new Population(this.populationSize, chromosomeLength);
    }


    /**
     * 评估染色体的适应度
     *
     * @param individual 染色体
     * @return 适应度
     */
    public abstract double calcFitness(Individual individual);


    /**
     * 评估终止条件
     *
     * @param population 待评估种群
     * @return 终止标志
     */
    public abstract boolean isTerminationConditionMet(Population population);

    /**
     * 评估种群
     *
     * @param population 待评估种群
     */
    public void evalPopulation(Population population) {
        double populationFitness = 0;
        for (Individual individual : population.getIndividuals()) {
            populationFitness += calcFitness(individual);
        }
        population.setPopulationFitness(populationFitness);
    }

    /**
     * 根据适应度对中群内的染色体进行排序
     *
     * @param population 需要排序的种群
     */
    public void sortPopulation(Population population) {
        Arrays.sort(population.getIndividuals(), (o1, o2) -> {
            if (o1.getFitness() > o2.getFitness()) {
                return -1;
            } else if (o1.getFitness() < o2.getFitness()) {
                return 1;
            }
            return 0;
        });
    }

    /**
     * 选择交配的父代
     * 轮盘赌注
     *
     * @param population 父代所在的种群
     * @return 选择出来的父代
     */
    private Individual selectParent(Population population) {
        Individual individuals[] = population.getIndividuals();
        double populationFitness = population.getPopulationFitness();

        //设置轮盘的位置
        double rouletteWheelPosition = Math.random() * populationFitness;
        double spinWheel = 0;
        for (Individual individual : individuals) {
            spinWheel += individual.getFitness();
            if (spinWheel >= rouletteWheelPosition) {
                return individual;
            }
        }
        return individuals[population.size() - 1];
    }

    /**
     * 选择交配的父代
     * 轮盘赌注
     *
     * @param population 父代所在的种群(该种群是残余种群)
     * @return 选择出来的父代
     */
    private Individual selectParent(List<Individual> population) {
        double populationFitness = population.stream()
                .map(Individual::getFitness)
                .reduce(.0, (o1, o2) -> o1 + o2);

        //设置轮盘的位置
        double rouletteWheelPosition = Math.random() * populationFitness;
        double spinWheel = 0;
        for (Individual individual : population) {
            spinWheel += individual.getFitness();
            if (spinWheel >= rouletteWheelPosition) {
                return individual;
            }
        }
        return population.get(population.size() - 1);
    }

    /**
     * 对种群内进行交配操作
     * 一次更新两条染色体
     * a′=(1 -α)· a +βb ,
     * b′=(1 -β)·b +αa , 0 < α, β <r
     * if a′(b′)<L then a′(b′)=L
     * if a′(b′)>R then a′(b′)=R
     * <p>
     * r为参数
     * α/β为随机数
     *
     * @param population 种群
     * @return 交配后的种群
     */
    public Population crossoverPopulation(Population population) {
        //进入交配环节时
        //population总的染色体保持整体有序
        //System.out.println("开始交配");

        Population newPopulation = new Population(population.size());

        List<Individual> list = new LinkedList<>();
        list.addAll(Arrays.asList(population.getIndividuals()));

        int idx = 0;
        for (int i = 0; i < elitismCount; i++) {
            //把精英直接放在新种群中
            newPopulation.setIndividual(idx++, list.get(i));
            list.remove(i);
        }

        while (!list.isEmpty()) {
            Individual person1 = list.get(0);
            list.remove(0);

            //如果list中没有染色体了
            if (list.size() == 0) {
                newPopulation.setIndividual(idx, person1);
                break;
            }

            //不需要交配
            if (Math.random() > crossoverRate) {
                newPopulation.setIndividual(idx++, person1);
                continue;
            }

            //在剩下的里面挑出一个亲代进行交配
            Individual person2 = selectParent(list);
            list.remove(person2);

            //两个子代
            Individual son1 = new Individual(person1.getChromosomeLength());
            Individual son2 = new Individual(person2.getChromosomeLength());

            //进行交叉操作
            for (int geneIndex = 0; geneIndex < person1.getChromosomeLength(); geneIndex++) {
                double a = person1.getGene(geneIndex);
                double b = person2.getGene(geneIndex);

                //计算随机算子
                double alpha = Math.random() * r;
                double beta = Math.random() * r;

                //计算交叉后的值
                double p1 = (1 - alpha) * a + beta * b;
                double p2 = alpha * a + (1 - beta) * b;

                //对交叉后的值进行限幅
                p1 = limit(p1, geneIndex);
                p2 = limit(p2, geneIndex);

                son1.setGene(geneIndex, p1);
                son2.setGene(geneIndex, p2);
            }

            newPopulation.setIndividual(idx++, son1);
            newPopulation.setIndividual(idx++, son2);
        }

//        System.out.println("交配完毕，新的种群规模是" + newPopulation.size());
//        Arrays.stream(newPopulation.getIndividuals()).forEach(System.out::println);
        return newPopulation;
    }

    /**
     * 交叉或者变异后的数值要在限定的范围内
     *
     * @param value 基因数值
     * @param idx   基因序号
     * @return 合法值
     */
    private double limit(double value, int idx) {
        double res = value;
        if (value > Individual.upLimit[idx]) {
            res = Individual.upLimit[idx];
        } else if (value < Individual.lowLimit[idx]) {
            res = Individual.lowLimit[idx];
        }
        return res;
    }

    /**
     * 变异
     * c′=
     * c +k ·(R -c)· γ, random(2)=0
     * c -k ·(c -L)· γ, random(2)=1
     * 其中R为上限，L为下限
     * k是系数0<k<1
     * γ是随机数0<γ<1
     *
     * @param population 需要变异操作的种群
     * @return 变异后的种群
     */
    public Population mutatePopulation(Population population) {
//        System.out.println("开始变异");

        Population newPopulation = new Population(this.populationSize);

        for (int populationIndex = 0; populationIndex < population.size(); populationIndex++) {
            Individual individual = population.getIndividual(populationIndex);

            //如果是精英，不需要变异直接跳过
            if (populationIndex < elitismCount) {
                newPopulation.setIndividual(populationIndex, individual);
            }

            //获得染色体
            double[] chromosome = individual.getChromosome();
            for (int i = 0; i < chromosome.length; i++) {
                if (Math.random() < mutationRate) {
                    //该位基因需要变异
                    double gamma = Math.random();
                    double gene = chromosome[i];
                    if (Math.random() > 0.5) {
                        chromosome[i] = gene + k * (Individual.upLimit[i] - gene) * gamma;
                    } else {
                        chromosome[i] = gene - k * (gene - Individual.lowLimit[i]) * gamma;
                    }
                }
            }

            newPopulation.setIndividual(populationIndex, population.getIndividual(populationIndex));
        }
//        System.out.println("变异完毕，新的种群规模是" + newPopulation.size());
//        Arrays.stream(newPopulation.getIndividuals()).forEach(System.out::println);
        return newPopulation;
    }
}
