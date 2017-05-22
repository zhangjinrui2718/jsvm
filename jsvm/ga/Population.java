package jsvm.ga;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Created by zjr on 2017/4/29.
 */
public class Population {
    private Individual population[];
    private double populationFitness = -1;

    public Population(int populationSize) {
        this.population = new Individual[populationSize];
    }

    /**
     * 随机初始化的种群
     *
     * @param populationSize   种群大小
     * @param chromosomeLength 基因长度
     */
    public Population(int populationSize, int chromosomeLength) {
        this.population = new Individual[populationSize];

        for (int individualCount = 0; individualCount < populationSize; individualCount++) {
            Individual individual = null;
            try {
                individual = new Individual(chromosomeLength);
            } catch (Exception e) {
                System.out.println(e.getMessage());
                e.printStackTrace();
            }
            this.population[individualCount] = individual;
        }
    }

    public Individual[] getIndividuals() {
        return this.population;
    }

    public void setPopulationFitness(double fitness) {
        this.populationFitness = fitness;
    }

    public double getPopulationFitness() {
        return this.populationFitness;
    }

    public int size() {
        return this.population.length;
    }

    public Individual setIndividual(int offset, Individual individual) {
        return population[offset] = individual;
    }

    public Individual getIndividual(int offset) {
        return population[offset];
    }
}
