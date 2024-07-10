public double getNumericalMean() {
    double sampleSize = (double) getSampleSize();
    double numSuccesses = (double) getNumberOfSuccesses();
    double popSize = (double) getPopulationSize();
    return (sampleSize * numSuccesses) / popSize;
}