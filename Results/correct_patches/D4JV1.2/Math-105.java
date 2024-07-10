public double getSumSquaredErrors() {
    if (sumXX == 0.0) {
        return Double.NaN;
    }
    return Math.max(0, sumYY - ((sumXY * sumXY) / sumXX));
}