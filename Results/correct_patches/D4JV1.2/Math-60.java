public double cumulativeProbability(double x) throws MathException {
    if (x <= Double.NEGATIVE_INFINITY) {
        return 0.0;
    } else if (x >= Double.POSITIVE_INFINITY) {
        return 1.0;
    } else if (FastMath.abs(x - mean) > 40 * standardDeviation) {
        if (x > mean) {
            return 1.0;
        } else {
            return 0.0;
        }
    } else {
        final double dev = x - mean;
        try {
            return 0.5 * (1.0 + Erf.erf((dev) /
                    (standardDeviation * FastMath.sqrt(2.0))));
        } catch (MaxIterationsExceededException ex) {
            throw new MathException("Error computing cumulative probability: convergence failure", ex);
        }
    }
}