public int compareTo(Fraction object) {
    long nOq = numerator * object.denominator;
    long dOn = denominator * object.numerator;
    return Long.compare(nOq, dOn);
}