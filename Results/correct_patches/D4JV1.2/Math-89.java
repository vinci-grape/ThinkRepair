public void addValue(Object v) {
    if (v instanceof Comparable<?>) {
        addValue((Comparable<?>) v);
    } else {
        throw new IllegalArgumentException("v must be comparable");
    }
}