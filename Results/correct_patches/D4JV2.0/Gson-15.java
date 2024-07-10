public JsonWriter value(double value) throws IOException {
    if (!lenient) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            // Check for NaN or infinity
            throw new IllegalArgumentException("Numeric values must be finite when lenient is disabled, but was " + value);
        }
    }
    writeDeferredName();
    beforeValue();
    out.append(Double.toString(value));
    return this;
}